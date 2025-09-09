# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import os
import json
import subprocess
import logging
from typing import Dict, Any

# Add deterministic seed for reproducibility
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
    TaskType,
    utils
)
from PIL import Image
import nvflare.client as flare

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class InstagramCaptionDataset(Dataset):
    """Dataset class for Instagram image-caption pairs using Hugging Face format"""
    
    def __init__(self, data_dir, processor, max_length=512, image_size=512):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        
        # Load dataset metadata
        self.data = self._load_dataset()
        logger.info(f"Loaded {len(self.data)} training samples")
    
    def _load_dataset(self):
        """Load dataset from directory structure or metadata file"""
        data = []
        
        # Check for metadata file first
        metadata_file = os.path.join(self.data_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                for item in metadata:
                    if 'image_path' in item and 'caption' in item:
                        data.append({
                            'image_path': os.path.join(self.data_dir, item['image_path']),
                            'caption': item['caption']
                        })
        else:
            # Fallback: scan directory for image files and look for corresponding caption files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            for root, dirs, files in os.walk(self.data_dir):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_path = os.path.join(root, file)
                        caption_path = os.path.splitext(image_path)[0] + '.txt'
                        
                        if os.path.exists(caption_path):
                            with open(caption_path, 'r', encoding='utf-8') as f:
                                caption = f.read().strip()
                                if caption:
                                    data.append({
                                        'image_path': image_path,
                                        'caption': caption
                                    })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_rows(self):
        """Required by HuggingFace Trainer"""
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Load and preprocess image
            image = Image.open(item['image_path']).convert('RGB')
            
            # Resize image if needed
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            
            # Create conversation format for training
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Generate an engaging Instagram caption for this image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": item['caption']
                }
            ]
            
            # Process inputs - first get the text, then tokenize it
            conversation_text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                return_tensors=None  # Get text first
            )
            
            # Now tokenize the text properly
            try:
                inputs = self.processor.tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Get the input_ids
                inputs = inputs["input_ids"].squeeze(0)
                
            except Exception as e:
                logger.warning(f"Error tokenizing conversation: {e}")
                # Create a dummy tensor as fallback
                inputs = torch.zeros(self.max_length, dtype=torch.long)
            
            # Truncate if too long
            if len(inputs) > self.max_length:
                inputs = inputs[:self.max_length]
            
            return {
                'input_ids': inputs,
                'attention_mask': torch.ones_like(inputs),
                'labels': inputs.clone()
            }
            
        except Exception as e:
            logger.warning(f"Error processing item {idx}: {e}")
            # Return a dummy item with proper error handling
            try:
                dummy_input = torch.zeros(self.max_length, dtype=torch.long)
                return {
                    'input_ids': dummy_input,
                    'attention_mask': torch.ones_like(dummy_input),
                    'labels': dummy_input.clone()
                }
            except Exception as fallback_error:
                logger.error(f"Critical error creating dummy item for {idx}: {fallback_error}")
                # Return minimal valid item
                return {
                    'input_ids': torch.tensor([0], dtype=torch.long),
                    'attention_mask': torch.tensor([1], dtype=torch.long),
                    'labels': torch.tensor([0], dtype=torch.long)
                }

def download_model_if_needed(model_name_or_path, download_model=False):
    """Download the LFM2-VL model if needed"""
    if download_model and not os.path.exists(model_name_or_path):
        logger.info(f"Model not found at {model_name_or_path}, downloading...")
        download_script = "save_lfm2_vl_model.py"
        if os.path.exists(download_script):
            try:
                logger.info("Running model download script...")
                result = subprocess.run([
                    "python", download_script
                ], capture_output=True, text=True, check=True)
                logger.info("Model download completed successfully!")
                logger.info(f"Download output: {result.stdout}")
                if os.path.exists(model_name_or_path):
                    logger.info(f"Model verified at {model_name_or_path}")
                else:
                    logger.warning(f"Model download completed but not found at {model_name_or_path}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Model download failed: {e}")
                logger.error(f"Error output: {e.stderr}")
                raise
        else:
            logger.warning(f"Download script {download_script} not found, skipping download")
            logger.warning("Please ensure the model is available at the specified path")
    elif os.path.exists(model_name_or_path):
        logger.info(f"Model found at {model_name_or_path}")
    else:
        logger.warning(f"Model not found at {model_name_or_path} and download not requested")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/franky/LiquidTraining/lfm2_vl_1_6b_model",
    )
    parser.add_argument(
        "--data_path_train",
        type=str,
        default="/home/franky/LiquidTraining/processed_dataset/instagram_dataset",
    )
    parser.add_argument(
        "--data_path_valid",
        type=str,
        default="/home/franky/LiquidTraining/processed_dataset/instagram_dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./peft",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="PEFT",
        help="training mode, SFT or PEFT, default to PEFT",
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="message mode, numpy or tensor, default to numpy",
    )
    # Training hyperparameters (aligned with train_lfm2_instagram_trainer.py)
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--clean_up", type=int, default=0)
    parser.add_argument("--download_model", action="store_true", help="Download model if not found")
    args = parser.parse_args()

    # Download model if needed
    download_model_if_needed(args.model_name_or_path, args.download_model)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    # Create full dataset then split into train/val like the centralized trainer
    logger.info("Creating Instagram caption dataset...")
    full_dataset = InstagramCaptionDataset(args.data_path_train, processor)
    logger.info(f"Dataset loaded successfully: {len(full_dataset)} total posts")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    logger.info(f"Training split: {train_size} posts, Validation split: {val_size} posts")

    if val_size > 0:
        dataset_train, dataset_valid = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    else:
        dataset_train = full_dataset
        dataset_valid = None

    # Training parameters (use provided args)
    batch_size = args.batch_size
    gra_accu_steps = args.gradient_accumulation_steps
    logging_steps = max(1, int(len(dataset_train) / (20 * batch_size * gra_accu_steps)))
    logger.info(f"logging_steps: {logging_steps}")

    # Load model
    logger.info("Loading LFM2-VL model...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name_or_path,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )

    # Train mode
    if args.train_mode.lower() == "sft":
        train_mode = 0
    elif args.train_mode.lower() == "peft":
        train_mode = 1
    else:
        raise ValueError(f"Invalid train_mode: {args.train_mode}, only SFT and PEFT are supported.")

    # PEFT specific
    if train_mode:
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        # PEFT configs for LFM2-VL
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=16,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.local_epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gra_accu_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_dir=os.path.join(args.output_path, "logs"),
        logging_steps=logging_steps,
        eval_strategy="epoch" if dataset_valid else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if dataset_valid else False,
        metric_for_best_model="eval_loss" if dataset_valid else None,
        greater_is_better=False if dataset_valid else None,
        fp16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        disable_tqdm=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_valid,
        data_collator=data_collator,
    )

    # Initialize NVFlare client API
    logger.info("Initializing NVFlare client API...")
    flare.init()
    logger.info("NVFlare client API initialized successfully")

    # Train federated rounds
    logger.info("Starting federated learning rounds...")
    while flare.is_running():
        logger.info("Inside training loop - flare.is_running() is True")
        try:
            # Receive FLModel from NVFlare
            logger.info("Waiting for FL model from server...")
            input_model = flare.receive()
            curr_round = input_model.current_round
            logger.info(f"current_round={curr_round}")

            # Update the key name received from global model if using model def file
            global_model = copy.deepcopy(input_model.params)
            for key in list(global_model.keys()):
                global_model[key.replace("model.", "", 1)] = global_model.pop(key)

            # Evaluation function
            def evaluate(input_weights, mode):
                if mode:
                    set_peft_model_state_dict(trainer.model, input_weights)
                else:
                    trainer.model.load_state_dict(input_weights)
                metric_score = trainer.evaluate()
                logger.info(f"Evaluation metric score: {metric_score}")
                return metric_score

            # Evaluate on received global model only if we have a validation set
            eval_loss = None
            if dataset_valid is not None:
                eval_metrics = evaluate(global_model, train_mode)
                eval_loss = float(eval_metrics["eval_loss"]) if "eval_loss" in eval_metrics else None

            # Load global model and previous training states
            if curr_round == 0:
                # First round, start from pretrained model
                logger.info("First round - starting training from pretrained model")
                trainer.train()
            else:
                # Replace local resume weights with global weights
                logger.info(f"Round {curr_round} - loading global weights")
                resume_from_checkpoint_folder = os.path.join(args.output_path, f"checkpoint-{curr_round-1}")
                if not os.path.exists(resume_from_checkpoint_folder):
                    os.makedirs(resume_from_checkpoint_folder, exist_ok=True)
                
                if train_mode:
                    # PEFT model small, directly save via torch.save
                    resume_model_file_path = os.path.join(resume_from_checkpoint_folder, utils.WEIGHTS_NAME)
                    torch.save(global_model, resume_model_file_path)
                else:
                    # SFT model can be large, save via HF API
                    trainer.model.save_pretrained(resume_from_checkpoint_folder, safe_serialization=False)
                
                # Increment num_train_epochs so that the trainer will continue training
                if args.clean_up:
                    trainer.args.num_train_epochs = (curr_round + 1) * args.local_epoch
                else:
                    trainer.args.num_train_epochs += args.local_epoch
                logger.info(f"Increment num_train_epochs to {trainer.args.num_train_epochs}")
                
                # Continue training
                trainer.train(resume_from_checkpoint=True)

            # Compose output model to send back to server
            if train_mode:
                # PEFT, load PEFT part from trainer model
                out_param = get_peft_model_state_dict(trainer.model)
            else:
                # SFT, load whole model state_dict
                out_param = trainer.model.state_dict()

            # Update the key name sent to global model
            if not train_mode:
                for key in list(out_param.keys()):
                    out_param["model." + key] = out_param.pop(key).cpu()

            if args.message_mode.lower() == "numpy":
                # Cast out_param to float32 preparing for communication with numpy
                out_param = {k: v.to(torch.float32) for k, v in out_param.items()}

            # Construct trained FL model
            metrics_out = {"eval_loss": eval_loss} if eval_loss is not None else {}
            num_steps_current_round = len(trainer.train_dataset) if hasattr(trainer, "train_dataset") else 0
            output_model = flare.FLModel(
                params=out_param,
                metrics=metrics_out,
                meta={"NUM_STEPS_CURRENT_ROUND": num_steps_current_round},
            )
            
            # Send model back to NVFlare
            logger.info("Sending trained model back to server...")
            flare.send(output_model)
            logger.info(f"Round {curr_round} completed successfully!")

        except Exception as e:
            logger.error(f"Error in federated learning round: {e}")
            raise

    logger.info("Federated learning completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Write crash info near the working directory for quick inspection
        err_path = os.path.join(os.getcwd(), "script_error.txt")
        try:
            import traceback
            with open(err_path, "w") as f:
                f.write(str(e) + "\n\n")
                traceback.print_exc(file=f)
        except Exception:
            pass
        raise
