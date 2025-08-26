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
import random
import sys
import traceback

import numpy as np
import torch
from PIL import Image
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
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)

import nvflare.client as flare

# Set deterministic seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Add comprehensive logging
def log_info(message):
    print(f"[INFO] {message}")
    sys.stdout.flush()

def log_error(message, error=None):
    print(f"[ERROR] {message}")
    if error:
        print(f"[ERROR] Details: {error}")
        traceback.print_exc()
    sys.stdout.flush()

def log_warning(message):
    print(f"[WARNING] {message}")
    sys.stdout.flush()


class InstagramCaptionDataset(torch.utils.data.Dataset):
    """Dataset class for Instagram image-caption pairs"""
    
    def __init__(self, data_dir, processor, max_length=512, image_size=512):
        self.data_dir = data_dir
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size
        
        log_info(f"Initializing dataset from: {data_dir}")
        
        # Load dataset metadata
        try:
            self.data = self._load_dataset()
            log_info(f"Successfully loaded {len(self.data)} training samples")
        except Exception as e:
            log_error(f"Failed to load dataset from {data_dir}", e)
            raise
    
    def _load_dataset(self):
        """Load dataset from metadata file"""
        data = []
        
        # Check for metadata file
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
            raise ValueError(f"Metadata file not found: {metadata_file}")
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_rows(self):
        """Return the number of rows for compatibility with HuggingFace Trainer"""
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            # Load and preprocess image
            image = Image.open(item['image_path']).convert('RGB')
            
            # Resize image if needed
            if image.size != (self.image_size, self.image_size):
                image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
            
            # Create conversation format for training (matching the working script)
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
            
            # Process inputs using the processor's chat template (correct approach)
            try:
                conversation_text = self.processor.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    return_tensors=None  # Get text first
                )
                
                # Now tokenize the text properly
                inputs = self.processor.tokenizer(
                    conversation_text,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Get the input_ids
                input_ids = inputs["input_ids"].squeeze(0)
                
            except Exception as e:
                log_warning(f"Error tokenizing conversation: {e}")
                # Create a dummy tensor as fallback
                input_ids = torch.zeros(self.max_length, dtype=torch.long)
            
            # Truncate if too long
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
            
            return {
                'input_ids': input_ids,
                'attention_mask': torch.ones_like(input_ids),
                'labels': input_ids.clone()
            }
            
        except Exception as e:
            log_warning(f"Error processing item {idx}: {e}")
            # Return a dummy item with proper error handling
            try:
                dummy_input = torch.zeros(self.max_length, dtype=torch.long)
                return {
                    'input_ids': dummy_input,
                    'attention_mask': torch.ones_like(dummy_input),
                    'labels': dummy_input.clone()
                }
            except Exception as fallback_error:
                log_error(f"Critical error creating dummy item for {idx}: {fallback_error}")
                # Return minimal valid item
                return {
                    'input_ids': torch.tensor([0], dtype=torch.long),
                    'attention_mask': torch.tensor([1], dtype=torch.long),
                    'labels': torch.tensor([0], dtype=torch.long)
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="./lfm2_vl_1_6b_model",
        help="Path to LFM2-VL model"
    )
    parser.add_argument(
        "--data_path_train",
        type=str,
        default="./processed_dataset/instagram_dataset",
        help="Path to Instagram training dataset"
    )
    parser.add_argument(
        "--data_path_valid",
        type=str,
        default="./processed_dataset/instagram_dataset",
        help="Path to Instagram validation dataset"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./workspace_federated/lfm2-instagram",
        help="Output directory for training"
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="PEFT",
        help="Training mode: SFT or PEFT"
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="Message mode: numpy or tensor"
    )
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--clean_up", type=int, default=0)
    args = parser.parse_args()

    # Load processor and model
    try:
        log_info("Loading LFM2-VL model and processor...")
        log_info(f"Model path: {args.model_name_or_path}")
        
        # Check if model path exists
        if not os.path.exists(args.model_name_or_path):
            raise FileNotFoundError(f"Model path does not exist: {args.model_name_or_path}")
        
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True
        )
        log_info("Processor loaded successfully")
        
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path,
            device_map="cpu",  # Use CPU for training
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        )
        log_info("Model loaded successfully")
        
    except Exception as e:
        log_error("Failed to load model or processor", e)
        raise

    # Train mode setup
    try:
        if args.train_mode.lower() == "sft":
            train_mode = 0
            log_info("Using SFT mode (full model fine-tuning)")
        elif args.train_mode.lower() == "peft":
            train_mode = 1
            log_info("Using PEFT mode (LoRA fine-tuning)")
            # Apply LoRA configuration
            log_info("Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
            log_info("Applying LoRA configuration...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
            log_info("LoRA configuration applied successfully")
        else:
            raise ValueError(f"Invalid train_mode: {args.train_mode}, only SFT and PEFT are supported.")
            
    except Exception as e:
        log_error("Failed to setup training mode", e)
        raise

    # Create datasets
    try:
        log_info("Creating Instagram caption datasets...")
        log_info(f"Training data path: {args.data_path_train}")
        log_info(f"Validation data path: {args.data_path_valid}")
        
        # Check if data paths exist
        if not os.path.exists(args.data_path_train):
            raise FileNotFoundError(f"Training data path does not exist: {args.data_path_train}")
        if not os.path.exists(args.data_path_valid):
            raise FileNotFoundError(f"Validation data path does not exist: {args.data_path_valid}")
        
        train_dataset = InstagramCaptionDataset(args.data_path_train, processor)
        val_dataset = InstagramCaptionDataset(args.data_path_valid, processor)
        
        log_info(f"Dataset sizes: training {len(train_dataset)}, validation {len(val_dataset)}")
        
    except Exception as e:
        log_error("Failed to create datasets", e)
        raise

    # Data collator
    try:
        log_info("Creating data collator...")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=processor.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
        )
        log_info("Data collator created successfully")
        
    except Exception as e:
        log_error("Failed to create data collator", e)
        raise

    # PEFT configuration for LoRA
    if args.train_mode.lower() == "peft":
        try:
            log_info("Setting up PEFT configuration...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            log_info("PEFT configuration created successfully")
        except Exception as e:
            log_error("Failed to create PEFT configuration", e)
            raise

    # Training arguments - CPU optimized
    try:
        log_info("Setting up training arguments...")
        train_args = TrainingArguments(
            output_dir=args.output_path,
            num_train_epochs=args.local_epoch,
            per_device_train_batch_size=1,  # Reduced for CPU
            per_device_eval_batch_size=1,   # Reduced for CPU
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=50,                # Reduced for CPU training
            max_grad_norm=1.0,
            logging_dir=os.path.join(args.output_path, "logs"),
            logging_steps=5,                # More frequent logging for CPU
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,             # Reduced to save disk space
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=False,                     # Disable for CPU
            bf16=False,                     # Disable for CPU
            dataloader_num_workers=0,       # Keep at 0 for CPU
            remove_unused_columns=False,
            report_to=[],
            run_name=f"lfm2-instagram-{args.train_mode.lower()}",
        )
        log_info("Training arguments configured successfully")
        
    except Exception as e:
        log_error("Failed to configure training arguments", e)
        raise

    # Initialize trainer
    try:
        log_info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        log_info("Trainer initialized successfully")
        
    except Exception as e:
        log_error("Failed to initialize trainer", e)
        raise

    # Initialize NVFlare client API
    try:
        log_info("Initializing NVFlare client API...")
        flare.init()
        log_info("NVFlare client API initialized successfully")
        
    except Exception as e:
        log_error("Failed to initialize NVFlare client API", e)
        raise

    # Train federated rounds
    try:
        log_info("Starting federated learning rounds...")
        
        while flare.is_running():
            try:
                # Receive FLModel from NVFlare
                log_info("Waiting for FL model from server...")
                input_model = flare.receive()
                curr_round = input_model.current_round
                print(f"current_round={curr_round}")
                log_info(f"Received model for round {curr_round}")
                
                # Update the key name received from global model
                try:
                    log_info("Processing received global model parameters...")
                    global_model = copy.deepcopy(input_model.params)
                    for key in list(global_model.keys()):
                        global_model[key.replace("model.", "", 1)] = global_model.pop(key)
                    log_info("Global model parameters processed successfully")
                except Exception as e:
                    log_error(f"Failed to process global model parameters", e)
                    raise

                # Evaluation function
                def evaluate(input_weights, mode):
                    try:
                        if mode:
                            set_peft_model_state_dict(trainer.model, input_weights)
                        else:
                            trainer.model.load_state_dict(input_weights)
                        metric_score = trainer.evaluate()
                        log_info(f"Evaluation metric score: {metric_score}")
                        return metric_score
                    except Exception as e:
                        log_error(f"Failed to evaluate model", e)
                        raise

                # Evaluate on received global model
                try:
                    log_info("Evaluating received global model...")
                    eval_loss = evaluate(global_model, train_mode)
                    eval_loss = float(eval_loss["eval_loss"])
                    log_info(f"Evaluation completed, loss: {eval_loss}")
                except Exception as e:
                    log_error(f"Failed to evaluate global model", e)
                    raise

                # Load global model and train
                try:
                    if curr_round == 0:
                        # First round, start from pretrained model
                        log_info("Starting first round training...")
                        trainer.train()
                        log_info("First round training completed")
                    else:
                        # Replace local resume weights with global weights
                        log_info(f"Starting round {curr_round} training...")
                        from transformers import trainer_utils
                        resume_from_checkpoint_folder = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
                        
                        if train_mode:
                            # PEFT model - save via torch.save
                            from peft import utils
                            resume_model_file_path = os.path.join(resume_from_checkpoint_folder, utils.WEIGHTS_NAME)
                            torch.save(global_model, resume_model_file_path)
                        else:
                            # SFT model - save via HF API
                            trainer.model.save_pretrained(resume_from_checkpoint_folder, safe_serialization=False)
                        
                        # Increment num_train_epochs
                        if args.clean_up:
                            trainer.args.num_train_epochs = (curr_round + 1) * args.local_epoch
                        else:
                            trainer.args.num_train_epochs += args.local_epoch
                        
                        print(f"Increment num_train_epochs to {trainer.args.num_train_epochs}")
                        log_info(f"Increment num_train_epochs to {trainer.args.num_train_epochs}")
                        trainer.train(resume_from_checkpoint=True)
                        log_info(f"Round {curr_round} training completed")
                        
                except Exception as e:
                    log_error(f"Failed to train in round {curr_round}", e)
                    raise

                # Compose output model to send back to server
                try:
                    if train_mode:
                        # PEFT - load PEFT part from trainer model
                        log_info("Extracting PEFT model parameters...")
                        out_param = get_peft_model_state_dict(trainer.model)
                    else:
                        # SFT - load whole model state_dict
                        log_info("Extracting full model parameters...")
                        out_param = trainer.model.state_dict()
                    log_info("Model parameters extracted successfully")
                except Exception as e:
                    log_error(f"Failed to extract model parameters", e)
                    raise

                # Update key names for global model
                try:
                    if not train_mode:
                        log_info("Updating key names for SFT mode...")
                        for key in list(out_param.keys()):
                            out_param["model." + key] = out_param.pop(key).cpu()

                    if args.message_mode.lower() == "numpy":
                        log_info("Converting parameters to numpy format...")
                        # Cast to float32 for numpy communication
                        out_param = {k: v.to(torch.float32) for k, v in out_param.items()}
                        
                    log_info("Model parameter formatting completed")
                except Exception as e:
                    log_error(f"Failed to format model parameters", e)
                    raise

                # Construct trained FL model
                try:
                    log_info("Constructing FL model for sending...")
                    output_model = flare.FLModel(
                        params=out_param,
                        metrics={"eval_loss": eval_loss},
                        meta={"NUM_STEPS_CURRENT_ROUND": trainer.train_dataset.num_rows},
                    )
                    log_info("FL model constructed successfully")
                except Exception as e:
                    log_error(f"Failed to construct FL model", e)
                    raise
                
                # Send model back to NVFlare
                try:
                    log_info("Sending trained model back to server...")
                    flare.send(output_model)
                    log_info("Model sent successfully")
                except Exception as e:
                    log_error(f"Failed to send model for round {curr_round}", e)
                    raise
                
            except Exception as e:
                log_error(f"Failed to process round {curr_round if 'curr_round' in locals() else 'unknown'}", e)
                raise
                
    except Exception as e:
        log_error("Failed in federated learning rounds", e)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_error("Script failed with unhandled exception", e)
        sys.exit(1)
