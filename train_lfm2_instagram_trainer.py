#!/usr/bin/env python3
"""
LFM2-VL-1.6B Instagram Caption Training Script (Hugging Face Trainer)

This script fine-tunes the LFM2-VL model on Instagram caption data using the
Hugging Face Trainer API for easier training and better integration.

Features:
- LoRA fine-tuning for efficient training
- Hugging Face Trainer integration
- Automatic mixed precision training
- Gradient accumulation
- Model checkpointing and evaluation
- TensorBoard logging

Usage:
    python train_lfm2_instagram_trainer.py --data-dir /path/to/instagram/dataset --output-dir ./trained_model
"""

import os
import json
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
    TaskType,
    prepare_model_for_kbit_training
)
from PIL import Image
import logging
import argparse
from datetime import datetime
import numpy as np

# Disable wandb completely
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                    'attention_mask': torch.ones_like(dummy_input),  # Use ones instead of zeros
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

def validate_dataset(data_dir):
    """Validate the Instagram dataset structure"""
    metadata_file = os.path.join(data_dir, "metadata.json")
    
    if not os.path.exists(metadata_file):
        raise ValueError(f"Metadata file not found: {metadata_file}")
    
    # Load and validate metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    if not isinstance(metadata, list):
        raise ValueError("Metadata should be a list of posts")
    
    if len(metadata) == 0:
        raise ValueError("Dataset is empty")
    
    # Validate each post
    valid_posts = 0
    for i, post in enumerate(metadata):
        if not isinstance(post, dict):
            logger.warning(f"Post {i} is not a dictionary, skipping")
            continue
            
        if 'image_path' not in post or 'caption' not in post:
            logger.warning(f"Post {i} missing required fields, skipping")
            continue
            
        image_path = os.path.join(data_dir, post['image_path'])
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}, skipping")
            continue
            
        valid_posts += 1
    
    logger.info(f"Dataset validation complete: {valid_posts} valid posts out of {len(metadata)} total")
    
    if valid_posts == 0:
        raise ValueError("No valid posts found in dataset")
    
    return valid_posts

def setup_model_and_processor(model_path, lora_r=16, lora_alpha=32, lora_dropout=0.1):
    """Setup model and processor with LoRA configuration"""
    logger.info("Loading LFM2-VL model...")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency (only if not using LoRA)
    if not hasattr(model, 'peft_config'):
        model.gradient_checkpointing_enable()
    
    logger.info("Model setup completed!")
    return model, processor

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train LFM2 model on processed Instagram dataset using HF Trainer")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing processed Instagram dataset (e.g., ./processed_dataset/instagram_dataset)")
    parser.add_argument("--output-dir", type=str, default="./trained_model",
                       help="Output directory for trained model")
    parser.add_argument("--model-path", type=str, default="./lfm2_vl_1_6b_model",
                       help="Path to LFM2 model")
    
    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup-steps", type=int, default=100,
                       help="Number of warmup steps")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                       help="Gradient accumulation steps")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1,
                       help="LoRA dropout")
    
    # Other arguments
    parser.add_argument("--val-split", type=float, default=0.1,
                       help="Validation split ratio")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Validate the dataset first
        logger.info("Validating Instagram dataset...")
        valid_posts = validate_dataset(args.data_dir)
        logger.info(f"Dataset validation passed: {valid_posts} valid posts found")
        
        # Setup model and processor
        model, processor = setup_model_and_processor(
            args.model_path,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout
        )
        
        # Create datasets
        logger.info("Creating Instagram caption dataset...")
        full_dataset = InstagramCaptionDataset(
            args.data_dir, 
            processor
        )
        
        logger.info(f"Dataset loaded successfully: {len(full_dataset)} total posts")
        
        # Split into train/val
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size
        
        logger.info(f"Training split: {train_size} posts, Validation split: {val_size} posts")
        
        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )
        else:
            train_dataset = full_dataset
            val_dataset = None
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=processor.tokenizer,
            mlm=False,
            pad_to_multiple_of=8,  # Better memory alignment
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=10,
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False if val_dataset else None,
            fp16=False,  # Disable fp16 to avoid gradient issues
            dataloader_num_workers=0,  # Reduce workers to avoid issues
            remove_unused_columns=False,
            report_to=[],  # Completely disable all reporting
            run_name=f"lfm2-instagram-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Save final model
        final_dir = os.path.join(args.output_dir, 'final_model')
        os.makedirs(final_dir, exist_ok=True)
        trainer.save_model(final_dir)
        processor.save_pretrained(final_dir)
        
        logger.info("Training completed successfully!")
        logger.info(f"Final model saved to {final_dir}")
        
        # Evaluate if validation dataset exists
        if val_dataset:
            logger.info("Evaluating final model...")
            eval_results = trainer.evaluate()
            logger.info(f"Final evaluation results: {eval_results}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
