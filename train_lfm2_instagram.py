#!/usr/bin/env python3
"""
LFM2-VL-1.6B Instagram Caption Training Script

This script fine-tunes the LFM2-VL model on Instagram caption data for improved
caption generation capabilities.

Features:
- LoRA fine-tuning for efficient training
- Instagram caption dataset preparation
- Custom training loop with validation
- Gradient checkpointing for memory efficiency
- Mixed precision training
- Model checkpointing and evaluation

Usage:
    python train_lfm2_instagram.py --data-dir /path/to/instagram/dataset --output-dir ./trained_model
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoProcessor, 
    AutoModelForImageTextToText,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_pt_utils import get_parameter_names
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from PIL import Image
import numpy as np
import logging
import argparse
from tqdm import tqdm
import wandb
from datetime import datetime
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class InstagramCaptionDataset(Dataset):
    """Dataset class for Instagram image-caption pairs"""
    
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
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            
            # Remove batch dimension
            inputs = inputs.squeeze(0)
            
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
            # Return a dummy item
            dummy_input = torch.zeros(self.max_length, dtype=torch.long)
            return {
                'input_ids': dummy_input,
                'attention_mask': torch.zeros_like(dummy_input),
                'labels': dummy_input.clone()
            }

class LFM2InstagramTrainer:
    """Main training class for LFM2 Instagram caption fine-tuning"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler()
        
        # Initialize model and processor
        self.processor = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"Using device: {self.device}")
        self._setup_model()
        self._setup_training()
    
    def _setup_model(self):
        """Load and prepare the LFM2 model for training"""
        logger.info("Loading LFM2-VL model...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.args.model_path,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.args.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing for memory efficiency
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info("Model setup completed!")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, and training parameters"""
        # Get trainable parameters
        decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.total_steps
        )
        
        logger.info("Training setup completed!")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.args.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Log to wandb
            if self.args.use_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0],
                    'epoch': epoch,
                    'step': epoch * num_batches + batch_idx
                })
            
            # Memory cleanup
            if batch_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return total_loss / num_batches
    
    def validate(self, val_loader, epoch):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        
        if self.args.use_wandb:
            wandb.log({
                'val_loss': avg_loss,
                'epoch': epoch
            })
        
        return avg_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        self.processor.save_pretrained(checkpoint_dir)
        
        # Save training state
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'args': vars(self.args)
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        # Save best model
        if is_best:
            best_dir = os.path.join(self.args.output_dir, 'best_model')
            os.makedirs(best_dir, exist_ok=True)
            self.model.save_pretrained(best_dir)
            self.processor.save_pretrained(best_dir)
            logger.info(f"Best model saved to {best_dir}")
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        logger.info("Starting training...")
        
        best_loss = float('inf')
        
        for epoch in range(self.args.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.args.num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch + 1)
            logger.info(f"Epoch {epoch + 1} - Training Loss: {train_loss:.4f}")
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch + 1)
                logger.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")
                
                # Check if this is the best model
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
            else:
                is_best = False
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, train_loss, is_best)
            
            # Early stopping
            if val_loader is not None and self.args.patience > 0:
                if epoch > 0 and val_loss > best_loss:
                    patience_counter += 1
                    if patience_counter >= self.args.patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                else:
                    patience_counter = 0
        
        logger.info("Training completed!")
        
        # Save final model
        final_dir = os.path.join(self.args.output_dir, 'final_model')
        os.makedirs(final_dir, exist_ok=True)
        self.model.save_pretrained(final_dir)
        self.processor.save_pretrained(final_dir)
        logger.info(f"Final model saved to {final_dir}")

def create_sample_dataset(data_dir):
    """Create a sample Instagram dataset structure"""
    os.makedirs(data_dir, exist_ok=True)
    
    # Create sample metadata
    sample_data = [
        {
            "image_path": "sample1.jpg",
            "caption": "Beautiful sunset vibes! 🌅 Nature never fails to amaze me. #sunset #nature #photography"
        },
        {
            "image_path": "sample2.jpg", 
            "caption": "When the sky paints itself in golden hour magic ✨ #goldenhour #sky #photography"
        },
        {
            "image_path": "sample3.jpg",
            "caption": "Sunset serenity - the perfect way to end the day 🌅 #serenity #sunset #peace"
        }
    ]
    
    metadata_file = os.path.join(data_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    logger.info(f"Sample dataset structure created in {data_dir}")
    logger.info("Please add your actual Instagram images and captions to this directory")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train LFM2 model on Instagram captions")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing Instagram dataset")
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
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    
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
    parser.add_argument("--patience", type=int, default=3,
                       help="Early stopping patience")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--create-sample", action="store_true",
                       help="Create sample dataset structure")
    
    args = parser.parse_args()
    
    # Create sample dataset if requested
    if args.create_sample:
        create_sample_dataset(args.data_dir)
        return
    
    # Setup wandb if requested
    if args.use_wandb:
        wandb.init(
            project="lfm2-instagram-training",
            name=f"lfm2-instagram-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args)
        )
    
    # Calculate total steps
    args.total_steps = args.num_epochs * 1000  # Approximate
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize trainer
        trainer = LFM2InstagramTrainer(args)
        
        # Create datasets
        train_dataset = InstagramCaptionDataset(
            args.data_dir, 
            trainer.processor
        )
        
        # Split into train/val
        val_size = int(len(train_dataset) * args.val_split)
        train_size = len(train_dataset) - val_size
        
        if val_size > 0:
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=2
            )
        else:
            val_loader = None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        # Start training
        trainer.train(train_loader, val_loader)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    
    finally:
        if args.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()
