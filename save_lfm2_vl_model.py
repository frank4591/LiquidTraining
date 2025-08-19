#!/usr/bin/env python3
"""
Script to download and save the LFM2-VL-1.6B model locally
Based on: https://huggingface.co/LiquidAI/LFM2-VL-1.6B
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model_locally():
    """Download the LFM2-VL-1.6B model and save it locally"""
    
    model_id = "LiquidAI/LFM2-VL-1.6B"
    local_model_path = "./lfm2_vl_1_6b_model"
    
    try:
        logger.info(f"Starting download of {model_id}...")
        
        # Create local directory if it doesn't exist
        os.makedirs(local_model_path, exist_ok=True)
        
        # Download the model files
        logger.info("Downloading model files...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_model_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        logger.info(f"Model downloaded successfully to {local_model_path}")
        
        # Test loading the model to ensure it works
        logger.info("Testing model loading...")
        processor = AutoProcessor.from_pretrained(
            local_model_path, 
            trust_remote_code=True
        )
        
        model = AutoModelForImageTextToText.from_pretrained(
            local_model_path,
            device_map="auto",
            torch_dtype="bfloat16",
            trust_remote_code=True
        )
        
        logger.info("Model loaded successfully!")
        logger.info(f"Model device: {model.device}")
        logger.info(f"Model dtype: {model.dtype}")
        
        # Save model info
        model_info = {
            "model_id": model_id,
            "local_path": local_model_path,
            "device": str(model.device),
            "dtype": str(model.dtype),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        with open(os.path.join(local_model_path, "model_info.txt"), "w") as f:
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Model info saved to model_info.txt")
        
        return local_model_path
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise

def main():
    """Main function"""
    try:
        local_path = download_model_locally()
        print(f"\n✅ Model successfully downloaded and saved to: {local_path}")
        print("You can now use the model for inference!")
        
    except Exception as e:
        print(f"\n❌ Failed to download model: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
