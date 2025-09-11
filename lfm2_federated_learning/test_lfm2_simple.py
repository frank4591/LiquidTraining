#!/usr/bin/env python3
"""
Simple test script for LFM2-VL model setup
"""

import os
import sys
import torch
from PIL import Image
import numpy as np

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print("✓ Transformers imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import transformers: {e}")
        return False

def test_model_loading(model_path):
    """Test if the model can be loaded"""
    print(f"\nTesting model loading from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"✗ Model path does not exist: {model_path}")
        return False
    
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        # Load processor
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✓ Processor loaded successfully")
        
        # Load model
        print("Loading model...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        print("✓ Model loaded successfully")
        
        return True, processor, model
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False, None, None

def test_processor_functionality(processor, model):
    """Test if the processor works correctly"""
    print("\nTesting processor functionality...")
    
    try:
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test conversation format
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": dummy_image}, 
                {"type": "text", "text": "Test prompt"}
            ]},
            {"role": "assistant", "content": "Test response"}
        ]
        
        # Test chat template
        conversation_text = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=False, 
            return_tensors=None
        )
        print("✓ Chat template applied successfully")
        
        # Test tokenization passing both text and image
        inputs = processor(
            text=conversation_text,
            images=dummy_image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        print("✓ Tokenization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Processor test failed: {e}")
        return False

def test_model_inference(model, processor):
    """Test if the model can perform inference"""
    print("\nTesting model inference...")
    
    try:
        # Create a dummy image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Create conversation
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": dummy_image}, 
                {"type": "text", "text": "Generate a test caption"}
            ]},
            {"role": "assistant", "content": ""}
        ]
        
        # Apply chat template
        conversation_text = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            return_tensors=None
        )
        
        # Process inputs passing both text and image
        inputs = processor(
            text=conversation_text,
            images=dummy_image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        print("✓ Model inference successful")
        return True
        
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        return False

def main():
    print("=" * 60)
    print("LFM2-VL Model Simple Test")
    print("=" * 60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install required packages.")
        return
    
    # Model path
    model_path = "/home/franky/LiquidTraining/lfm2_vl_1_6b_model/"
    
    # Test model loading
    success, processor, model = test_model_loading(model_path)
    if not success:
        print("\n❌ Model loading failed.")
        return
    
    # Test processor
    if not test_processor_functionality(processor, model):
        print("\n❌ Processor test failed.")
        return
    
    # Test inference
    if not test_model_inference(model, processor):
        print("\n❌ Model inference failed.")
        return
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! LFM2-VL model is working correctly.")
    print("=" * 60)

if __name__ == "__main__":
    main()
