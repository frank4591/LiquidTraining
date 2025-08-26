#!/usr/bin/env python3
"""
Test script to verify LFM2-VL model and Instagram dataset setup.

This script tests:
1. Model loading and processor initialization
2. Dataset loading and processing
3. Basic forward pass with sample data
"""

import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


def test_model_loading(model_path):
    """Test if LFM2-VL model can be loaded correctly"""
    print(f"Testing model loading from: {model_path}")
    
    try:
        # Test processor loading
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✓ Processor loaded successfully")
        
        # Test model loading
        print("Loading model...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="cpu",  # Force CPU for testing
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        )
        print("✓ Model loaded successfully")
        
        # Test PEFT setup
        print("Testing PEFT setup...")
        model = prepare_model_for_kbit_training(model)
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
        print("✓ PEFT setup successful")
        
        return processor, model
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None, None


def test_dataset_loading(data_path):
    """Test if Instagram dataset can be loaded correctly"""
    print(f"\nTesting dataset loading from: {data_path}")
    
    try:
        # Check metadata file
        metadata_file = os.path.join(data_path, "metadata.json")
        if not os.path.exists(metadata_file):
            print(f"✗ Metadata file not found: {metadata_file}")
            return None
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata loaded: {len(metadata)} samples")
        
        # Check first few samples
        for i, item in enumerate(metadata[:3]):
            image_path = os.path.join(data_path, item['image_path'])
            if os.path.exists(image_path):
                print(f"  ✓ Sample {i}: {item['image_path']} -> {item['caption'][:50]}...")
            else:
                print(f"  ✗ Sample {i}: Image not found: {image_path}")
        
        return metadata
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return None


def test_forward_pass(processor, model, data_path, metadata):
    """Test basic forward pass with sample data"""
    print(f"\nTesting forward pass...")
    
    try:
        # Get first sample
        if not metadata:
            print("✗ No metadata available for testing")
            return False
        
        sample = metadata[0]
        image_path = os.path.join(data_path, sample['image_path'])
        
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return False
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Create conversation format - try different approaches
        try:
            # Method 1: Direct image + text format
            inputs = processor(
                images=image,
                text=f"Generate an engaging Instagram caption for this image. Response: {sample['caption']}",
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=512
            )
        except Exception as e1:
            print(f"Method 1 failed: {e1}")
            try:
                # Method 2: Separate image and text processing
                image_inputs = processor(images=image, return_tensors="pt")
                text_inputs = processor.tokenizer(
                    f"Generate an engaging Instagram caption for this image. Response: {sample['caption']}",
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512
                )
                
                # Combine inputs
                inputs = {
                    'pixel_values': image_inputs['pixel_values'],
                    'input_ids': text_inputs['input_ids'],
                    'attention_mask': text_inputs['attention_mask']
                }
            except Exception as e2:
                print(f"Method 2 failed: {e2}")
                # Method 3: Simple text-only for testing
                inputs = processor.tokenizer(
                    f"Generate an engaging Instagram caption for this image. Response: {sample['caption']}",
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=512
                )
                print("Using text-only input for testing")
        
        # Move to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        print("Running forward pass...")
        with torch.no_grad():
            try:
                if 'pixel_values' in inputs:
                    # Image + text input
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        pixel_values=inputs['pixel_values']
                    )
                else:
                    # Text-only input
                    outputs = model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                
                print("✓ Forward pass successful")
                if hasattr(outputs, 'logits'):
                    print(f"  Output shape: {outputs.logits.shape}")
                else:
                    print(f"  Output shape: {outputs[0].shape if isinstance(outputs, tuple) else 'N/A'}")
                print(f"  Loss: {outputs.loss.item() if hasattr(outputs, 'loss') else 'N/A'}")
                
            except Exception as e:
                print(f"Forward pass failed: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def main():
    """Main test function"""
    print("LFM2-VL Instagram Training Setup Test")
    print("=" * 50)
    
    # Test paths
    model_path = "/home/franky/LiquidTraining/lfm2_vl_1_6b_model/"
    data_path = "/home/franky/LiquidTraining/processed_dataset/instagram_dataset"
    
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"✗ Model path not found: {model_path}")
        print("Please ensure the LFM2-VL model is downloaded and placed in the correct directory")
        return
    
    if not os.path.exists(data_path):
        print(f"✗ Dataset path not found: {data_path}")
        print("Please ensure the Instagram dataset is in the correct directory")
        return
    
    # Run tests
    processor, model = test_model_loading(model_path)
    if processor is None or model is None:
        print("Model loading failed. Cannot proceed with other tests.")
        return
    
    metadata = test_dataset_loading(data_path)
    if metadata is None:
        print("Dataset loading failed. Cannot proceed with forward pass test.")
        return
    
    forward_success = test_forward_pass(processor, model, data_path, metadata)
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Model Loading: {'✓ PASS' if processor and model else '✗ FAIL'}")
    print(f"Dataset Loading: {'✓ PASS' if metadata else '✗ FAIL'}")
    print(f"Forward Pass: {'✓ PASS' if forward_success else '✗ FAIL'}")
    
    if processor and model and metadata and forward_success:
        print("\n🎉 All tests passed! Your setup is ready for training.")
        print("\nNext steps:")
        print("1. Run: python prepare_instagram_data.py")
        print("2. Run: python lfm2_instagram_fl_job.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("Common issues:")
        print("- Model not downloaded or corrupted")
        print("- Dataset path incorrect or metadata.json missing")
        print("- Missing dependencies (install requirements_lfm2.txt)")
        print("- GPU memory issues (try reducing batch size)")


if __name__ == "__main__":
    main()
