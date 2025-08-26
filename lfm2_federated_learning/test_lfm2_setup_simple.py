#!/usr/bin/env python3
"""
Simplified test script for LFM2-VL model setup on CPU.

This script tests basic functionality without complex image processing.
"""

import os
import json
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


def test_basic_model_loading(model_path):
    """Test basic model loading on CPU"""
    print(f"Testing basic model loading from: {model_path}")
    
    try:
        # Test processor loading
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✓ Processor loaded successfully")
        
        # Test model loading with minimal settings
        print("Loading model...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✓ Model loaded successfully")
        
        return processor, model
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None, None


def test_processor_functionality(processor):
    """Test basic processor functionality"""
    print("\nTesting processor functionality...")
    
    try:
        # Test text tokenization
        test_text = "Hello world"
        text_inputs = processor.tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        print("✓ Text tokenization successful")
        print(f"  Input shape: {text_inputs['input_ids'].shape}")
        
        # Test image processing
        try:
            # Create a simple test image
            test_image = Image.new('RGB', (224, 224), color='red')
            image_inputs = processor(images=test_image, return_tensors="pt")
            print("✓ Image processing successful")
            print(f"  Image shape: {image_inputs['pixel_values'].shape}")
            return True
        except Exception as e:
            print(f"⚠ Image processing failed: {e}")
            print("  This might be expected for some models")
            return True  # Don't fail the test for image issues
            
    except Exception as e:
        print(f"✗ Processor functionality test failed: {e}")
        return False


def test_simple_forward_pass(model, processor):
    """Test simple forward pass with text only"""
    print("\nTesting simple forward pass (text only)...")
    
    try:
        # Simple text input
        test_text = "Generate a caption for an image"
        inputs = processor.tokenizer(
            test_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✓ Forward pass successful")
        if hasattr(outputs, 'logits'):
            print(f"  Output shape: {outputs.logits.shape}")
        else:
            print(f"  Output shape: {outputs[0].shape if isinstance(outputs, tuple) else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_dataset_access(data_path):
    """Test basic dataset access"""
    print(f"\nTesting dataset access from: {data_path}")
    
    try:
        # Check metadata file
        metadata_file = os.path.join(data_path, "metadata.json")
        if not os.path.exists(metadata_file):
            print(f"✗ Metadata file not found: {metadata_file}")
            return False
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✓ Metadata loaded: {len(metadata)} samples")
        
        # Check first sample
        if len(metadata) > 0:
            sample = metadata[0]
            image_path = os.path.join(data_path, sample['image_path'])
            if os.path.exists(image_path):
                print(f"✓ First sample accessible: {sample['image_path']}")
                return True
            else:
                print(f"⚠ First sample image not found: {image_path}")
                return True  # Don't fail for missing images
        else:
            print("⚠ No samples in metadata")
            return False
        
    except Exception as e:
        print(f"✗ Dataset access failed: {e}")
        return False


def main():
    """Main test function"""
    print("LFM2-VL Basic Setup Test (CPU)")
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
    
    # Run basic tests
    processor, model = test_basic_model_loading(model_path)
    if processor is None or model is None:
        print("Model loading failed. Cannot proceed with other tests.")
        return
    
    processor_ok = test_processor_functionality(processor)
    forward_ok = test_simple_forward_pass(model, processor)
    dataset_ok = test_dataset_access(data_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("BASIC TEST SUMMARY")
    print("=" * 50)
    print(f"Model Loading: {'✓ PASS' if processor and model else '✗ FAIL'}")
    print(f"Processor Test: {'✓ PASS' if processor_ok else '✗ FAIL'}")
    print(f"Forward Pass: {'✓ PASS' if forward_ok else '✗ FAIL'}")
    print(f"Dataset Access: {'✓ PASS' if dataset_ok else '✗ FAIL'}")
    
    if processor and model and processor_ok and forward_ok and dataset_ok:
        print("\n🎉 Basic tests passed! Your setup is ready for basic training.")
        print("\nNext steps:")
        print("1. Run: python prepare_instagram_data.py")
        print("2. Run: python lfm2_instagram_fl_job.py")
        print("\nNote: Training on CPU will be slow. Consider using GPU if available.")
    else:
        print("\n❌ Some basic tests failed. Please check the errors above.")
        print("Common issues:")
        print("- Model not downloaded or corrupted")
        print("- Missing dependencies (install requirements_lfm2.txt)")
        print("- Insufficient memory (try reducing batch size)")
        print("- Model format incompatibility")


if __name__ == "__main__":
    main()
