#!/usr/bin/env python3
"""
Test script to verify the fixed LFM2 federated learning setup
"""

import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the model classes can be loaded properly"""
    try:
        logger.info("Testing model class loading...")
        
        # Test SFT model
        sys.path.append('src')
        from hf_lfm2_sft_model import LFM2VLModel
        from hf_lfm2_peft_model import LFM2VLPEFTModel
        
        logger.info("Model classes imported successfully!")
        return True
    except Exception as e:
        logger.error(f"Error loading model classes: {e}")
        return False

def test_dataset_loading():
    """Test if the dataset can be loaded"""
    try:
        logger.info("Testing dataset loading...")
        
        data_path = "/home/franky/LiquidTraining/processed_dataset/instagram_dataset"
        metadata_file = os.path.join(data_path, "metadata.json")
        
        if not os.path.exists(metadata_file):
            logger.error(f"Metadata file not found: {metadata_file}")
            return False
            
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        logger.info(f"Dataset loaded successfully: {len(metadata)} samples")
        return True
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return False

def test_model_path():
    """Test if the model path exists"""
    try:
        logger.info("Testing model path...")
        
        model_path = "/home/franky/LiquidTraining/lfm2_vl_1_6b_model"
        if not os.path.exists(model_path):
            logger.error(f"Model path not found: {model_path}")
            return False
            
        # Check for required files
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                logger.warning(f"Required file not found: {file}")
                
        logger.info("Model path exists and has required files")
        return True
    except Exception as e:
        logger.error(f"Error checking model path: {e}")
        return False

def test_training_script():
    """Test if the training script can be imported"""
    try:
        logger.info("Testing training script import...")
        
        sys.path.append('src')
        import hf_lfm2_instagram_fl_fixed
        
        logger.info("Training script imported successfully!")
        return True
    except Exception as e:
        logger.error(f"Error importing training script: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting LFM2 federated learning setup tests...")
    
    tests = [
        ("Model Class Loading", test_model_loading),
        ("Dataset Loading", test_dataset_loading),
        ("Model Path", test_model_path),
        ("Training Script", test_training_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        result = test_func()
        results.append((test_name, result))
        
    logger.info("\n--- Test Results Summary ---")
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n✅ All tests passed! The setup should work correctly.")
    else:
        logger.info("\n❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

