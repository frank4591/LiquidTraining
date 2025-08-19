#!/usr/bin/env python3
"""
Test script to verify LFM2-VL-1.6B setup
"""

import os
import sys
import torch

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🔍 Testing dependencies...")
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not found")
        return False
    
    try:
        import PIL
        print(f"✅ PIL/Pillow: {PIL.__version__}")
    except ImportError:
        print("❌ PIL/Pillow not found")
        return False
    
    try:
        import huggingface_hub
        print(f"✅ HuggingFace Hub: {huggingface_hub.__version__}")
    except ImportError:
        print("❌ HuggingFace Hub not found")
        return False
    
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
        print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True

def test_model_path():
    """Test if the model path exists"""
    print("\n🔍 Testing model path...")
    
    model_path = "./lfm2_vl_1_6b_model"
    
    if os.path.exists(model_path):
        print(f"✅ Model path exists: {model_path}")
        
        # Check for key files
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"✅ Found: {file}")
            else:
                print(f"⚠️  Missing: {file}")
        
        return True
    else:
        print(f"❌ Model path not found: {model_path}")
        print("💡 Run 'python save_lfm2_vl_model.py' first to download the model")
        return False

def test_basic_inference():
    """Test basic inference if model is available"""
    print("\n🔍 Testing basic inference...")
    
    if not test_model_path():
        return False
    
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        print("Loading model...")
        processor = AutoProcessor.from_pretrained(
            "./lfm2_vl_1_6b_model", 
            trust_remote_code=True
        )
        
        model = AutoModelForImageTextToText.from_pretrained(
            "./lfm2_vl_1_6b_model",
            device_map="auto",
            torch_dtype="bfloat16",
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        print(f"✅ Model device: {model.device}")
        print(f"✅ Model dtype: {model.dtype}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing inference: {str(e)}")
        return False

def main():
    """Main test function"""
    print("🚀 LFM2-VL-1.6B Setup Test")
    print("=" * 50)
    
    # Test dependencies
    deps_ok = test_dependencies()
    
    # Test model path
    model_ok = test_model_path()
    
    # Test inference if possible
    inference_ok = False
    if deps_ok and model_ok:
        inference_ok = test_basic_inference()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"Model files: {'✅ PASS' if model_ok else '❌ FAIL'}")
    print(f"Inference: {'✅ PASS' if inference_ok else '❌ FAIL'}")
    
    if deps_ok and model_ok and inference_ok:
        print("\n🎉 All tests passed! You're ready to use LFM2-VL-1.6B")
        print("💡 Try: python instagram_caption_generator.py --image ../img1.jpg")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        
        if not deps_ok:
            print("💡 Install dependencies: pip install -r requirements.txt")
        
        if not model_ok:
            print("💡 Download model: python save_lfm2_vl_model.py")
    
    return 0

if __name__ == "__main__":
    exit(main())
