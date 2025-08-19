#!/usr/bin/env python3
"""
Demo script for LFM2-VL-1.6B Instagram Caption Generator
"""

import os
import sys

def main():
    """Demo function"""
    print("🎯 LFM2-VL-1.6B Instagram Caption Generator Demo")
    print("=" * 60)
    
    # Check if model exists
    model_path = "./lfm2_vl_1_6b_model"
    if not os.path.exists(model_path):
        print("❌ Model not found! Please run the setup first:")
        print("   1. python save_lfm2_vl_model.py")
        print("   2. python test_setup.py")
        return 1
    
    # Check if image exists
    image_path = "../img1.jpg"
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("   Please ensure img1.jpg exists in the parent directory")
        return 1
    
    print("✅ Model and image found!")
    print("\n🚀 Running Instagram caption generation...")
    print("-" * 60)
    
    # Import and run the caption generator
    try:
        from instagram_caption_generator import InstagramCaptionGenerator
        
        # Initialize generator
        generator = InstagramCaptionGenerator(model_path=model_path)
        
        # Generate captions in different styles
        styles = ["instagram", "creative", "casual"]
        
        for style in styles:
            print(f"\n🎨 Generating {style} style caption...")
            try:
                caption = generator.generate_caption(image_path, style=style)
                print(f"📸 {style.title()} Caption:")
                print(f"   {caption}")
                print("-" * 40)
            except Exception as e:
                print(f"❌ Error generating {style} caption: {str(e)}")
        
        print("\n🎉 Demo completed successfully!")
        print("\n💡 Try these commands for more options:")
        print("   python instagram_caption_generator.py --image ../img1.jpg --style creative --num-captions 5")
        print("   python instagram_caption_generator.py --image ../img1.jpg --style professional")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
