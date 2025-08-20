#!/usr/bin/env python3
"""
Test the trained LFM2 Instagram model
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import argparse

def test_trained_model(model_path, image_path):
    """Test the trained model on a new image"""
    
    print(f"🚀 Loading trained model from: {model_path}")
    
    # Load the trained model and processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("✅ Model loaded successfully!")
    
    # Load and process the test image
    image = Image.open(image_path).convert('RGB')
    print(f"📸 Loaded image: {image_path}")
    
    # Create conversation format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Generate an engaging Instagram caption for this image."},
            ],
        },
    ]
    
    # Process inputs - get text first, then tokenize
    conversation_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors=None  # Get text first
    )
    
    # Now tokenize the text properly
    inputs = processor.tokenizer(
        conversation_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=512
    )
    
    # Generate caption
    print("🔄 Generating caption...")
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
    
    # Decode the output
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the generated part
    if "assistant" in caption:
        caption = caption.split("assistant")[-1].strip()
    
    print("\n🎯 Generated Caption:")
    print("=" * 50)
    print(caption)
    print("=" * 50)
    
    return caption

def main():
    parser = argparse.ArgumentParser(description="Test the trained LFM2 Instagram model")
    parser.add_argument("--model-path", type=str, default="./trained_model/final_model",
                       help="Path to trained model")
    parser.add_argument("--image", type=str, default="./img1.jpg",
                       help="Path to test image")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model path not found: {args.model_path}")
        return 1
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return 1
    
    try:
        test_trained_model(args.model_path, args.image)
        return 0
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
