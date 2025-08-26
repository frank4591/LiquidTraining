#!/usr/bin/env python3
"""
Simple inference script for LFM2-VL model with Instagram images
"""

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import argparse

def load_model_and_processor(model_path):
    """Load the LFM2-VL model and processor"""
    print(f"Loading model from: {model_path}")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    # Load model with CPU optimization
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        device_map="cpu",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    print("Model and processor loaded successfully!")
    return model, processor

def generate_caption(model, processor, image_path, prompt="Generate an engaging Instagram caption for this image."):
    """Generate caption for a single image"""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Create conversation format
        conversation = [
            {"role": "user", "content": [
                {"type": "image", "image": image}, 
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": ""}
        ]
        
        # Apply chat template
        conversation_text = processor.apply_chat_template(
            conversation, 
            add_generation_prompt=True, 
            return_tensors=None
        )
        
        # Process inputs: pass text AND image to the processor
        inputs = processor(
            text=conversation_text,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Generate caption
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove input)
        if "assistant" in generated_text:
            generated_text = generated_text.split("assistant")[-1].strip()
        
        return generated_text
        
    except Exception as e:
        print(f"Error generating caption: {e}")
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="LFM2-VL Instagram Caption Generation")
    parser.add_argument("--model_path", type=str, default="/home/franky/LiquidTraining/lfm2_vl_1_6b_model/", 
                       help="Path to the LFM2-VL model")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the image file")
    parser.add_argument("--prompt", type=str, 
                       default="Generate an engaging Instagram caption for this image.",
                       help="Text prompt for caption generation")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        return
    
    # Check if image path exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image path {args.image_path} does not exist!")
        return
    
    print("=" * 50)
    print("LFM2-VL Instagram Caption Generator")
    print("=" * 50)
    
    # Load model and processor
    model, processor = load_model_and_processor(args.model_path)
    
    print(f"\nGenerating caption for: {args.image_path}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    
    # Generate caption
    caption = generate_caption(model, processor, args.image_path, args.prompt)
    
    print(f"Generated Caption: {caption}")
    print("=" * 50)

if __name__ == "__main__":
    main()
