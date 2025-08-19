#!/usr/bin/env python3
"""
Instagram Caption Generator using LFM2-VL-1.6B
Based on: https://huggingface.co/LiquidAI/LFM2-VL-1.6B
"""

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.image_utils import load_image
import logging
from PIL import Image
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InstagramCaptionGenerator:
    def __init__(self, model_path="./lfm2_vl_1_6b_model"):
        """Initialize the caption generator with the LFM2-VL model"""
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using device: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Load the LFM2-VL model and processor"""
        try:
            logger.info("Loading LFM2-VL model...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                device_map="auto",
                torch_dtype="bfloat16",
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_caption(self, image_path, style="instagram", max_tokens=128):
        """
        Generate Instagram-like caption for an image
        
        Args:
            image_path (str): Path to the image file
            style (str): Caption style ('instagram', 'professional', 'casual', 'creative')
            max_tokens (int): Maximum number of tokens to generate
        
        Returns:
            str: Generated caption
        """
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image
            image = load_image(image_path)
            
            # Define style-specific prompts
            style_prompts = {
                "instagram": "Create an engaging Instagram caption for this image. Make it trendy, relatable, and use relevant hashtags. Keep it under 220 characters.",
                "professional": "Write a professional, descriptive caption for this image suitable for business or professional social media.",
                "casual": "Write a casual, friendly caption for this image that feels natural and conversational.",
                "creative": "Write a creative, artistic caption for this image that captures the mood and aesthetic."
            }
            
            prompt = style_prompts.get(style, style_prompts["instagram"])
            
            # Create conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            
            # Process inputs
            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
            ).to(self.model.device)
            
            # Generate caption
            logger.info("Generating caption...")
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
            # Decode output
            caption = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Clean up the caption (remove the prompt part)
            if "assistant" in caption:
                caption = caption.split("assistant")[-1].strip()
            
            return caption.strip()
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            raise
    
    def generate_multiple_captions(self, image_path, num_captions=3, style="instagram"):
        """Generate multiple caption variations"""
        captions = []
        
        for i in range(num_captions):
            try:
                caption = self.generate_caption(image_path, style=style)
                captions.append(caption)
                logger.info(f"Generated caption {i+1}: {caption[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to generate caption {i+1}: {str(e)}")
        
        return captions
    
    def save_captions(self, image_path, captions, output_file="generated_captions.txt"):
        """Save generated captions to a file"""
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"Captions for: {image_path}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, caption in enumerate(captions, 1):
                    f.write(f"Caption {i}:\n{caption}\n\n")
                    f.write("-" * 30 + "\n\n")
            
            logger.info(f"Captions saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving captions: {str(e)}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Instagram captions using LFM2-VL-1.6B")
    parser.add_argument("--image", "-i", required=True, help="Path to the image file")
    parser.add_argument("--style", "-s", default="instagram", 
                       choices=["instagram", "professional", "casual", "creative"],
                       help="Caption style")
    parser.add_argument("--num-captions", "-n", type=int, default=3, 
                       help="Number of captions to generate")
    parser.add_argument("--output", "-o", default="generated_captions.txt",
                       help="Output file for captions")
    parser.add_argument("--model-path", "-m", default="./lfm2_vl_1_6b_model",
                       help="Path to the local LFM2-VL model")
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = InstagramCaptionGenerator(model_path=args.model_path)
        
        # Generate captions
        print(f"Generating {args.num_captions} {args.style} captions for {args.image}...")
        captions = generator.generate_multiple_captions(
            args.image, 
            num_captions=args.num_captions, 
            style=args.style
        )
        
        # Display captions
        print("\n" + "="*60)
        print("GENERATED CAPTIONS")
        print("="*60)
        
        for i, caption in enumerate(captions, 1):
            print(f"\n📸 Caption {i}:")
            print(f"{caption}")
            print("-" * 50)
        
        # Save captions
        generator.save_captions(args.image, captions, args.output)
        
        print(f"\n✅ Captions saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
