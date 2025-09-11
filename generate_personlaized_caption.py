#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel


def load_model_and_processor(base_model_path: str, adapter_path: str, device_map: Optional[str] = "auto"):
    processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device_map,
    )

    # Attach LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, processor


def merge_and_save(model, processor, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    merged = model.merge_and_unload()
    # Ensure compact dtype to avoid large files (float32 -> ~2x bigger)
    merged = merged.to(torch.bfloat16)
    if hasattr(merged, 'config'):
        merged.config.torch_dtype = torch.bfloat16
    merged.save_pretrained(output_dir, safe_serialization=True)
    processor.save_pretrained(output_dir)
    return output_dir


def build_inputs(processor, image_path: Optional[str], text_prompt: str, device: torch.device):
    if image_path:
        image = Image.open(image_path).convert("RGB")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        prompt_text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors=None
        )
        inputs = processor.tokenizer(prompt_text, return_tensors="pt")
    else:
        # Text-only path
        inputs = processor.tokenizer(text_prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Some models don't use token_type_ids; remove to avoid generate() validation error
    if 'token_type_ids' in inputs:
        inputs.pop('token_type_ids')
    return inputs


def generate_caption(
    base_model_path: str,
    adapter_path: str,
    image_path: Optional[str],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    merge_output: Optional[str] = None,
):
    model, processor = load_model_and_processor(base_model_path, adapter_path)

    if merge_output:
        save_dir = merge_and_save(model, processor, merge_output)
        print(f"[INFO] Merged model saved to: {save_dir}")
        # Reload merged model for inference
        model = AutoModelForImageTextToText.from_pretrained(
            save_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(save_dir, trust_remote_code=True)

    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = build_inputs(processor, image_path, prompt, device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
    }
    # Only add sampling params if sampling is enabled to avoid warnings
    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\n===== Generated Caption =====\n" + text + "\n============================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge PEFT adapter into base model and generate a caption")
    parser.add_argument("--base-model", required=True, help="Path to base model directory (e.g., LFM2-VL-450M)")
    parser.add_argument("--adapter", required=True, help="Path to trained adapter directory (e.g., final_model)")
    parser.add_argument("--image", default=None, help="Optional path to image for vision+text captioning")
    parser.add_argument("--prompt", default="Generate an engaging Instagram caption for this image.", help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling; defaults to greedy if not set")
    parser.add_argument("--merge-output", default=None, help="If set, merge adapter into base and save to this dir")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.base_model):
        print(f"[ERROR] Base model path does not exist or is not a directory: {args.base_model}")
        sys.exit(1)
    if not os.path.isdir(args.adapter):
        print(f"[ERROR] Adapter path does not exist or is not a directory: {args.adapter}")
        sys.exit(1)
    if args.image and not os.path.isfile(args.image):
        print(f"[ERROR] Image path does not exist: {args.image}")
        sys.exit(1)

    generate_caption(
        base_model_path=args.base_model,
        adapter_path=args.adapter,
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
        merge_output=args.merge_output,
    )


if __name__ == "__main__":
    main()


