# LFM2-VL-1.6B Instagram Caption Training Guide

This guide explains how to fine-tune the LFM2-VL-1.6B model on Instagram caption data for improved caption generation capabilities.

## 🚀 Overview

The training scripts provide two approaches to fine-tune the LFM2 model:

1. **Custom Training Loop** (`train_lfm2_instagram.py`) - Full control over training process
2. **Hugging Face Trainer** (`train_lfm2_instagram_trainer.py`) - Easier training with HF integration

Both scripts use **LoRA (Low-Rank Adaptation)** for efficient fine-tuning, requiring minimal GPU memory while achieving good results.

## 📋 Prerequisites

### 1. Install Training Dependencies

```bash
pip install -r requirements_training.txt
```

### 2. Download the Base Model

```bash
python save_lfm2_vl_model.py
```

### 3. Prepare Your Dataset

The training scripts support two dataset formats:

#### Option A: Metadata JSON (Recommended)
Create a `metadata.json` file in your dataset directory:

```json
[
    {
        "image_path": "image1.jpg",
        "caption": "Beautiful sunset vibes! 🌅 #sunset #nature #photography"
    },
    {
        "image_path": "image2.jpg",
        "caption": "When the sky paints itself in golden hour magic ✨ #goldenhour"
    }
]
```

#### Option B: Paired Files
Place each image with a corresponding `.txt` file containing the caption:
```
dataset/
├── image1.jpg
├── image1.txt
├── image2.jpg
└── image2.txt
```

### 4. Create Sample Dataset Structure

```bash
python train_lfm2_instagram.py --create-sample --data-dir ./instagram_dataset
```

This creates the basic structure - add your actual images and captions to the generated `metadata.json`.

## 🎯 Training Commands

### Basic Training (HF Trainer - Recommended)

```bash
python train_lfm2_instagram_trainer.py \
    --data-dir ./instagram_dataset \
    --output-dir ./trained_model \
    --num-epochs 10 \
    --batch-size 2 \
    --learning-rate 5e-5
```

### Advanced Training (Custom Loop)

```bash
python train_lfm2_instagram.py \
    --data-dir ./instagram_dataset \
    --output-dir ./trained_model \
    --num-epochs 15 \
    --batch-size 1 \
    --learning-rate 3e-5 \
    --gradient-checkpointing \
    --lora-r 32 \
    --lora-alpha 64
```

### Training with Weights & Biases

```bash
python train_lfm2_instagram_trainer.py \
    --data-dir ./instagram_dataset \
    --output-dir ./trained_model \
    --use-wandb \
    --num-epochs 10
```

## ⚙️ Training Parameters

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-epochs` | 10 | Number of training epochs |
| `--batch-size` | 2 | Training batch size per device |
| `--learning-rate` | 5e-5 | Learning rate for optimization |
| `--weight-decay` | 0.01 | Weight decay for regularization |
| `--warmup-steps` | 100 | Number of warmup steps |

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora-r` | 16 | LoRA rank (higher = more parameters) |
| `--lora-alpha` | 32 | LoRA alpha scaling factor |
| `--lora-dropout` | 0.1 | LoRA dropout rate |

### Memory Optimization

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gradient-checkpointing` | False | Enable gradient checkpointing |
| `--gradient-accumulation-steps` | 4 | Gradient accumulation steps |
| `--max-grad-norm` | 1.0 | Maximum gradient norm for clipping |

## 💾 Output Structure

After training, your output directory will contain:

```
trained_model/
├── checkpoint-1/          # Epoch checkpoints
├── checkpoint-2/
├── checkpoint-3/
├── best_model/            # Best model based on validation
├── final_model/           # Final trained model
└── logs/                  # Training logs (TensorBoard)
```

## 🔍 Monitoring Training

### TensorBoard Logs
```bash
tensorboard --logdir ./trained_model/logs
```

### Weights & Biases
If using `--use-wandb`, training metrics are automatically logged to W&B.

## 🧪 Testing the Trained Model

### Load and Test

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# Load trained model
model_path = "./trained_model/final_model"
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True)

# Test with an image
image = Image.open("test_image.jpg")
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Generate an engaging Instagram caption for this image."},
        ],
    },
]

inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=128)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)
```

## 📊 Performance Tips

### Memory Management
- Start with small batch sizes (1-2) and increase if memory allows
- Use `--gradient-checkpointing` for memory efficiency
- Enable `--gradient-accumulation-steps` to simulate larger batch sizes

### Training Quality
- Higher LoRA rank (32-64) for better quality, lower (8-16) for faster training
- Use validation split to monitor overfitting
- Implement early stopping with `--patience` parameter

### Dataset Quality
- Ensure captions are high-quality and consistent
- Use diverse image types and caption styles
- Aim for at least 1000+ training samples for good results

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 1`
   - Enable gradient checkpointing: `--gradient-checkpointing`
   - Increase gradient accumulation: `--gradient-accumulation-steps 8`

2. **Training Loss Not Decreasing**
   - Check learning rate: try `--learning-rate 1e-5` or `--learning-rate 1e-4`
   - Verify dataset quality and format
   - Check if captions are properly formatted

3. **Model Not Saving**
   - Ensure output directory is writable
   - Check disk space
   - Verify model path is correct

4. **Import Errors**
   - Install all requirements: `pip install -r requirements_training.txt`
   - Check PyTorch and Transformers versions compatibility

### Performance Monitoring

Monitor these metrics during training:
- **Training Loss**: Should decrease over time
- **Validation Loss**: Should decrease and not diverge from training loss
- **Learning Rate**: Should follow warmup schedule
- **GPU Memory**: Should remain stable

## 🔬 Advanced Usage

### Custom Training Loop Modifications

The custom training script (`train_lfm2_instagram.py`) allows you to:
- Modify the loss function
- Implement custom evaluation metrics
- Add custom data augmentation
- Implement custom learning rate schedules

### Multi-GPU Training

For multi-GPU training, use:
```bash
torchrun --nproc_per_node=2 train_lfm2_instagram_trainer.py \
    --data-dir ./instagram_dataset \
    --output-dir ./trained_model
```

### Resume Training

To resume from a checkpoint:
```bash
python train_lfm2_instagram_trainer.py \
    --data-dir ./instagram_dataset \
    --output-dir ./trained_model \
    --resume-from-checkpoint ./trained_model/checkpoint-5
```

## 📚 References

- [LFM2-VL-1.6B Model](https://huggingface.co/LiquidAI/LFM2-VL-1.6B)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [Transformers Training](https://huggingface.co/docs/transformers/training)

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the model documentation on Hugging Face
3. Test with the sample dataset creation
4. Check GPU memory and system requirements

## 📄 License

This training code follows the same license as the base LFM2 model (LFM Open License v1.0).
