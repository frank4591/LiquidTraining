# LFM2-VL Instagram Caption Training with NVFlare

This directory contains the adapted NVFlare framework for training the LFM2-VL model on Instagram caption data using federated learning.

## Overview

The original NVFlare LLM training framework has been modified to support:
- **LFM2-VL model**: A vision-language model for image-text tasks
- **Instagram dataset**: Image-caption pairs for caption generation training
- **Federated learning**: Distributed training across multiple clients
- **Both SFT and PEFT modes**: Full fine-tuning or LoRA-based efficient fine-tuning

## Files Structure

```
llm_hf/
├── src/
│   ├── hf_lfm2_sft_model.py      # SFT model class for LFM2-VL
│   ├── hf_lfm2_peft_model.py     # PEFT model class for LFM2-VL
│   └── hf_lfm2_instagram_fl.py   # Main training script for Instagram data
├── lfm2_instagram_fl_job.py      # Main job configuration for LFM2-VL
├── prepare_instagram_data.py      # Data preprocessing script
├── requirements_lfm2.txt          # Dependencies for LFM2-VL training
└── README_LFM2_Instagram.md      # This file
```

## Prerequisites

1. **LFM2-VL Model**: Download and place the LFM2-VL model in `./lfm2_vl_1_6b_model/`
2. **Instagram Dataset**: Ensure your processed dataset is in `../LiquidTraining/processed_dataset/instagram_dataset/`
3. **Dependencies**: Install required packages from `requirements_lfm2.txt`
4. **Hardware**: CPU training is supported but will be significantly slower than GPU

## Installation

```bash
# Install dependencies
pip install -r requirements_lfm2.txt

# Verify NVFlare installation
python -c "import nvflare; print(nvflare.__version__)"

# Test basic setup (CPU-friendly)
python test_lfm2_setup_simple.py
```

## Data Preparation

For federated learning, each client should have their own complete Instagram dataset. You have two options:

### Option 1: Each client has their own dataset directory
```bash
# Prepare federated learning structure for existing client datasets
python prepare_instagram_data.py \
    --client_data_dirs /path/to/client1_dataset /path/to/client2_dataset /path/to/client3_dataset \
    --output_dir ./federated_instagram_data

# Or with custom client IDs
python prepare_instagram_data.py \
    --client_data_dirs /path/to/client1_dataset /path/to/client2_dataset /path/to/client3_dataset \
    --output_dir ./federated_instagram_data \
    --client_ids client_a client_b client_c
```
cd /home/franky/LiquidTraining/lfm2_federated_learning && source ~/FL/bin/activate && python lfm2_instagram_fl_job_fixed.py --client_ids client_00 --data_paths /home/franky/LiquidTraining/processed_dataset/instagram_dataset --model_name_or_path LiquidAI/LFM2-VL-450M --train_mode PEFT --num_rounds 1 --threads 1 | tee run_small_model3.log

### Option 2: Use a base directory with client subdirectories
```bash
# If your data is organized as:
# /base_path/
#   ├── client_00/
#   │   ├── metadata.json
#   │   └── images/
#   ├── client_01/
#   │   ├── metadata.json
#   │   └── images/
#   └── client_02/
#       ├── metadata.json
#       └── images/
```

This will create:
```
federated_instagram_data/
├── client_00/
│   └── data_reference.txt
├── client_01/
│   └── data_reference.txt
├── client_02/
│   └── data_reference.txt
└── federated_setup_summary.json
```

## Training Configuration

### Basic Training (PEFT mode - recommended)

```bash
# Option 1: Each client has their own dataset directory
python lfm2_instagram_fl_job.py \
    --client_ids client_00 client_01 client_02 \
    --data_paths /path/to/client1_dataset /path/to/client2_dataset /path/to/client3_dataset \
    --num_rounds 5 \
    --model_name_or_path ./lfm2_vl_1_6b_model \
    --train_mode PEFT \
    --workspace_dir /tmp/nvflare/lfm2_instagram/workdir \
    --job_dir /tmp/nvflare/lfm2_instagram/jobdir

# Option 2: Use base directory with client subdirectories
python lfm2_instagram_fl_job.py \
    --client_ids client_00 client_01 client_02 \
    --data_path ./federated_instagram_data \
    --num_rounds 5 \
    --model_name_or_path ./lfm2_vl_1_6b_model \
    --train_mode PEFT \
    --workspace_dir /tmp/nvflare/lfm2_instagram/workdir \
    --job_dir /tmp/nvflare/lfm2_instagram/jobdir
```

### Full Fine-tuning (SFT mode)

```bash
# Option 1: Each client has their own dataset directory
python lfm2_instagram_fl_job.py \
    --client_ids client_00 client_01 client_02 \
    --data_paths /path/to/client1_dataset /path/to/client2_dataset /path/to/client3_dataset \
    --num_rounds 3 \
    --model_name_or_path ./lfm2_vl_1_6b_model \
    --train_mode SFT \
    --workspace_dir /tmp/nvflare/lfm2_instagram/workdir \
    --job_dir /tmp/nvflare/lfm2_instagram/jobdir

# Option 2: Use base directory with client subdirectories
python lfm2_instagram_fl_job.py \
    --client_ids client_00 client_01 client_02 \
    --data_path ./federated_instagram_data \
    --num_rounds 3 \
    --model_name_or_path ./lfm2_vl_1_6b_model \
    --train_mode SFT \
    --workspace_dir /tmp/nvflare/lfm2_instagram/workdir \
    --job_dir /tmp/nvflare/lfm2_instagram/jobdir
```

### Advanced Configuration

```bash
python lfm2_instagram_fl_job.py \
    --client_ids client_00 client_01 client_02 client_03 \
    --num_rounds 10 \
    --model_name_or_path ./lfm2_vl_1_6b_model \
    --data_path ./federated_instagram_data \
    --train_mode PEFT \
    --message_mode tensor \
    --gpu 0,1,2,3 \
    --workspace_dir /tmp/nvflare/lfm2_instagram/workdir \
    --job_dir /tmp/nvflare/lfm2_instagram/jobdir
```

## Key Differences from Original Framework

### 1. Model Architecture
- **Original**: `AutoModelForCausalLM` (text-only models like LLaMA)
- **New**: `AutoModelForImageTextToText` (vision-language models like LFM2-VL)

### 2. Data Format
- **Original**: JSONL files with text input/output pairs
- **New**: Image files + metadata.json with image-caption pairs

### 3. Training Process
- **Original**: Text-only instruction following
- **New**: Image-to-text caption generation

### 4. Model Classes
- **Original**: `CausalLMModel` and `CausalLMPEFTModel`
- **New**: `LFM2VLModel` and `LFM2VLPEFTModel`

## Training Modes

### PEFT Mode (Recommended)
- Uses LoRA for efficient fine-tuning
- Trains only adapter weights (~16-32M parameters)
- Faster training and lower memory usage
- Good for most use cases

### SFT Mode
- Full model fine-tuning
- Trains all model parameters (~1.6B parameters)
- Higher memory usage and longer training time
- Better for domain-specific adaptation

## Monitoring and Logs

Training logs and checkpoints are saved in:
- **Workspace**: `/tmp/nvflare/lfm2_instagram/workdir/`
- **Job artifacts**: `/tmp/nvflare/lfm2_instagram/jobdir/`
- **Client outputs**: `./workspace_federated/lfm2-instagram/`

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use PEFT mode
2. **Model Loading Error**: Verify LFM2-VL model path and structure
3. **Dataset Error**: Check metadata.json format and image file paths
4. **NVFlare Error**: Ensure proper installation and GPU availability
5. **Bitsandbytes Warning**: GPU quantization warnings can be ignored for CPU training

### Performance Tuning

- **Batch Size**: Start with 1 for CPU, increase to 2-4 if GPU available
- **Gradient Accumulation**: Default is 8 for CPU, reduce to 4 for GPU
- **Learning Rate**: Default is 5e-5, may need adjustment for your dataset
- **LoRA Rank**: Default is 16, increase for better performance (more parameters)

### CPU Training Considerations

- **Memory Usage**: CPU training uses more RAM than GPU training
- **Training Speed**: Expect 10-50x slower training on CPU vs GPU
- **Batch Size**: Keep batch size at 1 for CPU to avoid memory issues
- **Gradient Accumulation**: Increase to 8-16 to maintain effective batch size

## Example Output

Successful training will show:
```
current_round=0
Dataset sizes: training 100, validation 100
Starting training...
Training completed. Do not forget to share your model on huggingface.co/models =)
current_round=1
Evaluation metric score: {'eval_loss': 2.3456}
...
```

## Next Steps

After training:
1. **Model Evaluation**: Test the trained model on new images
2. **Model Deployment**: Save and deploy the fine-tuned model
3. **Performance Analysis**: Analyze training metrics and convergence
4. **Hyperparameter Tuning**: Experiment with different configurations

## Support

For issues related to:
- **LFM2-VL Model**: Check the original model documentation
- **NVFlare Framework**: Refer to NVFlare documentation
- **Training Scripts**: Review the code comments and error messages
- **Dataset Issues**: Verify data format and preprocessing steps
