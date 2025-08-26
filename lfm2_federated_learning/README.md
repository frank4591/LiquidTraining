# LFM2-VL Federated Learning Setup

This folder contains all the necessary scripts and configurations for training the LFM2-VL model using federated learning with NVFlare framework.

## Folder Structure

```
lfm2_federated_learning/
├── src/                           # Source code for training
│   ├── hf_lfm2_instagram_fl.py   # Main training script
│   ├── hf_lfm2_peft_model.py     # PEFT model wrapper
│   └── hf_lfm2_sft_model.py      # SFT model wrapper
├── lfm2_instagram_fl_job.py      # NVFlare job configuration
├── run_fl_job.py                  # Simple script to run FL job
├── inference_lfm2.py              # Inference script for testing
├── test_lfm2_simple.py           # Simple model test
├── requirements_lfm2.txt          # Python dependencies
└── README.md                      # This file
```

## Prerequisites

1. **Python Virtual Environment**: Activate the FL environment:
   ```bash
   source ~/FL/bin/activate
   ```

2. **Model**: Ensure LFM2-VL model is available at:
   ```
   /home/franky/LiquidTraining/lfm2_vl_1_6b_model/
   ```

3. **Dataset**: Ensure processed Instagram dataset is available at:
   ```
   /home/franky/LiquidTraining/processed_dataset/instagram_dataset/
   ```

## Quick Start

### 1. Test Model Setup
First, verify that the model and processor work correctly:
```bash
source ~/FL/bin/activate
cd /home/franky/LiquidTraining/lfm2_federated_learning
python test_lfm2_simple.py
```

### 2. Test Inference
Test caption generation on a sample image:
```bash
source ~/FL/bin/activate
python inference_lfm2.py \
  --model_path /home/franky/LiquidTraining/lfm2_vl_1_6b_model/ \
  --image_path /home/franky/LiquidTraining/processed_dataset/instagram_dataset/post_000003.jpg
```

### 3. Run Federated Learning Job
Run the federated learning training:

**Single Client (for testing):**
```bash
source ~/FL/bin/activate
python run_fl_job.py
```

**Multi-Client (for production):**
```bash
source ~/FL/bin/activate
python run_fl_multi_client.py multi
```

## Manual Job Execution

You can also run the job manually with custom parameters:

```bash
source ~/FL/bin/activate
python lfm2_instagram_fl_job.py \
  --client_ids client_00 \
  --data_paths /home/franky/LiquidTraining/processed_dataset/instagram_dataset \
  --model_name_or_path /home/franky/LiquidTraining/lfm2_vl_1_6b_model \
  --train_mode PEFT \
  --num_rounds 3 \
  --workspace_dir ./workdir \
  --job_dir ./jobdir
```

## Job Parameters

- `--client_ids`: List of client identifiers
- `--data_paths`: Paths to client datasets (each client has their own complete dataset)
- `--model_name_or_path`: Path to the LFM2-VL model
- `--train_mode`: Training mode - "SFT" (Supervised Fine-Tuning) or "PEFT" (Parameter-Efficient Fine-Tuning with LoRA)
- `--num_rounds`: Number of federated learning rounds
- `--workspace_dir`: Working directory for FL simulation
- `--job_dir`: Directory for job export

## Training Modes

### PEFT Mode (Recommended)
- Uses LoRA (Low-Rank Adaptation) for efficient training
- Faster training, lower memory usage
- Good for quick experiments and fine-tuning

### SFT Mode
- Full model fine-tuning
- Higher memory usage, slower training
- Better for comprehensive model adaptation

## Data Structure

The federated learning setup expects each client to have their own complete dataset. The data structure should be:

```
processed_dataset/instagram_dataset/
├── post_000001.jpg
├── post_000001.txt
├── post_000002.jpg
├── post_000002.txt
└── ...
```

Where each `.txt` file contains the caption for the corresponding image.

## Troubleshooting

### Common Issues

1. **"Invalid input type" Error**: This was fixed by ensuring the processor receives both text and image inputs.

2. **Model Loading Issues**: Ensure the model path is correct and the model files are accessible.

3. **Dataset Path Issues**: Verify that the dataset path exists and contains the expected files.

### Logs

- Job logs are stored in `./workdir/` and `./jobdir/`
- Client logs are in `./workdir/site-client_00/log.txt`

## Notes

- The training script is optimized for CPU training
- The persistor configuration is commented out to ensure training works properly
- **Data Access**: Each client accesses the same dataset structure (no data splitting required)
- **Training**: Each client trains on the complete dataset independently
- **Aggregation**: Server aggregates model updates from all clients after each round
- **Compatibility**: Uses the same dataset structure as your `train_lfm2_instagram_trainer.py` script

## Support

For issues or questions, check the logs in the workdir and jobdir folders, or refer to the original NVFlare documentation.
