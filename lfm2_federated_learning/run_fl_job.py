#!/usr/bin/env python3
"""
Simple script to run the LFM2-VL federated learning job
"""

import subprocess
import sys
import os

def run_fl_job():
    """Run the federated learning job with default settings"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default paths
    model_path = "/home/franky/LiquidTraining/lfm2_vl_1_6b_model"
    data_path = "/home/franky/LiquidTraining/processed_dataset/instagram_dataset"
    
    # Check if paths exist
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("Please ensure the LFM2-VL model is downloaded to the correct location")
        return False
    
    if not os.path.exists(data_path):
        print(f"❌ Data path not found: {data_path}")
        print("Please ensure the processed dataset is available at the correct location")
        return False
    
    print("=" * 60)
    print("LFM2-VL Federated Learning Job")
    print("=" * 60)
    print(f"Model Path: {model_path}")
    print(f"Data Path: {data_path}")
    print("=" * 60)
    
    # Command to run the federated learning job
    cmd = [
        sys.executable,
        os.path.join(script_dir, "lfm2_instagram_fl_job.py"),
        "--client_ids", "client_00",
        "--data_paths", data_path,
        "--model_name_or_path", model_path,
        "--train_mode", "PEFT",
        "--num_rounds", "1",
        "--workspace_dir", "./workdir",
        "--job_dir", "./jobdir"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 60)
    
    try:
        # Run the job
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Job completed successfully!")
        print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Job failed!")
        print("Error:", e.stderr)
        return False

if __name__ == "__main__":
    success = run_fl_job()
    sys.exit(0 if success else 1)
