#!/usr/bin/env python3
"""
Script to run LFM2-VL federated learning with multiple clients
Each client accesses the same processed dataset structure
"""

import subprocess
import sys
import os

def run_multi_client_fl():
    """Run federated learning with multiple clients"""
    
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
    
    print("=" * 70)
    print("LFM2-VL Multi-Client Federated Learning Job")
    print("=" * 70)
    print(f"Model Path: {model_path}")
    print(f"Data Path: {data_path}")
    print("=" * 70)
    
    # Define multiple clients - each will access the same dataset
    client_ids = ["client_00", "client_01", "client_02"]
    
    # Each client uses the same data path (same dataset structure)
    data_paths = [data_path] * len(client_ids)
    
    print(f"Number of clients: {len(client_ids)}")
    print(f"Client IDs: {client_ids}")
    print(f"Data paths: {data_paths}")
    print("=" * 70)
    
    # Command to run the federated learning job with multiple clients
    cmd = [
        sys.executable,
        os.path.join(script_dir, "lfm2_instagram_fl_job.py"),
        "--client_ids", *client_ids,
        "--data_paths", *data_paths,
        "--model_name_or_path", model_path,
        "--train_mode", "PEFT",
        "--num_rounds", "3",
        "--workspace_dir", "./workdir",
        "--job_dir", "./jobdir"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 70)
    
    try:
        # Run the job
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Multi-client federated learning job completed successfully!")
        print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Multi-client federated learning job failed!")
        print("Error:", e.stderr)
        return False

def run_single_client_fl():
    """Run federated learning with a single client"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Default paths
    model_path = "/home/franky/LiquidTraining/lfm2_vl_1_6b_model"
    data_path = "/home/franky/LiquidTraining/processed_dataset/instagram_dataset"
    
    print("=" * 70)
    print("LFM2-VL Single Client Federated Learning Job")
    print("=" * 70)
    
    # Command for single client
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
    print("=" * 70)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Single client federated learning job completed successfully!")
        print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Single client federated learning job failed!")
        print("Error:", e.stderr)
        return False

def main():
    """Main function with options for different client configurations"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "multi":
        print("Running multi-client federated learning...")
        success = run_multi_client_fl()
    else:
        print("Running single client federated learning...")
        print("Use 'python run_fl_multi_client.py multi' for multi-client setup")
        success = run_single_client_fl()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
