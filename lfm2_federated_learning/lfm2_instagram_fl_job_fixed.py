# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

from nvflare import FedJob, FilterType
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.quantization.dequantizer import ModelDequantizer
from nvflare.app_opt.pt.quantization.quantizer import ModelQuantizer
from nvflare.job_config.script_runner import ScriptRunner


def main():
    args = define_parser()
    train_script = "src/hf_ 2_instagram_fl_fixed.py"
    client_ids = args.client_ids
    num_clients = len(client_ids)

    if args.threads:
        num_threads = args.threads
    else:
        num_threads = num_clients

    if num_threads < num_clients:
        print("The number of threads smaller than the number of clients, runner clean-up will be performed.")
        clean_up = 1
    else:
        clean_up = 0

    num_rounds = args.num_rounds
    workspace_dir = args.workspace_dir
    job_dir = args.job_dir
    model_name_or_path = args.model_name_or_path
    train_mode = args.train_mode
    message_mode = args.message_mode

    # Create the FedJob
    if train_mode.lower() == "sft":
        job = FedJob(name="lfm2_instagram_sft", min_clients=num_clients)
        output_path = "sft"
    elif train_mode.lower() == "peft":
        job = FedJob(name="lfm2_instagram_peft", min_clients=num_clients)
        output_path = "peft"
    else:
        raise ValueError(f"Invalid train_mode: {train_mode}, only SFT and PEFT are supported.")

    # Define the FedAvg controller workflow and send to server
    controller = FedAvg(
        num_clients=num_clients,
        num_rounds=num_rounds,
    )
    job.to(controller, "server")

    if args.quantize_mode:
        # If using quantization, add quantize filters.
        quantizer = ModelQuantizer(quantization_type=args.quantize_mode)
        dequantizer = ModelDequantizer()
        job.to(quantizer, "server", tasks=["train"], filter_type=FilterType.TASK_DATA)
        job.to(dequantizer, "server", tasks=["train"], filter_type=FilterType.TASK_RESULT)

    # Define the model persistor and send to server
    if train_mode.lower() == "sft":
        # First send the model to the server
        job.to("src/hf_lfm2_sft_model.py", "server")
        # Then send the model persistor to the server
        model_args = {"path": "src.hf_lfm2_sft_model.LFM2VLModel", "args": {"model_name_or_path": model_name_or_path}}
    elif train_mode.lower() == "peft":
        # First send the model to the server
        job.to("src/hf_lfm2_peft_model.py", "server")
        # Then send the model persistor to the server
        model_args = {"path": "src.hf_lfm2_peft_model.LFM2VLPEFTModel", "args": {"model_name_or_path": model_name_or_path}}
    
    job.to(PTFileModelPersistor(model=model_args, allow_numpy_conversion=False), "server", id="persistor")

    # Add model selection widget and send to server
    job.to(IntimeModelSelector(key_metric="eval_loss", negate_key_metric=True), "server", id="model_selector")

    # Send ScriptRunner to all clients
    for i in range(num_clients):
        client_id = client_ids[i]
        site_name = f"{client_id}"
        data_path_train = args.data_paths[i] if isinstance(args.data_paths, list) else args.data_paths
        data_path_valid = data_path_train  # Use same path for validation

        script_args = f"--model_name_or_path {model_name_or_path} --data_path_train {data_path_train} --data_path_valid {data_path_valid} --output_path {output_path} --train_mode {train_mode} --message_mode {message_mode} --clean_up {clean_up} --batch_size 1 --gradient_accumulation_steps 1 --val_split 0.1"
        if message_mode == "tensor":
            server_expected_format = "pytorch"
        elif message_mode == "numpy":
            server_expected_format = "numpy"
        else:
            raise ValueError(f"Invalid message_mode: {message_mode}, only numpy and tensor are supported.")

        runner = ScriptRunner(
            script=train_script,
            script_args=script_args,
            server_expected_format=server_expected_format,
            launch_external_process=False,
        )
        job.to(runner, site_name, tasks=["train"])

        if args.quantize_mode:
            job.to(quantizer, site_name, tasks=["train"], filter_type=FilterType.TASK_RESULT)
            job.to(dequantizer, site_name, tasks=["train"], filter_type=FilterType.TASK_DATA)

    # Export the job
    print("job_dir=", job_dir)
    job.export_job(job_dir)

    # Run the job
    print("workspace_dir=", workspace_dir)
    print("num_threads=", num_threads)
    # job.simulator_run(workspace_dir, threads=num_threads, gpu=args.gpu)


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_ids",
        nargs="+",
        type=str,
        default=["client_00"],
        help="Client IDs, used to get the data path for each client",
    )
    parser.add_argument(
        "--data_paths",
        nargs="+",
        type=str,
        default=["/home/franky/LiquidTraining/processed_dataset/instagram_dataset"],
        help="Data paths for each client",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help="Number of rounds, default to 3",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="./workdir",
        help="work directory, default to './workdir'",
    )
    parser.add_argument(
        "--job_dir",
        type=str,
        default="./jobdir",
        help="directory for job export, default to './jobdir'",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/home/franky/LiquidTraining/LFM2-VL-450M",
        help="model name or path",
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="PEFT",
        help="training mode, SFT or PEFT, default to PEFT",
    )
    parser.add_argument(
        "--quantize_mode",
        type=str,
        default=None,
        help="quantization mode, default to None (no quantization)",
    )
    parser.add_argument(
        "--message_mode",
        type=str,
        default="numpy",
        help="message mode, numpy or tensor, default to numpy",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads to use for FL simulation, default to the number of clients",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="gpu assignments for simulating clients, comma separated, default to single gpu",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
