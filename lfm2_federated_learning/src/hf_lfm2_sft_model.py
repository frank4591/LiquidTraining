# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class LFM2VLModel(torch.nn.Module):
    def __init__(self, model_name_or_path):
        super(LFM2VLModel, self).__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map="cpu",  # Use CPU
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

    def forward(self, input_ids, attention_mask=None, labels=None, pixel_values=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            return_dict=False
        )
        return output
