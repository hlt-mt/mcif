# Copyright 2025 FBK, KIT

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)


def load_model():
    model_name = "DAMO-NLP-SG/VideoLLaMA3-7B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    return model, processor


def generate(model_processor, prompt, example_path, modality):
    model, processor = model_processor

    if modality != "video":
        raise NotImplementedError("Videollama only supports video!")

    video_path = example_path
    question = prompt

    # Video conversation
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Only return the answer requested. Do not include any explanation or introductions.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {"video_path": video_path, "fps": 1, "max_frames": 128},
                },
                {"type": "text", "text": question},
            ],
        },
    ]

    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {
        k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=4096)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return response
