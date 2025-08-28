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
import transformers

from utils import read_txt_file


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Replace with your model path
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )
    return pipeline


def generate(pipeline, prompt, example_path, modality):

    if modality != "text":
        raise NotImplementedError("Llama only supports text!")

    example = read_txt_file(example_path)

    system_prompt = "A chat between a curious user and an artificial intelligence assistant. Only return the answer requested. Do not include any explanation or introductions.\n"
    user_prompt = f"{example}\n{prompt}"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=4096,
    )

    response = outputs[0]["generated_text"][-1]["content"].strip()
    return response
