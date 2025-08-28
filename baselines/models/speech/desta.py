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
from transformers import AutoModel


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(
        "DeSTA-ntu/DeSTA2-8B-beta", trust_remote_code=True
    ).to(device)
    return model


def generate(model, prompt, example_path, modality):

    if modality != "audio":
        raise NotImplementedError("Desta only supports audio!")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Only return the answer requested. "
                       "Do not include any explanation or introductions.",
        },
        {"role": "audio", "content": example_path},
        {"role": "user", "content": prompt},
    ]

    generated_ids = model.chat(
        messages, max_new_tokens=4096, do_sample=True, temperature=0.6, top_p=0.9
    )

    response = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
