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

import transformers

from utils import read_txt_file


def load_model():
    pipeline = transformers.pipeline(
        "text-generation",
        model="microsoft/phi-4",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )
    return pipeline


def generate(pipeline, prompt, example_path, modality):

    if modality != "text":
        raise NotImplementedError("Phi4 only supports text!")

    example = read_txt_file(example_path)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Only return the answer requested. "
                       "Do not include any explanation or introductions.",
        },
        {"role": "user", "content": f"{example}\n{prompt}"},
    ]

    response = pipeline(messages, max_new_tokens=4096)[0]["generated_text"][-1][
        "content"
    ]
    return response
