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

from transformers import pipeline

from utils import read_txt_file


def load_model():
    pipe = pipeline("text-generation", model="Unbabel/Tower-Plus-9B", device_map="auto")
    return pipe


def generate(pipe, prompt, example_path, modality):

    if modality != "text":
        raise NotImplementedError("Tower only supports text!")

    example = read_txt_file(example_path)

    messages = [{"role": "user", "content": f"{example}\n{prompt}"}]
    outputs = pipe(messages, max_new_tokens=4096, do_sample=False)

    response = outputs[0]["generated_text"][-1]["content"].strip()

    return response
