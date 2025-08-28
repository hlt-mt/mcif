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


def load_model():
    pipe = transformers.pipeline(
        model="fixie-ai/ultravox-v0_5-llama-3_2-1b", trust_remote_code=True
    )
    return pipe


def generate(model, prompt, example_path, modality):
    import librosa

    if modality != "audio":
        raise NotImplementedError("Ultravox only supports audio!")

    audio, sr = librosa.load(example_path, sr=16000)

    turns = [
        {
            "role": "system",
            "content": "You are a friendly and helpful character. You love to answer questions for people. Only return the answer requested. Do not include any explanation or introductions.",
        },
        {"role": "user", "content": f"<|audio|>{prompt}"},
    ]

    response = model(
        {"audio": audio, "turns": turns, "sampling_rate": sr}, max_new_tokens=4096
    )
    return response
