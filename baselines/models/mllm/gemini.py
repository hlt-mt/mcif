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

import os
import tempfile
import time

from utils import read_txt_file

API_KEY = "YOUR_API_KEY"


def remove_audio(input_path):
    import ffmpeg

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name

    (
        ffmpeg.input(input_path)
        .output(
            temp_path, c="copy", audio=False
        )  # c='copy' keeps video as-is, an=None removes audio
        .run(overwrite_output=True, quiet=True)
    )

    return temp_path


def load_model():
    from google import genai

    client = genai.Client(api_key=API_KEY)
    return client


def generate(client, prompt, example_path, modality):

    if modality == "mllm":
        mp4_file = client.files.upload(file=example_path)
        while mp4_file.state == "PROCESSING":
            time.sleep(1)
            mp4_file = client.files.get(name=mp4_file.name)
        content = [mp4_file, prompt]

    elif modality == "video":
        temp_path = remove_audio(example_path)

        mp4_file = client.files.upload(file=temp_path)
        while mp4_file.state == "PROCESSING":
            time.sleep(1)
            mp4_file = client.files.get(name=mp4_file.name)
        content = [mp4_file, prompt]

    elif modality == "audio":
        wav_file = client.files.upload(file=example_path)
        content = [wav_file, prompt]

    elif modality == "text":
        example = read_txt_file(example_path)
        content = [f"{example}\n{prompt}\n"]

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=content
    )

    if modality == "video":
        os.remove(temp_path)
    response = response.text
    return response
