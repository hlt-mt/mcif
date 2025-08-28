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

import math
import tempfile

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from utils import read_txt_file


def load_model():
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-o-2_6",
        trust_remote_code=True,
        attn_implementation="sdpa",  # sdpa or flash_attention_2
        torch_dtype=torch.bfloat16,
        init_vision=True,
        init_audio=True,
        init_tts=False,
    )

    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-o-2_6", trust_remote_code=True
    )

    return model, tokenizer


def get_video_chunk_content(video_path, flatten=True, use_audio=True):
    import librosa
    from moviepy import VideoFileClip

    video = VideoFileClip(video_path, audio=use_audio)

    if use_audio:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
            temp_audio_file_path = temp_audio_file.name
            video.audio.write_audiofile(
                temp_audio_file_path, codec="pcm_s16le", fps=16000
            )
            audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        frame = video.get_frame(i + 1)
        image = Image.fromarray((frame).astype(np.uint8))
        if use_audio:
            audio = audio_np[sr * i : sr * (i + 1)]
            if flatten:
                contents.extend(["<unit>", image, audio])
            else:
                contents.append(["<unit>", image, audio])
        else:
            if flatten:
                contents.extend(["<unit>", image])
            else:
                contents.append(["<unit>", image])

    return contents


def generate(model_tokenizer, prompt, example_path, modality):

    model, tokenizer = model_tokenizer

    if modality == "mllm" or modality == "video":
        video_path = example_path
        contents = get_video_chunk_content(
            video_path, use_audio=True if modality == "mllm" else False
        )
        sys_msg = model.get_sys_prompt(mode="omni", language="en")
        msg = {"role": "user", "content": [prompt] + contents}
        msgs = [sys_msg, msg]
        response = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.5,
            max_new_tokens=4096,
            use_tts_template=True,
            generate_audio=False,
            output_audio_path=None,
            max_slice_nums=1,
            use_image_id=False,
            return_dict=True,
        )
        response = response.text

    elif modality == "audio":
        import librosa

        audio_input, _ = librosa.load(
            example_path, sr=16000, mono=True
        )  # load the audio to be captioned
        msgs = [{"role": "user", "content": [prompt, audio_input]}]
        response = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=4096,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
            output_audio_path=None,
        )

    elif modality == "text":
        example = read_txt_file(example_path)
        msgs = [{"role": "user", "content": [f"{example}\n{prompt}\n"]}]
        response = model.chat(
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            max_new_tokens=4096,
            use_tts_template=True,
            generate_audio=False,
            temperature=0.3,
            output_audio_path=None,
        )

    return response
