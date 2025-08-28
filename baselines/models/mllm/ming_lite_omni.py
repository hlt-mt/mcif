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
from transformers import AutoProcessor, GenerationConfig

from utils import read_txt_file


def load_model():
    import os
    import sys

    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    )
    from ming_lite_model.modeling_bailingmm import (
        BailingMMNativeForConditionalGeneration,
    )

    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        "inclusionAI/Ming-Lite-Omni", # path to your download of the repo
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(
        "inclusionAI/Ming-Lite-Omni", # path to your download of the repo
        trust_remote_code=True,
    )
    return model, processor


def generate(model_processor, prompt, example_path, modality):

    model, processor = model_processor
    if modality == "mllm":
        USE_WHISPER_ENCODER = True
        audio_example_path = example_path.replace(".mp4", ".wav").replace(
            "VIDEOS", "AUDIOS"
        )
        user_conv_content = [
            {"type": "video", "video": example_path},
            {"type": "audio", "audio": audio_example_path},
            {"type": "text", "text": prompt},
        ]

    elif modality == "video":
        USE_WHISPER_ENCODER = False
        user_conv_content = [
            {"type": "video", "video": example_path},
            {"type": "text", "text": prompt},
        ]

    elif modality == "audio":
        USE_WHISPER_ENCODER = True
        user_conv_content = [
            {"type": "audio", "audio": example_path},
            {"type": "text", "text": prompt},
        ]

    elif modality == "text":
        USE_WHISPER_ENCODER = False
        example = read_txt_file(example_path)
        user_conv_content = [
            {"type": "text", "text": f"{example}\n{prompt}\n"},
        ]

    # audio and video processed separately

    messages = [
        {"role": "HUMAN", "content": user_conv_content},
    ]

    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    if USE_WHISPER_ENCODER:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={"use_whisper_encoder": True},
        )

    else:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            audios=audio_inputs,
            return_tensors="pt",
            audio_kwargs={"use_whisper_encoder": True},
        )
    inputs = inputs.to(model.device)
    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)

    # call generate
    generation_config = GenerationConfig.from_dict({"no_repeat_ngram_size": 10})

    if USE_WHISPER_ENCODER:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            generation_config=generation_config,
            use_whisper_encoder=True,
        )
    else:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            use_cache=True,
            eos_token_id=processor.gen_terminator,
            generation_config=generation_config,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return response

