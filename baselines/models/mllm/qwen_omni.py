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

from utils import read_txt_file


def load_model():
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

    return model, processor


def generate(model_processor, prompt, example_path, modality):
    from qwen_omni_utils import process_mm_info

    model, processor = model_processor

    if modality == "mllm":
        USE_AUDIO_IN_VIDEO = True
        user_conv_content = [
            {"type": "video", "video": example_path},
            {"type": "text", "text": prompt},
        ]

    elif modality == "video":
        USE_AUDIO_IN_VIDEO = False
        user_conv_content = [
            {"type": "video", "video": example_path},
            {"type": "text", "text": prompt},
        ]

    elif modality == "audio":
        USE_AUDIO_IN_VIDEO = False
        user_conv_content = [
            {"type": "audio", "audio": example_path},
            {"type": "text", "text": prompt},
        ]

    elif modality == "text":
        USE_AUDIO_IN_VIDEO = False
        example = read_txt_file(example_path)
        user_conv_content = [
            {"type": "text", "text": f"{example}\n{prompt}\n"},
        ]

    system_conv = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                        "capable of perceiving auditory and visual inputs, as well as generating "
                        "text and speech. Only return the answer requested. Do not include any "
                        "explanation or introductions.",
            }
        ],
    }

    user_conv = {
        "role": "user",
        "content": user_conv_content,
    }

    conversation = [system_conv, user_conv]

    # Preparation for inference
    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audios, images, videos = process_mm_info(
        conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output text and audio
    text_ids = model.generate(
        **inputs,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
        return_audio=False,
        max_new_tokens=4096,
        thinker_max_new_tokens=256,
        thinker_do_sample=False,
    )
    text = processor.batch_decode(
        text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # postprocess
    response = text[-1].split("\nassistant")[-1].strip()
    return response
