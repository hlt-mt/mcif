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
    AutoTokenizer,
)


def load_model():
    torch.cuda.reset_peak_memory_stats()
    # load model
    model_path = "BAAI/Video-XL-2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map=device,
        quantization_config=None,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )  # sdpa

    return model, tokenizer


def generate(model_tokenizer, prompt, example_path, modality):
    model, tokenizer = model_tokenizer

    if modality != "video":
        raise NotImplementedError("Video-XL-2 only supports video!")

    gen_kwargs = {
        "do_sample": False,
        "temperature": 0.01,
        "top_p": 0.001,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": 4096,
    }

    model.config.enable_chunk_prefill = True
    prefill_config = {
        "chunk_prefill_mode": "streaming",
        "chunk_size": 4,
        "step_size": 1,
        "offload": True,
        "chunk_size_for_vision_tower": 24,
    }
    model.config.prefill_config = prefill_config

    # input data
    video_path = example_path
    question1 = prompt

    # params
    max_num_frames = 1300
    sample_fps = None  # uniform sampling
    max_sample_fps = None

    with torch.inference_mode():
        response = model.chat(
            video_path,
            tokenizer,
            question1,
            chat_history=None,
            return_history=False,
            max_num_frames=max_num_frames,
            sample_fps=sample_fps,
            max_sample_fps=max_sample_fps,
            generation_config=gen_kwargs,
        )

    return response
