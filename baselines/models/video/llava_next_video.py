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

import numpy as np
import torch


def load_model():

    from transformers import (
        LlavaNextVideoForConditionalGeneration,
        LlavaNextVideoProcessor,
    )

    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)

    processor = LlavaNextVideoProcessor.from_pretrained(model_id)

    return model, processor


def read_video_pyav(container, indices):
    """
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def generate(model_processor, prompt, example_path, modality):
    import av

    model, processor = model_processor

    if modality != "video":
        raise NotImplementedError("LlaVA-NeXT-Video-7b-hf only supports video!")

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "video"},
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    video_path = example_path
    container = av.open(video_path)

    # sample uniformly 8 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)
    inputs_video = processor(
        text=prompt, videos=clip, padding=True, return_tensors="pt"
    ).to(model.device)
    len_inputs = inputs_video.input_ids.shape[1]
    output = model.generate(**inputs_video, max_new_tokens=4096, do_sample=False)
    response = processor.decode(output[0][len_inputs:], skip_special_tokens=True)

    return response
