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

import numpy as np
import torch
from PIL import Image

from utils import read_txt_file


def load_model():
    import os

    os.environ["LOWRES_RESIZE"] = "384x32"
    os.environ["HIGHRES_BASE"] = "0x32"
    os.environ["VIDEO_RESIZE"] = "0x64"
    os.environ["VIDEO_MAXRES"] = "480"
    os.environ["VIDEO_MINRES"] = "288"
    os.environ["MAXRES"] = "1536"
    os.environ["MINRES"] = "0"
    os.environ["FORCE_NO_DOWNSAMPLE"] = "1"
    os.environ["LOAD_VISION_EARLY"] = "1"
    os.environ["PAD2STRIDE"] = "1"
    import os

    from ola.model.builder import load_pretrained_model

    model_path = "THUdyh/Ola-7b"
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None)
    model = model.to("cuda").eval()
    model = model.bfloat16()

    return tokenizer, model, image_processor


def load_audio(audio_file_name):
    import librosa
    import whisper

    speech_wav, samplerate = librosa.load(audio_file_name, sr=16000)
    if len(speech_wav.shape) > 1:
        speech_wav = speech_wav[:, 0]
    speech_wav = speech_wav.astype(np.float32)
    CHUNK_LIM = 480000
    SAMPLE_RATE = 16000
    speechs = []
    speech_wavs = []

    if len(speech_wav) <= CHUNK_LIM:
        speech = whisper.pad_or_trim(speech_wav)
        speech_wav = whisper.pad_or_trim(speech_wav)
        speechs.append(speech)
        speech_wavs.append(torch.from_numpy(speech_wav).unsqueeze(0))
    else:
        for i in range(0, len(speech_wav), CHUNK_LIM):
            chunk = speech_wav[i : i + CHUNK_LIM]
            if len(chunk) < CHUNK_LIM:
                chunk = whisper.pad_or_trim(chunk)
            speechs.append(chunk)
            speech_wavs.append(torch.from_numpy(chunk).unsqueeze(0))
    mels = []
    for chunk in speechs:
        chunk = (
            whisper.log_mel_spectrogram(chunk, n_mels=128).permute(1, 0).unsqueeze(0)
        )
        mels.append(chunk)

    mels = torch.cat(mels, dim=0)
    speech_wavs = torch.cat(speech_wavs, dim=0)
    if mels.shape[0] > 25:
        mels = mels[:25]
        speech_wavs = speech_wavs[:25]

    speech_length = torch.LongTensor([mels.shape[1]] * mels.shape[0])
    speech_chunks = torch.LongTensor([mels.shape[0]])
    return mels, speech_length, speech_chunks, speech_wavs


def extract_audio(videos_file_path):
    import moviepy as mp

    my_clip = mp.VideoFileClip(videos_file_path)
    return my_clip.audio


def generate(tokenizer_model_image_processor, prompt, example_path, modality):
    import uuid

    from decord import VideoReader, cpu
    from ola.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_SPEECH_TOKEN,
        IGNORE_INDEX,
        IMAGE_TOKEN_INDEX,
        SPEECH_TOKEN_INDEX,
    )
    from ola.conversation import SeparatorStyle, conv_templates
    from ola.datasets.preprocess import (
        tokenizer_image_token,
        tokenizer_speech_image_token,
        tokenizer_speech_question_image_token,
        tokenizer_speech_token,
    )
    from ola.mm_utils import (
        KeywordsStoppingCriteria,
        get_model_name_from_path,
        process_anyres_highres_image,
        process_anyres_video,
    )

    tokenizer, model, image_processor = tokenizer_model_image_processor

    if modality == "mllm":
        USE_SPEECH = True
        video_path = example_path
        audio_path = False
        visual, text = video_path, prompt
        modality = "video"

    elif modality == "video":
        USE_SPEECH = False
        video_path = example_path
        audio_path = False
        visual, text = video_path, prompt

    elif modality == "audio":
        USE_SPEECH = True
        video_path = False
        audio_path = example_path
        visual, text = video_path, prompt
        modality = "text"  # behaviour from ola code

    elif modality == "text":
        USE_SPEECH = False
        example = read_txt_file(example_path)
        video_path = False
        audio_path = False
        visual, text = video_path, f"{example}\n{prompt}\n"

    image_path = False

    speechs = []
    speech_lengths = []
    speech_wavs = []
    speech_chunks = []
    if modality == "video":
        vr = VideoReader(visual, ctx=cpu(0))
        total_frame_num = len(vr)
        fps = round(vr.get_avg_fps())
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, 64, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        video = [Image.fromarray(frame) for frame in spare_frames]
    elif modality == "image":
        image = [Image.open(visual)]
        image_sizes = [image[0].size]
    else:
        images = [
            torch.zeros(1, 3, 224, 224).to(
                dtype=torch.bfloat16, device="cuda", non_blocking=True
            )
        ]
        images_highres = [
            torch.zeros(1, 3, 224, 224).to(
                dtype=torch.bfloat16, device="cuda", non_blocking=True
            )
        ]
        image_sizes = [(224, 224)]

    if USE_SPEECH and audio_path:
        audio_path = audio_path
        speech, speech_length, speech_chunk, speech_wav = load_audio(audio_path)
        speechs.append(speech.bfloat16().to("cuda"))
        speech_lengths.append(speech_length.to("cuda"))
        speech_chunks.append(speech_chunk.to("cuda"))
        speech_wavs.append(speech_wav.to("cuda"))
        print("load audio")
    elif USE_SPEECH and not audio_path:
        # parse audio in the video
        audio = extract_audio(visual)
        unique_id = uuid.uuid4().hex  # generate unique string
        video_audio_path = f"./video_audio_{unique_id}.wav"
        audio.write_audiofile(video_audio_path)
        speech, speech_length, speech_chunk, speech_wav = load_audio(video_audio_path)
        speechs.append(speech.bfloat16().to("cuda"))
        speech_lengths.append(speech_length.to("cuda"))
        speech_chunks.append(speech_chunk.to("cuda"))
        speech_wavs.append(speech_wav.to("cuda"))
        os.remove(video_audio_path)
    else:
        speechs = [torch.zeros(1, 3000, 128).bfloat16().to("cuda")]
        speech_lengths = [torch.LongTensor([3000]).to("cuda")]
        speech_wavs = [torch.zeros([1, 480000]).to("cuda")]
        speech_chunks = [torch.LongTensor([1]).to("cuda")]

    conv_mode = "qwen_1_5"
    if text:
        qs = text
    else:
        qs = ""

    if USE_SPEECH and audio_path and image_path:  # image + speech instruction
        qs = (
            DEFAULT_IMAGE_TOKEN
            + "\n"
            + "User's question in speech: "
            + DEFAULT_SPEECH_TOKEN
            + "\n"
        )
    elif USE_SPEECH and video_path:  # video + audio
        qs = DEFAULT_SPEECH_TOKEN + DEFAULT_IMAGE_TOKEN + "\n" + qs
    elif USE_SPEECH and audio_path:  # audio + text
        qs = DEFAULT_SPEECH_TOKEN + "\n" + qs
    elif image_path or video_path:  # image / video
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    elif text:  # text
        qs = qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    if USE_SPEECH and audio_path and image_path:  # image + speech instruction
        input_ids = (
            tokenizer_speech_question_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to("cuda")
        )
    elif USE_SPEECH and video_path:  # video + audio
        input_ids = (
            tokenizer_speech_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to("cuda")
        )
    elif USE_SPEECH and audio_path:  # audio + text
        input_ids = (
            tokenizer_speech_token(
                prompt, tokenizer, SPEECH_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to("cuda")
        )
    else:
        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to("cuda")
        )

    if modality == "video":
        video_processed = []
        for idx, frame in enumerate(video):
            image_processor.do_resize = False
            image_processor.do_center_crop = False
            frame = process_anyres_video(frame, image_processor)

            if frame_idx is not None and idx in frame_idx:
                video_processed.append(frame.unsqueeze(0))
            elif frame_idx is None:
                video_processed.append(frame.unsqueeze(0))

        if frame_idx is None:
            frame_idx = np.arange(0, len(video_processed), dtype=int).tolist()

        video_processed = torch.cat(video_processed, dim=0).bfloat16().to("cuda")
        video_processed = (video_processed, video_processed)

        video_data = (video_processed, (384, 384), "video")
    elif modality == "image":
        image_processor.do_resize = False
        image_processor.do_center_crop = False
        image_tensor, image_highres_tensor = [], []
        for visual in image:
            image_tensor_, image_highres_tensor_ = process_anyres_highres_image(
                visual, image_processor
            )
            image_tensor.append(image_tensor_)

            image_highres_tensor.append(image_highres_tensor_)
        if all(x.shape == image_tensor[0].shape for x in image_tensor):
            image_tensor = torch.stack(image_tensor, dim=0)
        if all(x.shape == image_highres_tensor[0].shape for x in image_highres_tensor):
            image_highres_tensor = torch.stack(image_highres_tensor, dim=0)
        if type(image_tensor) is list:
            image_tensor = [_image.bfloat16().to("cuda") for _image in image_tensor]
        else:
            image_tensor = image_tensor.bfloat16().to("cuda")
        if type(image_highres_tensor) is list:
            image_highres_tensor = [
                _image.bfloat16().to("cuda") for _image in image_highres_tensor
            ]
        else:
            image_highres_tensor = image_highres_tensor.bfloat16().to("cuda")

    pad_token_ids = 151643

    attention_masks = input_ids.ne(pad_token_ids).long().to("cuda")
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    gen_kwargs = {}

    if "max_new_tokens" not in gen_kwargs:
        gen_kwargs["max_new_tokens"] = 4096
    if "temperature" not in gen_kwargs:
        gen_kwargs["temperature"] = 0.2
    if "top_p" not in gen_kwargs:
        gen_kwargs["top_p"] = None
    if "num_beams" not in gen_kwargs:
        gen_kwargs["num_beams"] = 1

    with torch.inference_mode():
        if modality == "video":
            output_ids = model.generate(
                inputs=input_ids,
                images=video_data[0][0],
                images_highres=video_data[0][1],
                modalities=video_data[2],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )

        elif modality == "image":
            output_ids = model.generate(
                inputs=input_ids,
                images=image_tensor,
                images_highres=image_highres_tensor,
                image_sizes=image_sizes,
                modalities=["image"],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )
        elif modality == "text":
            output_ids = model.generate(
                input_ids,
                images=images,
                images_highres=images_highres,
                image_sizes=image_sizes,
                modalities=["text"],
                speech=speechs,
                speech_lengths=speech_lengths,
                speech_chunks=speech_chunks,
                speech_wav=speech_wavs,
                attention_mask=attention_masks,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
            )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    response = outputs.strip()
    return response
