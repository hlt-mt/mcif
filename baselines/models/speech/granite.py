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
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = "ibm-granite/granite-speech-3.3-8b"
    speech_granite_processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = speech_granite_processor.tokenizer
    speech_granite = AutoModelForSpeechSeq2Seq.from_pretrained(model_name).to(device)

    return speech_granite, speech_granite_processor, tokenizer


def generate(model_processor_tokenizer, prompt, example_path, modality):
    import torchaudio

    if modality != "audio":
        raise NotImplementedError("Granite only supports audio!")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    speech_granite, speech_granite_processor, tokenizer = model_processor_tokenizer

    # prepare speech and text prompt, using the appropriate prompt template
    wav, sr = torchaudio.load(example_path, normalize=True)
    assert wav.shape[0] == 1 and sr == 16000  # mono, 16khz

    # create text prompt
    chat = [
        {
            "role": "system",
            "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant. Only return the answer requested. Do not include any explanation or introductions.",
        },
        {
            "role": "user",
            "content": f"<|audio|>{prompt}",
        },
    ]

    text = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # compute audio embeddings
    model_inputs = speech_granite_processor(
        text,
        wav,
        device=device,  # Computation device; returned tensors are put on CPU
        return_tensors="pt",
    ).to(device)

    model_outputs = speech_granite.generate(
        **model_inputs,
        max_new_tokens=4096,
        num_beams=4,
        do_sample=False,
        min_length=1,
        top_p=1.0,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Transformers includes the input IDs in the response.
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)

    response = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )[0]

    return response
