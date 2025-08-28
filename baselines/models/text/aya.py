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

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import read_txt_file


def load_model():
    model_id = "CohereLabs/aya-expanse-8b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    return model, tokenizer


def generate(model_tokenizer, prompt, example_path, modality):
    model, tokenizer = model_tokenizer
    if modality != "text":
        raise NotImplementedError("Aya only supports text!")

    example = read_txt_file(example_path)

    messages = [{"role": "user", "content": f"{example}\n{prompt}"}]
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    input_len = input_ids.shape[-1]
    gen_tokens = model.generate(
        input_ids,
        max_new_tokens=4096,
        do_sample=True,
        temperature=0.3,
    )

    response = tokenizer.decode(gen_tokens[0][input_len:]).replace(
        "<|END_OF_TURN_TOKEN|>", ""
    )
    return response
