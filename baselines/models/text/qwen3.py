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
    model_name = "Qwen/Qwen3-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer


def generate(model_tokenizer, prompt, example_path, modality):
    model, tokenizer = model_tokenizer
    if modality != "text":
        raise NotImplementedError("Qwen3 only supports text!")

    example = read_txt_file(example_path)

    # prepare the model input
    messages = [{"role": "user", "content": f"{example}\n{prompt}"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # generation
    generated_ids = model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    index = 0  # no thinking tokens
    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip(
        "\n"
    )

    return response
