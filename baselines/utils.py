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

import io
import logging
import os
import sys
import xml.etree.ElementTree as ET

from mcif import __benchmark_version__
from mcif.io import OutputSample, write_output

TASK_ATTRIB = ["track", "text_lang"]


def set_up_logging(output_file_path):
    log_file = output_file_path.replace(".xml", ".log")

    # Clear existing handlers if rerunning in interactive environments
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set a unified format without milliseconds
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to root logger
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])


def set_up_dirs(in_data_folder, out_folder, model, track, lang, modality, prompt):
    if not os.path.isdir(in_data_folder):
        raise FileNotFoundError(f"Input folder does not exist: {in_data_folder}")

    # Set up model-specific output directory
    model_out_folder = os.path.join(out_folder, model)
    os.makedirs(model_out_folder, exist_ok=True)

    # Define output file paths
    output_file_name = f"{modality}_{track}_{lang}_{prompt}_{model}_output.xml"
    output_file_path = os.path.join(model_out_folder, output_file_name)

    return output_file_path


def read_from_xml(folder_path, lang, track, modality, prompt, version=__benchmark_version__):
    if modality == "text" and track == "short":
        raise ValueError("Text-to-text is not available in the short track.")

    xml_path = f"{folder_path}/MCIF{version}.IF.{track}.{lang}.src.{prompt}prompts.xml"

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # List to hold the tuples
    data_list = []

    # Iterate over each 'sample' element in the XML
    for sample in root.findall(".//sample"):
        # Extract the sample id, instruction, and audio path
        sample_id = sample.get("id")
        instruction = sample.find("instruction").text
        if modality != "mllm":
            node = sample.find(f"{modality}_path")
            if node is None:
                continue
            example_path = (
                f"{folder_path}/{track.upper()}_{modality.upper()}S/{node.text}"
            )
        elif modality == "mllm":
            node = sample.find("video_path")
            if node is None:
                continue
            example_path = f"{folder_path}/{track.upper()}_VIDEOS/{node.text}"
        else:
            raise NotImplementedError(f"No example path found for modality {modality}")

        # Append the tuple to the list
        data_list.append((sample_id, instruction, example_path))

    return data_list


def write_to_xml(outputs, lang, track, output_file):
    samples = [OutputSample(sample_id, pred) for sample_id, pred in outputs]

    buffer = io.BytesIO()
    write_output(samples, track, lang, "MFIC Baselines", buffer)
    xml = buffer.getvalue().decode("utf-8")
    sys.stdout.write(xml)
    with open(output_file, "w") as f:
        f.write(xml)


def read_txt_file(file_path):
    with open(file_path, "r", encoding="UTF8") as example:
        example = example.readlines()[0].strip()
    return example
