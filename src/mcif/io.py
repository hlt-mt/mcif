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
from dataclasses import dataclass
from pathlib import Path
from typing import List, TextIO, Union
import xml.etree.ElementTree as ET


@dataclass
class OutputSample:
    id: int
    value: str


def write_output(
        samples: List[OutputSample],
        track: str,
        language: str,
        output_name: str,
        output: Union[Path, TextIO]) -> None:
    """
    Writes the outputs to an XML file, which can be used for the evaluation.

    Args:
        samples (List[OutputSample]): List of outputs.
        track (str): Name of the track (short or long).
        language (str): Language id (two-digit code).
        output_name (str): Symbolic name of the output, telling what it represents.
        output (str): A file path or an open file-like object where to write the output.
    """
    xml = ET.Element("testset", attrib={"name": output_name, "type": "output"})
    xml_track = ET.SubElement(
        xml, "task", attrib={"track": track, "text_lang": language}
    )

    for sample in samples:
        ET.SubElement(xml_track, "sample", attrib={"id": str(sample.id)}).text = sample.value

    tree = ET.ElementTree(xml)
    ET.indent(tree)
    tree.write(output, encoding="utf-8", xml_declaration=True)
