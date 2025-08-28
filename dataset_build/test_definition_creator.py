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

# Code used to build an example of how the test set will be released to participants.

import sys
import xml.etree.ElementTree as ET

# TODO: read into a dictionary
TASK_ATTRIB = ["track", "text_lang"]
# website example
tst_definitions = [{
    "track": "long",
    "text_lang": "en",
    "samples": [
        {
            "id": "1",
            "audio": "2022.acl-long.268.wav",
            "instruction": "Transcribe the English audio."
        },
        {
            "id": "3",
            "audio": "2022.acl-long.268.wav",
            "instruction":
                "Answer the following question given the English audio: Who is the presenter?"
        },
        {
            "id": "5",
            "audio": "2022.acl-long.268.wav",
            "instruction": "Summarize the English audio in maximum 200 words."
        },
    ]}, {
    "track": "long",
    "text_lang": "de",
    "samples": [
        {
            "id": "2",
            "audio": "2022.acl-long.268.wav",
            "instruction": "Übersetzen Sie den englischen Ton ins Deutsche."
        },
    ]}, {
    "track": "long",
    "text_lang": "zh",
    "samples": [
        {"id": "4", "audio": "2022.acl-long.268.wav", "instruction": "根据英语音频，回答以下问题： 演讲者是?"},
    ]}, {
    "track": "long",
    "text_lang": "it",
    "samples": [
        {
            "id": "4",
            "audio": "2022.acl-long.268.wav",
            "instruction": "Riassumi il contenuto dell'audio usando al masssimo 200 parole."
        },
    ]},
]


xml = ET.Element("testset", attrib={'name': "IWSLT2025"})
for tst_definition in tst_definitions:
    xml_track = ET.SubElement(
        xml, "task", attrib={key: tst_definition[key] for key in TASK_ATTRIB})
    for sample in tst_definition["samples"]:
        xml_sample = ET.SubElement(xml_track, "sample", attrib={'id': sample['id']})
        ET.SubElement(xml_sample, "audio_path").text = sample["audio"]
        ET.SubElement(xml_sample, "instruction").text = sample["instruction"]

tree = ET.ElementTree(xml)
ET.indent(tree)
tree.write(sys.stdout.buffer, encoding="utf-8", xml_declaration=True)
