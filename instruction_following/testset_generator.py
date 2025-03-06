# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, Any
import xml.etree.ElementTree as ET


TEST_SET_DEF_FNAME = "[IWSLT 2025] Test Set - ASR, ST, SQA, SSUM final.tsv"


class Instructions:
    """
    Class responsible for providing instructions for a given task.
    At the moment the instructions are static, but in the future this may be extended
    to return more variegate and less deterministic instructions to make the task more challenging.
    """
    def asr(self, lang="en"):
        assert lang in {"en"}
        return "Transcribe the English audio."

    def sqa(self, question, lang="en"):
        assert lang in {"en"}
        return f"Answer the following question given the English audio: {question}"

    def ssum(self, lang="en"):
        assert lang in {"en"}
        return "Summarize the English audio in maximum 200 words."


class TestsetDefinitionLine:
    """
    Parses a line of the testset definition file enabling easy access to its elements.
    """
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    def __init__(self, line: Dict[str, Any]):
        self.line = line

    def video_id(self) -> int:
        return int(self.line["Video ID"])

    def audio(self) -> str:
        base_name = self.line["Video Link"].replace("https://aclanthology.org/", "")[:-4]
        return base_name + ".wav"

    def transcript(self) -> str:
        return self._RE_COMBINE_WHITESPACE.sub(" ", self.line["Revised Transcript"]).strip()

    def abstract(self) -> str:
        return self.line["Abstract"].strip()

    def question_id(self) -> int:
        return int(self.line['Question ID'])

    def question(self) -> str:
        return self.line['Question']

    def question_type(self) -> str:
        assert self.line['Question Type'] in {"AV", "A", "V", "NA"}
        return self.line['Question Type']

    def answer(self) -> str:
        return self.line['Answer']

    def unique_id(self) -> str:
        return f"{self.line['Video ID']}_{self.line['ID']}"


def long_track(args):
    """
    Writes the src and ref files for the long track in the `output-dir`.

    :param args: an argparse.Namespace containing the `source-dir` and `output-dir`
    """
    instruction_builder = Instructions()
    source_path = Path(args.source_dir)
    output_path = Path(args.output_dir)
    test_elements = []
    # Read test elements from the TSV definition
    with open(source_path / TEST_SET_DEF_FNAME, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in reader:
            test_item_def = TestsetDefinitionLine(line)
            if test_item_def.question_id() == 1:
                current_audio_path = source_path / "AUDIO" / test_item_def.audio()
                assert current_audio_path.exists(), f"{current_audio_path} not found."
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "instruction": instruction_builder.asr(),
                    "reference": test_item_def.transcript(),
                    "task": "ASR",
                    "iid": "ASR_" + str(test_item_def.video_id())
                })
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "instruction": instruction_builder.ssum(),
                    "reference": test_item_def.abstract(),
                    "task": "SSUM",
                    "iid": "SSUM_" + str(test_item_def.video_id())
                })
            if test_item_def.question_type() in {"AV", "A"}:
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "instruction": instruction_builder.sqa(test_item_def.question()),
                    "reference": test_item_def.answer(),
                    "task": "SQA",
                    "iid": "SQA_" + str(test_item_def.unique_id())
                })
            elif test_item_def.question_type() == "NA":
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "instruction": instruction_builder.sqa(test_item_def.question()),
                    "reference": "Not answerable.",
                    "task": "SQA",
                    "iid": "SQA_" + str(test_item_def.unique_id())
                })

    # Shuffle test elements to avoid clear patterns in instructions
    random.seed(42)
    random.shuffle(test_elements)

    xml_src = ET.Element("testset", attrib={'name': "IWSLT2025"})
    xml_ref = ET.Element("testset", attrib={'name': "IWSLT2025"})
    xml_src_track = ET.SubElement(xml_src, "task", attrib={"track": "long", "text_lang": "en"})
    xml_ref_track = ET.SubElement(xml_ref, "task", attrib={"track": "long", "text_lang": "en"})
    for sample_id, sample in enumerate(test_elements):
        xml_src_sample = ET.SubElement(xml_src_track, "sample", attrib={'id': str(sample_id)})
        ET.SubElement(xml_src_sample, "audio_path").text = sample["audio"]
        ET.SubElement(xml_src_sample, "instruction").text = sample["instruction"]
        xml_ref_sample = ET.SubElement(
            xml_ref_track,
            "sample",
            attrib={'id': str(sample_id), "iid": sample["iid"], "task": sample["task"]})
        ET.SubElement(xml_ref_sample, "audio_path").text = sample["audio"]
        ET.SubElement(xml_ref_sample, "reference").text = sample["reference"]

    tree_src = ET.ElementTree(xml_src)
    tree_ref = ET.ElementTree(xml_ref)
    ET.indent(tree_src)
    ET.indent(tree_ref)
    tree_src.write(
        output_path / "IWSLT2025.IF.long.en.src.xml", encoding="utf-8", xml_declaration=True)
    tree_ref.write(
        output_path / "IWSLT2025.IF.long.en.ref.xml", encoding="utf-8", xml_declaration=True)


def cli_script():
    """
    Starting from the test set definitions collected in TSV format, this scripts outputs:
     - IWSLT2025.IF.<track>.<lang>.src.xml: XML files containing the test set definitions to be
       circulated to participants.
     - IWSLT2025.IF.<track>.<lang>.ref.xml: XML files containing the corresponding references, to
       be used to compute the scores.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output-dir', '-o', type=str, required=True,
        help="the path to a folder where to write the output files")
    parser.add_argument(
        '--source-dir', '-s', type=str, required=True,
        help='the path to the folder containing the test set definition.')
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    long_track(args)
    # TODO: short_track(args)


if __name__ == "__main__":
    cli_script()
