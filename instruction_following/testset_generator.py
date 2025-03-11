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
import datetime
import logging
import os
import random
import re
import shutil
import string
import wave
from pathlib import Path
from typing import Dict, Any, List
import xml.etree.ElementTree as ET

import yaml


TEST_SET_DEF_FNAME = "[IWSLT 2025] Test Set - ASR, ST, SQA, SSUM final.tsv"

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)
logger = logging.getLogger("iwslt2025_testset_generator")


class AudioToAlias:
    """
    Helper class that takes audio names and maps each of them to an anonymous name.
    """
    _ALREADY_RETURNED_NAMES = set()

    def __init__(self):
        self.names_map = {}

    def __getitem__(self, item: str):
        if item not in self.names_map:
            self.names_map[item] = AudioToAlias.get_random_name()
        return self.names_map[item]

    def __iter__(self):
        return self.names_map.__iter__()

    @staticmethod
    def get_random_name():
        while True:
            candidate = ''.join(random.choices(string.ascii_letters, k=10)) + ".wav"
            if candidate not in AudioToAlias._ALREADY_RETURNED_NAMES:
                AudioToAlias._ALREADY_RETURNED_NAMES.add(candidate)
                return candidate


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

    @staticmethod
    def _str_to_seconds(time_string):
        t = datetime.datetime.strptime(time_string, "%H:%M:%S")
        return datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).total_seconds()

    def answer_start(self) -> float:
        return self._str_to_seconds(self.line["Answer Start"])

    def answer_end(self) -> float:
        return self._str_to_seconds(self.line["Answer End"])


class SegmentedAudios:
    """
    Wrapper giving access to a YAML file with definition of audio splits.
    """

    def __init__(self, yaml_fname: str):
        self.audio_to_segments = {}
        with open(yaml_fname) as f:
            segments = yaml.load(f, Loader=yaml.BaseLoader)
            segm_i = 0
            for segm in segments:
                if segm["wav"] not in self.audio_to_segments:
                    self.audio_to_segments[segm["wav"]] = []
                    segm_i = 0
                self.audio_to_segments[segm["wav"]].append({
                    "wav": segm["wav"].replace(".wav", f"_{segm_i}.wav"),
                    "segm_i": segm_i,
                    "start": float(segm["offset"]),
                    "end": float(segm["offset"]) + float(segm["duration"])
                })
                segm_i += 1

    def corresponding_segments(self, wav: str, start: float, end: float) -> List[Dict[str, Any]]:
        result = []
        for segm in self.audio_to_segments[wav]:
            if end < segm["start"]:
                break
            if start < segm["end"]:
                result.append(segm)
        return result


def merge_wav_files(wavs: List[Path], output_fname: Path):
    with wave.open(output_fname.as_posix(), 'wb') as wav_out:
        for wav_path in wavs:
            with wave.open(wav_path.as_posix(), 'rb') as wav_in:
                if not wav_out.getnframes():
                    wav_out.setparams(wav_in.getparams())
                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))


def read_test_elements(source_path: Path) -> List[Dict[str, Any]]:
    """
    Reads the test set definition and returns a dictionary with the corresponding information.
    """
    instruction_builder = Instructions()
    test_elements = []
    audio_segments = SegmentedAudios(
        (source_path / "SEGMENTED_AUDIO" / "shas_segmentation.yaml").as_posix())
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
                    "iid": "ASR_" + str(test_item_def.video_id()),
                    "short_audio_segments": audio_segments.audio_to_segments[test_item_def.audio()]
                })
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "instruction": instruction_builder.ssum(),
                    "reference": test_item_def.abstract(),
                    "task": "SSUM",
                    "iid": "SSUM_" + str(test_item_def.video_id())
                })
            if test_item_def.question_type() in {"AV", "A"}:
                corresponding_audio_segments = audio_segments.corresponding_segments(
                    test_item_def.audio(),
                    test_item_def.answer_start(),
                    test_item_def.answer_end()
                )
                assert len(corresponding_audio_segments) > 0, \
                    f"No audio segment for question {test_item_def.unique_id()}"
                if len(corresponding_audio_segments) > 1:
                    logger.warning(
                        f"Question {test_item_def.unique_id()} [{test_item_def.answer_start()}"
                        f"-{test_item_def.answer_end()}] is associated with multiple speech "
                        f"segments: {corresponding_audio_segments}")
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "instruction": instruction_builder.sqa(test_item_def.question()),
                    "reference": test_item_def.answer(),
                    "task": "SQA",
                    "iid": "SQA_" + str(test_item_def.unique_id()),
                    "short_audio_segments": corresponding_audio_segments
                })
            elif test_item_def.question_type() == "NA":
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "instruction": instruction_builder.sqa(test_item_def.question()),
                    "reference": "Not answerable.",
                    "task": "SQA",
                    "iid": "SQA_" + str(test_item_def.unique_id()),
                    "short_audio_segments": [random.choice(
                        audio_segments.audio_to_segments[test_item_def.audio()])]
                })
    return test_elements


def long_track(
        test_elements: List[Dict[str, Any]],
        source_path: Path,
        output_path: Path) -> Dict[str, str]:
    """
    Writes the src and ref files for the long track in the `output_path`.
    """
    audio_to_alias = AudioToAlias()
    xml_src = ET.Element("testset", attrib={'name': "IWSLT2025"})
    xml_ref = ET.Element("testset", attrib={'name': "IWSLT2025"})
    xml_src_track = ET.SubElement(xml_src, "task", attrib={"track": "long", "text_lang": "en"})
    xml_ref_track = ET.SubElement(xml_ref, "task", attrib={"track": "long", "text_lang": "en"})
    for sample_id, sample in enumerate(test_elements):
        xml_src_sample = ET.SubElement(xml_src_track, "sample", attrib={'id': str(sample_id)})
        ET.SubElement(xml_src_sample, "audio_path").text = audio_to_alias[sample["audio"]]
        ET.SubElement(xml_src_sample, "instruction").text = sample["instruction"]
        xml_ref_sample = ET.SubElement(
            xml_ref_track,
            "sample",
            attrib={'id': str(sample_id), "iid": sample["iid"], "task": sample["task"]})
        ET.SubElement(xml_ref_sample, "audio_path").text = audio_to_alias[sample["audio"]]
        ET.SubElement(xml_ref_sample, "reference").text = sample["reference"]

    tree_src = ET.ElementTree(xml_src)
    tree_ref = ET.ElementTree(xml_ref)
    ET.indent(tree_src)
    ET.indent(tree_ref)
    tree_src.write(
        output_path / "IWSLT2025.IF.long.en.src.xml", encoding="utf-8", xml_declaration=True)
    tree_ref.write(
        output_path / "IWSLT2025.IF.long.en.ref.xml", encoding="utf-8", xml_declaration=True)

    base_audio_path = source_path / "AUDIO"
    output_audio_path = output_path / "LONG_AUDIOS"
    output_audio_path.mkdir()
    for original_name in audio_to_alias:
        shutil.copyfile(
            base_audio_path / original_name, output_audio_path / audio_to_alias[original_name])

    return audio_to_alias.names_map


def short_track(
        test_elements: List[Dict[str, Any]],
        long_audio_map: Dict[str, str],
        source_path: Path,
        output_path: Path) -> None:
    """
    Writes the src and ref files for the short track in the `output_path`.
    """
    audio_to_alias = AudioToAlias()
    short_segments_path = source_path / "SEGMENTED_AUDIO" / "shas_segments"
    audio_output_path = output_path / "SHORT_AUDIOS"
    audio_output_path.mkdir()
    xml_src = ET.Element("testset", attrib={'name': "IWSLT2025"})
    xml_ref = ET.Element("testset", attrib={'name': "IWSLT2025"})
    xml_src_track = ET.SubElement(xml_src, "task", attrib={"track": "short", "text_lang": "en"})
    xml_ref_track = ET.SubElement(xml_ref, "task", attrib={"track": "short", "text_lang": "en"})
    sample_id = 0
    for sample in test_elements:
        if sample["task"] == "SSUM":
            continue
        if sample["task"] == "ASR":
            sample_ids = []
            for short_audio_segm in sample["short_audio_segments"]:
                xml_src_sample = ET.SubElement(
                    xml_src_track, "sample", attrib={'id': str(sample_id)})
                ET.SubElement(xml_src_sample, "audio_path").text = audio_to_alias[
                    short_audio_segm["wav"]]
                ET.SubElement(xml_src_sample, "instruction").text = sample["instruction"]
                sample_ids.append(sample_id)
                sample_id += 1
            xml_ref_sample = ET.SubElement(
                xml_ref_track,
                "sample",
                attrib={
                    'id': ",".join(str(s) for s in sample_ids),
                    "iid": sample["iid"],
                    "task": sample["task"]})
            ET.SubElement(xml_ref_sample, "audio_path").text = long_audio_map[sample["audio"]]
            ET.SubElement(xml_ref_sample, "reference").text = sample["reference"]
        else:
            assert sample["task"] == "SQA", f"Unsupported task {sample['task']}"
            xml_src_sample = ET.SubElement(xml_src_track, "sample", attrib={'id': str(sample_id)})
            if len(sample["short_audio_segments"]) == 1:
                short_audio = audio_to_alias[sample["short_audio_segments"][0]["wav"]]
            else:
                short_audio = AudioToAlias.get_random_name()
                merge_wav_files(
                    [short_segments_path / s["wav"] for s in sample["short_audio_segments"]],
                    audio_output_path / short_audio)
            ET.SubElement(xml_src_sample, "audio_path").text = short_audio
            ET.SubElement(xml_src_sample, "instruction").text = sample["instruction"]
            xml_ref_sample = ET.SubElement(
                xml_ref_track,
                "sample",
                attrib={"id": str(sample_id), "iid": sample["iid"], "task": sample["task"]})
            sample_id += 1
            ET.SubElement(xml_ref_sample, "audio_path").text = short_audio
            ET.SubElement(xml_ref_sample, "reference").text = sample["reference"]

    tree_src = ET.ElementTree(xml_src)
    tree_ref = ET.ElementTree(xml_ref)
    ET.indent(tree_src)
    ET.indent(tree_ref)
    tree_src.write(
        output_path / "IWSLT2025.IF.short.en.src.xml", encoding="utf-8", xml_declaration=True)
    tree_ref.write(
        output_path / "IWSLT2025.IF.short.en.ref.xml", encoding="utf-8", xml_declaration=True)

    for original_name in audio_to_alias:
        shutil.copyfile(
            short_segments_path / original_name, audio_output_path / audio_to_alias[original_name])


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
    source_path = Path(args.source_dir)
    # we set the seed to make reproducible the test set generation even though
    # there are random choices
    random.seed(3)  # in read_test_elements we select a random segment fon NA questions
    test_elements = read_test_elements(source_path)
    # shuffle test elements to avoid clear patterns in instructions
    random.seed(42)
    random.shuffle(test_elements)
    # write the XML test definition and reference
    long_audio_map = long_track(test_elements, source_path, output_path)
    short_track(test_elements, long_audio_map, source_path, output_path)


if __name__ == "__main__":
    cli_script()
