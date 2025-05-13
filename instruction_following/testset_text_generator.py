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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Set
import xml.etree.ElementTree as ET


TGT_LANGS = ["de", "it", "zh"]
LANG_INSTRUCTIONS = {
    "en": {
        "SQA": "Answer the following question given the English text:",
        "SSUM": "Summarize the English audio in maximum 200 words."
    },
    "de": {
        "ST": "Übersetze den englischen Text nach Deutsch.",
        "SQA": "Beantworte die folgende Frage basierend auf dem englischen Text:",
        "SSUM": "Fasse den Inhalt des englischen Texts in maximal 200 Wörtern zusammen."
    },
    "it": {
        "ST": "Traduci il testo inglese in italiano.",
        "SQA": "Rispondi alla seguente domanda dato il testo inglese:",
        "SSUM": "Riassumi il contenuto del testo inglese usando al massimo 200 parole."
    },
    "zh": {
        "ST": "将英文文本翻译成中文。",
        "SQA": "根据英语文本，回答以下问题：",
        "SSUM": "用不超过 200 个字概括给出的英语文本。"
    },
}
SQA_LANG_INSTRUCTIONS = {
    "en": "Answer the following question given the English audio:",
    "de": "Beantworte die folgende Frage basierend auf der englischen Audioaufnahme:",
    "it": "Rispondi alla seguente domanda dato l’audio inglese:",
    "zh": "根据英语音频，回答以下问题：",
}


@dataclass
class ReferenceSample:
    sample_ids: List[str]
    reference: str
    audio_path: str
    metadata: Dict[str, str] = None


def read_reference(
        source_path: Path, track: str, language: str) -> Dict[str, Dict[str, ReferenceSample]]:
    xml = ET.parse(source_path / f"IWSLT2025.IF.long.{language}.ref.xml")
    avail_tasks = []
    for task in xml.getroot().iter("task"):
        if task.attrib['track'] == track and task.attrib['text_lang'] == language:
            samples_by_subtask = {}
            for sample in task.iter("sample"):
                if sample.attrib['task'] not in samples_by_subtask:
                    samples_by_subtask[sample.attrib['task']] = {}
                sample_ids = sample.attrib['id'].split(",")
                sample_reference = next(sample.iter('reference')).text
                audio_path = next(sample.iter('audio_path')).text
                sample_metadata = {}
                for metadata in sample.iter('metadata'):
                    for metadata_field in metadata.iter():
                        sample_metadata[metadata_field.tag] = metadata_field.text
                samples_by_subtask[sample.attrib['task']][sample.attrib['iid']] = ReferenceSample(
                    sample_ids, sample_reference, audio_path, sample_metadata)
            return samples_by_subtask
        avail_tasks.append((task.attrib['track'], task.attrib['text_lang']))
    raise Exception(
        f"Task '{track}' for language '{language}' not available in {source_path}. "
        f"Available tasks are: {avail_tasks}.")


def read_sqa_questions(
        source_path: Path, track: str, language: str) -> Dict[str, str]:
    xml = ET.parse(source_path / f"IWSLT2025.IF.long.{language}.src.xml")
    avail_tasks = []
    for task in xml.getroot().iter("task"):
        if task.attrib['track'] == track and task.attrib['text_lang'] == language:
            question_by_sample_id = {}
            for sample in task.iter("sample"):
                sample_id = sample.attrib['id']
                sample_instruction = next(sample.iter('instruction')).text
                if sample_instruction.startswith(SQA_LANG_INSTRUCTIONS[language]):
                    question_by_sample_id[sample_id] = \
                        sample_instruction[len(SQA_LANG_INSTRUCTIONS[language]):]
            return question_by_sample_id
        avail_tasks.append((task.attrib['track'], task.attrib['text_lang']))
    raise Exception(
        f"Task '{track}' for language '{language}' not available in {source_path}. "
        f"Available tasks are: {avail_tasks}.")


def write_texts(output_path: Path, asr_elements: Dict[str, Any]) -> List[str]:
    text_output_path = output_path / "LONG_TEXTS"
    text_output_path.mkdir(exist_ok=True)
    audios_with_transcripts = []
    for tt in asr_elements.values():
        fname = tt.audio_path[:-3] + "en"
        with open(text_output_path / fname, "w") as f:
            f.write(tt.reference)
        audios_with_transcripts.append(tt.audio_path)
    return audios_with_transcripts


def write_long_track(
        reference: Dict[str, Dict[str, ReferenceSample]],
        sqa_questions: Dict[str, str],
        audios_with_transcripts: Set[str],
        output_path: Path,
        language: str):
    """
    Writes the src and ref files for the long track in the `output_path`.
    """
    xml_src = ET.Element("testset", attrib={'name': "IWSLT2025"})
    xml_ref = ET.Element("testset", attrib={'name': "IWSLT2025"})
    xml_src_track = ET.SubElement(
        xml_src, "task", attrib={"track": "long", "text_lang": language})
    xml_ref_track = ET.SubElement(
        xml_ref, "task", attrib={"track": "long", "text_lang": language})
    for task, sample in reference.items():
        if task == "ASR":
            continue
        for iid, ref_item in sample.items():
            if ref_item.audio_path not in audios_with_transcripts:
                continue
            assert len(ref_item.sample_ids) == 1
            sample_id = ref_item.sample_ids[0]
            xml_src_sample = ET.SubElement(
                xml_src_track, "sample", attrib={'id': str(sample_id)})
            ET.SubElement(xml_src_sample, "text_path").text = ref_item.audio_path[:-3] + "en"
            instruction = LANG_INSTRUCTIONS[language][task]
            if task == "SQA":
                instruction = instruction + sqa_questions[sample_id]
            ET.SubElement(xml_src_sample, "instruction").text = instruction
            xml_ref_sample = ET.SubElement(
                xml_ref_track,
                "sample",
                attrib={'id': str(sample_id), "iid": iid, "task": task})
            ET.SubElement(xml_ref_sample, "text_path").text = ref_item.audio_path[:-3] + "en"
            ET.SubElement(xml_ref_sample, "reference").text = ref_item.reference
            if task == "ST":
                xml_metadata = ET.SubElement(xml_ref_sample, "metadata")
                ET.SubElement(xml_metadata, "transcript").text = ref_item.metadata["transcript"]

    tree_src = ET.ElementTree(xml_src)
    tree_ref = ET.ElementTree(xml_ref)
    ET.indent(tree_src)
    ET.indent(tree_ref)
    tree_src.write(
        output_path / f"IWSLT2025.IF.long.{language}.src.xml",
        encoding="utf-8",
        xml_declaration=True)
    tree_ref.write(
        output_path / f"IWSLT2025.IF.long.{language}.ref.xml",
        encoding="utf-8",
        xml_declaration=True)


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
        '--source-dir', '-s', type=str, required=True,
        help='the path to the folder containing the test set definition.')
    parser.add_argument(
        '--tgt-dir', '-t', type=str, required=True,
        help='the path to the folder where to write the text test set definition.')
    args = parser.parse_args()
    source_path = Path(args.source_dir)
    tgt_path = Path(args.tgt_dir)
    tgt_path.mkdir(exist_ok=True, parents=True)
    test_elements = read_reference(source_path, "long", "en")
    sqa_questions = read_sqa_questions(source_path, "long", "en")
    audios_with_transcripts = set(write_texts(tgt_path, test_elements["ASR"]))
    write_long_track(test_elements, sqa_questions, audios_with_transcripts, tgt_path, "en")
    for lang in TGT_LANGS:
        test_elements = read_reference(source_path, "long", lang)
        sqa_questions = read_sqa_questions(source_path, "long", lang)
        write_long_track(test_elements, sqa_questions, audios_with_transcripts, tgt_path, lang)


if __name__ == "__main__":
    cli_script()
