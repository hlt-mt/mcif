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
from moviepy import VideoFileClip, concatenate_videoclips


TEST_SET_DEF_FNAME = "[IWSLT 2025] Test Set - ASR, ST, SQA (also cross-lingual), SSUM final .tsv"
TEST_SET_SSUM_DEF_FNAME = "[IWSLT 2025] Test Set - SSUM (only abstract in English).tsv"
TGT_LANGS = ["de", "it", "zh"]
LANG_INSTRUCTIONS = {
    "en": {
        "asr": "Transcribe the English content.",
        "sqa": "Answer the following question given the English content:",
        "ssum": "Summarize the English content in an abstract of approximately 200 words.",
    },
    "de": {
        "st": "Übersetze den englischem Inhalt nach Deutsch.",
        "sqa": "Beantworte die folgende Frage basierend auf dem englischen Inhalt:",
        "ssum": "Fasse den englischen Inhalt in einem Abstract mit maximal 200 Wörtern zusammen."
    },
    "it": {
        "st": "Traduci il contenuto inglese in italiano.",
        "sqa": "Rispondi alla seguente domanda dato il contenuto inglese:",
        "ssum": "Riassumi il contenuto inglese in un abstract di circa 200 parole."
    },
    "zh": {
        "st": "将英文内容翻译成中文。",
        "sqa": "根据所给的英文内容，回答以下问题：",
        "ssum": "用400个字左右概括所给的英语内容。"
    },
}
POOL_INSTRUCTIONS = {
    "en": {
        "asr": [
            "Please write down what is said in the English content.",
            "Generate a transcription of the English content.",
            "Convert the English content into text.",
            "Produce a written version of the English content.",
            "Provide a text transcript of the English content.",
            "Accurately transcribe the English content.",
            "Turn the English content into written text.",
            "Create a verbatim transcript of the English content.",
            "Write out the English content as it is stated."
        ],
        "sqa": [
            "Based on the English content, respond to this question:",
            "Use the English content to answer the question below:",
            "Consider the English content and reply to the question:",
            "Given the English content, what is the answer to the question:",
            "Relying on the English content, provide your answer:",
            "Interpret the English content and respond to the following:",
            "Consider the English content and answer this:",
            "Use the content in English to formulate your response:",
            "Refer to the English content to answer the question:"
        ],
        "ssum": [
            "Provide a summary of the English content using roughly 200 words.",
            "Condense the English content into a summary of about 200 words.",
            "Write a brief summary (about 200 words) of the English content.",
            "Summarize the English content, keeping it around 200 words.",
            "Create a concise summary of the English content in about 200 words.",
            "Using approximately 200 words, summarize the English audio content.",
            "Capture the main points of the English content in about 200 words.",
            "Give a summary of approximately 200 words of the English content.",
            "Write a short summary (about 200 words) of what’s in the English content."
        ]
    },
    "de": {
        "st": [
            "Übersetze den englischen Inhalt ins Deutsche.",
            "Gib den englischen Inhalt auf Deutsch wieder.",
            "Übertrage den englischen Inhalt ins Deutsche.",
            "Führe eine Übersetzung des englischen Inhalts ins Deutsche durch.",
            "Übersetze den Inhalt aus dem Englischen ins Deutsche.",
            "Formuliere den englischen Inhalt auf Deutsch.",
            "Erstelle eine deutsche Übersetzung des englischen Inhalts.",
            "Übertrage den englischen Inhalt in die deutsche Sprache.",
            "Gib den englischen Inhalt sinngemäß auf Deutsch wieder."
        ],
        "sqa": [
            "Beantworte folgende Frage unter Bezugnahme auf den englischen Inhalt:",
            "Verwende den englischen Inhalt, um diese Frage zu beantworten:",
            "Beziehe dich auf den englischen Inhalt an und gib eine Antwort auf die Frage:",
            "Basierend auf dem englischen Inhalt, beantworte die nachfolgende Frage:",
            "Nutze den englischen Inhalt zur Beantwortung der Frage:",
            "Analysiere den englischen Inhalt und beantworte die Frage:",
            "Beantworte diese Frage mithilfe des englischen Inhalts:",
            "Analysiere den englischen Inhalt und beantworte dann diese Frage:",
            "Orientiere dich am englischen Inhalt und gib eine Antwort:"
        ],
        "ssum": [
            "Fasse den englischen Inhalt in ungefähr 200 Wörtern zusammen.",
            "Erstelle eine Zusammenfassung (um die 200 Wörter) des englischen Inhalts.",
            "Schreibe eine kurze Zusammenfassung des englischen Inhalts mit ungefähr 200 Wörtern",
            "Gib den englischen Inhalt in ca. 200 Wörtern wieder.",
            "Fasse den Inhalt auf Englisch in ungefähr 200 Wörtern zusammen.",
            "Verfasse eine ungefähr 200 Wörter lange Zusammenfassung des englischen Inhalts.",
            "Erstelle eine kompakte Zusammenfassung des englischen Inhalts in ungefähr 200 Wörtern.",  # noqa: E501
            "Gib eine Kurzfassung des englischen Inhalts in ca. 200 Wörtern.",
            "Formuliere eine Zusammenfassung des englischen Inhalts mit ungefähr 200 Wörtern."
        ]
    },
    "it": {
        "st": [
            "Dammi una traduzione in italiano del contenuto in inglese.",
            "Converti il contenuto inglese in italiano.",
            "Scrivi una traduzione italiana del contenuto in inglese.",
            "Traduci in italiano ciò che viene detto in inglese.",
            "Riporta il contenuto inglese in lingua italiana.",
            "Fornisci una versione italiana del contenuto inglese.",
            "Effettua la traduzione del contenuto inglese in italiano.",
            "Trasforma il contenuto in inglese in una versione italiana.",
            "Rendi in italiano il contenuto in inglese."
        ],
        "sqa": [
            "Rispondi alla seguente domanda utilizzando il contenuto inglese:",
            "Esamina il contenuto inglese e rispondi alla domanda:",
            "Fornisci una risposta alla domanda basandoti sul contenuto inglese:",
            "Considera il contenuto inglese e rispondi a questa domanda:",
            "Rispondi alla domanda servendoti del contenuto inglese:",
            "Sulla base del contenuto inglese, dai una risposta alla domanda:",
            "Rispondi alla domanda usando le informazioni del contenuto inglese:",
            "Considera il contenuto inglese per rispondere alla seguente domanda:",
            "Utilizza il contenuto inglese come base per rispondere alla domanda:"
        ],
        "ssum": [
            "Riassumi il contenuto inglese in circa 200 parole.",
            "Fai un riassunto del contenuto in inglese con circa 200 parole.",
            "Scrivi un breve riassunto del contenuto inglese (circa 200 parole).",
            "Sintetizza il contenuto inglese in circa 200 parole.",
            "Riassumi quanto detto nel contenuto inglese usando circa 200 parole.",
            "Rendi in sintesi il contenuto inglese (circa 200 parole).",
            "Scrivi un riassunto in circa 200 parole dell’audio inglese.",
            "Esprimi in forma sintetica il contenuto inglese (circa 200 parole).",
            "Fornisci una sintesi del contenuto audio inglese in circa 200 parole."
        ]
    },
    "zh": {
        "st": [
            "把英文内容翻译成中文。",
            "将所给的英文内容转换成中文。",
            "请将所给出的英文翻译成中文。",
            "将该段英文内容翻译为中文。",
            "将这段英语内容表达为中文。",
            "用中文翻译所给内容中的英文。",
            "请将英文内容转换为汉语。",
            "收到英文内容后，用中文表述其意思。",
            "将这段英语内容用中文重新表达。"
        ],
        "sqa": [
            "根据英语内容，回答下面的问题：",
            "接收到英文内容后，回答以下问题：",
            "请结合英语内容，对如下问题作答：",
            "根据所给英文内容，给出答案：",
            "请基于所给内容中的英文信息回答问题：",
            "听完英语内容后，请为以下提问作答：",
            "参考英语内容，回答下列问题：",
            "使用所给内容中的英文来回答问题：",
            "请依据英文内容回答问题："
        ],
        "ssum": [
            "将英文内容用400个字概括。",
            "请用400字左右总结这段英文内容的要点。",
            "对这段英文内容做出400字左右的简要概括。",
            "用大约400个汉字总结这段英文内容。",
            "将这段英语内容的核心内容简要描述（400字左右）。",
            "以简洁语言，约400字总结英文内容。",
            "提炼英文内容的主要信息，用400字左右表达。",
            "用大约400字写出这段英文内容的总结。",
            "对所给的英语内容进行400字左右的总结。"
        ]
    }
}
LANG_NOT_ANSWERABLE = {
    "en": "Not answerable.",
    "de": "Nicht zu beantworten.",
    "it": "Non è possibile rispondere.",
    "zh": "无法回答。"
}

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
        return LANG_INSTRUCTIONS[lang]['asr']

    def sqa(self, question, lang="en"):
        assert lang == "en" or lang in TGT_LANGS
        return f"{LANG_INSTRUCTIONS[lang]['sqa']} {question}"

    def ssum(self, lang="en"):
        assert lang == "en" or lang in TGT_LANGS
        return LANG_INSTRUCTIONS[lang]['ssum']

    def st(self, lang):
        assert lang in TGT_LANGS
        return LANG_INSTRUCTIONS[lang]['st']


class RandomInstructions:
    def __init__(self):
        self.instructions = POOL_INSTRUCTIONS
        for lang, task_instr in LANG_INSTRUCTIONS.items():
            for task, instruction in task_instr.items():
                self.instructions[lang][task].append(instruction)

    @staticmethod
    def _random(pool):
        return random.choice(pool)

    def asr(self, lang="en"):
        assert lang in {"en"}
        return self._random(self.instructions[lang]['asr'])

    def sqa(self, question, lang="en"):
        assert lang == "en" or lang in TGT_LANGS
        return f"{self._random(self.instructions[lang]['sqa'])} {question}"

    def ssum(self, lang="en"):
        assert lang == "en" or lang in TGT_LANGS
        return self._random(self.instructions[lang]['ssum'])

    def st(self, lang):
        assert lang in TGT_LANGS
        return self._random(self.instructions[lang]['st'])


class TestsetDefinitionLine:
    """
    Parses a line of the testset definition file enabling easy access to its elements.
    """
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    def __init__(self, line: Dict[str, Any]):
        self.line = line

    def video_id(self) -> int:
        return int(self.line["Video ID"])

    def line_id(self) -> int:
        return int(self.line["ID"])

    def audio(self) -> str:
        base_name = self.line["Video Link"].replace("https://aclanthology.org/", "")[:-4]
        return base_name + ".wav"

    def transcript(self) -> str:
        return self._RE_COMBINE_WHITESPACE.sub(" ", self.line["Revised Transcript"]).strip()

    def translation(self, lang) -> str:
        return self._RE_COMBINE_WHITESPACE.sub(" ", self.line[f"Translation_{lang}"]).strip()

    def abstract(self, lang="en") -> str:
        return self.line[f"Abstract_{lang}"].strip()

    def question_id(self) -> int:
        return int(self.line['Question ID'])

    def question(self, lang="en") -> str:
        return self.line[f'Question_{lang}']

    def question_type(self) -> str:
        assert self.line['Question Type'] in {"AV", "A", "V", "NA"}
        return self.line['Question Type']

    def answer(self, lang="en") -> str:
        return self.line[f'Answer_{lang}']

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

    def question_origin(self):
        return self.line['Question Origin'][:-2]  # strip ' Q'


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


class Translation:
    def __init__(self, translation_file: Path):
        self.videos = {}
        with open(translation_file) as f:
            reader = csv.reader(f, delimiter='\t', quotechar=None)
            next(reader)  # skip header
            for line in reader:
                assert len(line) % 2 == 0  # transcript and translation lines are alternated
                transcript_lines = []
                translation_lines = []
                for i in range(2, len(line)):
                    if line[i] == '':
                        continue
                    if i % 2 == 0:
                        transcript_lines.append(line[i])
                    else:
                        translation_lines.append(line[i])
                assert len(transcript_lines) == len(translation_lines)
                self.videos[int(line[0])] = {
                    "transcript": "\n".join(transcript_lines),
                    "translation": "\n".join(translation_lines),
                }

    def __getitem__(self, item):
        return self.videos[item]


class AbstractTranslation:
    def __init__(self, translation_file: Path):
        self.videos = {}
        with open(translation_file) as f:
            reader = csv.reader(f, delimiter='\t', quotechar=None)
            next(reader)  # skip header
            for line in reader:
                assert len(line) % 2 == 0  # transcript and translation lines are alternated
                transcript_lines = []
                translation_lines = []
                for i in range(2, len(line)):
                    if line[i] == '':
                        continue
                    if i % 2 == 0:
                        transcript_lines.append(line[i])
                    else:
                        translation_lines.append(line[i])
                assert len(transcript_lines) == len(translation_lines)
                self.videos[int(line[0])] = " ".join(translation_lines).strip()

    def __getitem__(self, item):
        return self.videos[item]


def merge_wav_files(wavs: List[Path], output_fname: Path):
    with wave.open(output_fname.as_posix(), 'wb') as wav_out:
        for wav_path in wavs:
            with wave.open(wav_path.as_posix(), 'rb') as wav_in:
                if not wav_out.getnframes():
                    wav_out.setparams(wav_in.getparams())
                wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))


def merge_mp4_files(mp4s: List[Path], output_fname: Path):
    video_clips = [VideoFileClip(mp4) for mp4 in mp4s]
    final_clip = concatenate_videoclips(video_clips)
    final_clip.write_videofile(output_fname, audio_codec='aac')


def total_segments_duration(segments: List[Dict[str, Any]]) -> float:
    total_seconds = 0.
    for segment in segments:
        total_seconds += segment["end"] - segment["start"]
    return total_seconds


def read_test_elements(source_path: Path, include_video: bool = False) -> List[Dict[str, Any]]:
    """
    Reads the test set definition and returns a dictionary with the corresponding information.
    """
    instruction_builder = Instructions()
    random_instruction_builder = RandomInstructions()
    test_elements = []
    audio_segments = SegmentedAudios(
        (source_path / "SEGMENTED_AUDIO" / "shas_segmentation.yaml").as_posix())
    translations = {
        "de": Translation(
            source_path / "[IWSLT 2025] Test Set - TRANSCRIPT_german - REVISED.tsv"),
        "it": Translation(
            source_path / "[IWSLT 2025] Test Set - TRANSCRIPT_italian - REVISED.tsv"),
        "zh": Translation(
            source_path / "[IWSLT 2025] Test Set - TRANSCRIPT_chinese - REVISED.tsv"),
    }
    abstract_translations = {
        "de": AbstractTranslation(
            source_path / "[IWSLT 2025] Test Set - SSUM_german - REVISED.tsv"),
        "it": AbstractTranslation(
            source_path / "[IWSLT 2025] Test Set - SSUM_italian - REVISED.tsv"),
        "zh": AbstractTranslation(
            source_path / "[IWSLT 2025] Test Set - SSUM_chinese - REVISED.tsv"),
    }
    video_ids = set()
    qa_types = {"A", "AV"}
    if include_video:
        qa_types.add("V")
    # Read test elements from the TSV definition
    with open(source_path / TEST_SET_DEF_FNAME, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in reader:
            test_item_def = TestsetDefinitionLine(line)
            video_id = test_item_def.video_id()
            if test_item_def.question_id() == 1:
                current_audio_path = source_path / "AUDIO" / test_item_def.audio()
                assert current_audio_path.exists(), f"{current_audio_path} not found."
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "langs": {
                        "en": {
                            "instruction": instruction_builder.asr(),
                            "random_instruction": random_instruction_builder.asr(),
                            "reference": test_item_def.transcript()
                        }
                    },
                    "task": "ASR",
                    "iid": "ASR_" + str(video_id),
                    "short_audio_segments": audio_segments.audio_to_segments[test_item_def.audio()]
                })
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "langs": {
                        lang: {
                            "instruction": instruction_builder.st(lang=lang),
                            "random_instruction": random_instruction_builder.st(lang=lang),
                            "reference": translations[lang][video_id]["translation"],
                            "transcript": translations[lang][video_id]["transcript"],
                        }
                        for lang in TGT_LANGS
                    },
                    "task": "TRANS",
                    "iid": "TRANS_" + str(video_id),
                    "short_audio_segments": audio_segments.audio_to_segments[test_item_def.audio()]
                })
                langs = {
                    "en": {
                        "instruction": instruction_builder.ssum(),
                        "random_instruction": random_instruction_builder.ssum(),
                        "reference": test_item_def.abstract()
                    }
                }
                for lang in TGT_LANGS:
                    langs[lang] = {
                        "instruction": instruction_builder.ssum(lang=lang),
                        "random_instruction": random_instruction_builder.ssum(lang=lang),
                        "reference": abstract_translations[lang][video_id]
                    }
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "langs": langs,
                    "task": "SUM",
                    "iid": "SUM_" + str(video_id)
                })
                video_ids.add(video_id)
            if test_item_def.question_type() in qa_types:
                corresponding_audio_segments = audio_segments.corresponding_segments(
                    test_item_def.audio(),
                    test_item_def.answer_start(),
                    test_item_def.answer_end()
                )
                assert len(corresponding_audio_segments) > 0, \
                    f"No audio segment for question {test_item_def.unique_id()}"
                if len(corresponding_audio_segments) > 1:
                    logger.debug(
                        f"Question {test_item_def.unique_id()} [{test_item_def.answer_start()}"
                        f"-{test_item_def.answer_end()}] is associated with multiple speech "
                        f"segments: {corresponding_audio_segments}")
                    total_duration = total_segments_duration(corresponding_audio_segments)
                    if total_duration > 30.:
                        logger.warning(
                            f"Skipping question {test_item_def.unique_id()} with duration "
                            f"{total_duration} seconds.")
                        continue
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "langs": {
                        lang: {
                            "instruction": instruction_builder.sqa(
                                test_item_def.question(lang=lang), lang=lang),
                            "random_instruction": random_instruction_builder.sqa(
                                test_item_def.question(lang=lang), lang=lang),
                            "reference": test_item_def.answer(lang=lang)
                        }
                        for lang in {"en"}.union(TGT_LANGS)
                    },
                    "task": "QA",
                    "iid": "QA_" + str(test_item_def.unique_id()),
                    "type": test_item_def.question_type(),
                    "origin": test_item_def.question_origin(),
                    "short_audio_segments": corresponding_audio_segments
                })
            elif test_item_def.question_type() == "NA":
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "langs": {
                        lang: {
                            "instruction": instruction_builder.sqa(
                                test_item_def.question(lang=lang), lang=lang),
                            "random_instruction": random_instruction_builder.sqa(
                                test_item_def.question(lang=lang), lang=lang),
                            "reference": LANG_NOT_ANSWERABLE[lang]
                        }
                        for lang in {"en"}.union(TGT_LANGS)
                    },
                    "task": "QA",
                    "iid": "QA_" + str(test_item_def.unique_id()),
                    "type": "NA",
                    "origin": test_item_def.question_origin(),
                    "short_audio_segments": [random.choice(
                        audio_segments.audio_to_segments[test_item_def.audio()])]
                })
    with open(source_path / TEST_SET_SSUM_DEF_FNAME, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for line in reader:
            test_item_def = TestsetDefinitionLine(line)
            if test_item_def.line_id() not in video_ids:
                langs = {
                    "en": {
                        "instruction": instruction_builder.ssum(),
                        "random_instruction": random_instruction_builder.ssum(),
                        "reference": test_item_def.abstract()
                    }
                }
                for lang in TGT_LANGS:
                    langs[lang] = {
                        "instruction": instruction_builder.ssum(lang=lang),
                        "random_instruction": random_instruction_builder.ssum(lang=lang),
                        "reference": abstract_translations[lang][video_id]
                    }
                test_elements.append({
                    "audio": test_item_def.audio(),
                    "langs": langs,
                    "task": "SUM",
                    "iid": "SUM_" + str(test_item_def.line_id())
                })
    return test_elements


def long_track(
        test_elements: List[Dict[str, Any]],
        source_path: Path,
        output_path: Path,
        include_video: bool = False,
        include_text: bool = False) -> Dict[str, str]:
    """
    Writes the src and ref files for the long track in the `output_path`.
    """
    audio_to_alias = AudioToAlias()
    xml_src = {}
    xml_src_rand = {}
    xml_ref = {}
    xml_src_track = {}
    xml_src_track_rand = {}
    xml_ref_track = {}
    transcripts_map = {}
    if include_text:
        for test_element in test_elements:
            if test_element["task"] == "ASR":
                audio_path = audio_to_alias[test_element["audio"]]
                transcripts_map[audio_path] = test_element["langs"]["en"]["reference"]
    for lang in {"en"}.union(TGT_LANGS):
        xml_src[lang] = ET.Element("testset", attrib={'name': "IWSLT2025"})
        xml_src_rand[lang] = ET.Element("testset", attrib={'name': "IWSLT2025"})
        xml_ref[lang] = ET.Element("testset", attrib={'name': "IWSLT2025"})
        xml_src_track[lang] = ET.SubElement(
            xml_src[lang], "task", attrib={"track": "long", "text_lang": lang})
        xml_src_track_rand[lang] = ET.SubElement(
            xml_src_rand[lang], "task", attrib={"track": "long", "text_lang": lang})
        xml_ref_track[lang] = ET.SubElement(
            xml_ref[lang], "task", attrib={"track": "long", "text_lang": lang})
    for sample_id, sample in enumerate(test_elements):
        for lang in sample["langs"]:
            xml_src_sample = ET.SubElement(
                xml_src_track[lang], "sample", attrib={'id': str(sample_id)})
            xml_src_sample_rand = ET.SubElement(
                xml_src_track_rand[lang], "sample", attrib={'id': str(sample_id)})
            audio_path = audio_to_alias[sample["audio"]]
            ET.SubElement(xml_src_sample, "audio_path").text = audio_path
            ET.SubElement(xml_src_sample_rand, "audio_path").text = audio_path
            if include_video:
                ET.SubElement(xml_src_sample, "video_path").text = audio_path.replace("wav", "mp4")
                ET.SubElement(xml_src_sample_rand, "video_path").text = \
                    audio_path.replace("wav", "mp4")
            if sample["task"] != "ASR" and audio_path in transcripts_map:
                ET.SubElement(xml_src_sample, "text_path").text = audio_path.replace("wav", "en")
                ET.SubElement(xml_src_sample_rand, "text_path").text = \
                    audio_path.replace("wav", "en")
            ET.SubElement(xml_src_sample, "instruction").text = \
                sample["langs"][lang]["instruction"]
            ET.SubElement(xml_src_sample_rand, "instruction").text = \
                sample["langs"][lang]["random_instruction"]

            attribs = {'id': str(sample_id), "iid": sample["iid"], "task": sample["task"]}
            if sample["task"] == "QA":
                attribs["qa_type"] = sample["type"]
                attribs["qa_origin"] = sample["origin"]
            xml_ref_sample = ET.SubElement(xml_ref_track[lang], "sample", attrib=attribs)
            ET.SubElement(xml_ref_sample, "audio_path").text = audio_path
            if include_video:
                ET.SubElement(xml_ref_sample, "video_path").text = audio_path.replace("wav", "mp4")
            if sample["task"] != "ASR" and audio_path in transcripts_map:
                ET.SubElement(xml_ref_sample, "text_path").text = audio_path.replace("wav", "en")
            ET.SubElement(xml_ref_sample, "reference").text = sample["langs"][lang]["reference"]
            if sample["task"] == "TRANS":
                xml_metadata = ET.SubElement(xml_ref_sample, "metadata")
                ET.SubElement(xml_metadata, "transcript").text = \
                    sample["langs"][lang]["transcript"]

    for lang in {"en"}.union(TGT_LANGS):
        tree_src = ET.ElementTree(xml_src[lang])
        tree_src_rand = ET.ElementTree(xml_src_rand[lang])
        tree_ref = ET.ElementTree(xml_ref[lang])
        ET.indent(tree_src)
        ET.indent(tree_src_rand)
        ET.indent(tree_ref)
        tree_src.write(
            output_path / f"IWSLT2025.IF.long.{lang}.src.fixedprompts.xml",
            encoding="utf-8",
            xml_declaration=True)
        tree_src_rand.write(
            output_path / f"IWSLT2025.IF.long.{lang}.src.randomprompts.xml",
            encoding="utf-8",
            xml_declaration=True)
        tree_ref.write(
            output_path / f"IWSLT2025.IF.long.{lang}.ref.xml",
            encoding="utf-8",
            xml_declaration=True)

    base_audio_path = source_path / "AUDIO"
    base_video_path = source_path / "VIDEO"
    output_audio_path = output_path / "LONG_AUDIOS"
    output_video_path = output_path / "LONG_VIDEOS"
    output_text_path = output_path / "LONG_TEXTS"
    output_audio_path.mkdir()
    if include_video:
        output_video_path.mkdir()
    if include_text:
        output_text_path.mkdir()
    for original_name in audio_to_alias:
        shutil.copyfile(
            base_audio_path / original_name, output_audio_path / audio_to_alias[original_name])
        if include_video:
            shutil.copyfile(
                base_video_path / original_name.replace("wav", "mp4"),
                output_video_path / audio_to_alias[original_name].replace("wav", "mp4"))
    if include_text:
        for original_name, transcript in transcripts_map.items():
            fname = original_name.replace("wav", "en")
            with open(output_text_path / fname, "w") as f:
                f.write(transcript)

    return audio_to_alias.names_map


def short_track(
        test_elements: List[Dict[str, Any]],
        long_audio_map: Dict[str, str],
        source_path: Path,
        output_path: Path,
        include_video: bool = False) -> None:
    """
    Writes the src and ref files for the short track in the `output_path`.
    """
    audio_to_alias = AudioToAlias()
    short_segments_path = source_path / "SEGMENTED_AUDIO" / "shas_segments"
    short_video_segments_path = source_path / "SEGMENTED_VIDEO"
    audio_output_path = output_path / "SHORT_AUDIOS"
    audio_output_path.mkdir()
    video_output_path = output_path / "SHORT_VIDEOS"
    if include_video:
        video_output_path.mkdir()
    xml_src = {}
    xml_src_rand = {}
    xml_ref = {}
    xml_src_track = {}
    xml_src_track_rand = {}
    xml_ref_track = {}
    for lang in {"en"}.union(TGT_LANGS):
        xml_src[lang] = ET.Element("testset", attrib={'name': "IWSLT2025"})
        xml_src_rand[lang] = ET.Element("testset", attrib={'name': "IWSLT2025"})
        xml_ref[lang] = ET.Element("testset", attrib={'name': "IWSLT2025"})
        xml_src_track[lang] = ET.SubElement(
            xml_src[lang], "task", attrib={"track": "short", "text_lang": lang})
        xml_src_track_rand[lang] = ET.SubElement(
            xml_src_rand[lang], "task", attrib={"track": "short", "text_lang": lang})
        xml_ref_track[lang] = ET.SubElement(
            xml_ref[lang], "task", attrib={"track": "short", "text_lang": lang})
    sample_id = 0
    for sample in test_elements:
        if sample["task"] == "SUM":
            continue
        if sample["task"] == "ASR" or sample["task"] == "TRANS":
            sample_ids = []
            for short_audio_segm in sample["short_audio_segments"]:
                for lang in sample["langs"]:
                    audio_path = audio_to_alias[short_audio_segm["wav"]]
                    xml_src_sample = ET.SubElement(
                        xml_src_track[lang], "sample", attrib={'id': str(sample_id)})
                    xml_src_sample_rand = ET.SubElement(
                        xml_src_track_rand[lang], "sample", attrib={'id': str(sample_id)})
                    ET.SubElement(xml_src_sample, "audio_path").text = audio_path
                    ET.SubElement(xml_src_sample_rand, "audio_path").text = audio_path
                    if include_video:
                        ET.SubElement(xml_src_sample, "video_path").text = \
                            audio_path.replace(".wav", ".mp4")
                        ET.SubElement(xml_src_sample_rand, "video_path").text = \
                            audio_path.replace(".wav", ".mp4")
                    ET.SubElement(xml_src_sample, "instruction").text = \
                        sample["langs"][lang]["instruction"]
                    ET.SubElement(xml_src_sample_rand, "instruction").text = \
                        sample["langs"][lang]["random_instruction"]
                sample_ids.append(sample_id)
                sample_id += 1
            for lang in sample["langs"]:
                xml_ref_sample = ET.SubElement(
                    xml_ref_track[lang],
                    "sample",
                    attrib={
                        'id': ",".join(str(s) for s in sample_ids),
                        "iid": sample["iid"],
                        "task": sample["task"]})
                ET.SubElement(xml_ref_sample, "audio_path").text = long_audio_map[sample["audio"]]
                if include_video:
                    ET.SubElement(xml_ref_sample, "video_path").text = \
                        long_audio_map[sample["audio"]].replace(".wav", ".mp4")
                ET.SubElement(xml_ref_sample, "reference").text = \
                    sample["langs"][lang]["reference"]
                if sample["task"] == "TRANS":
                    xml_metadata = ET.SubElement(xml_ref_sample, "metadata")
                    ET.SubElement(xml_metadata, "transcript").text = \
                        sample["langs"][lang]["transcript"]
        else:
            assert sample["task"] == "QA", f"Unsupported task {sample['task']}"
            if len(sample["short_audio_segments"]) == 1:
                short_audio = audio_to_alias[sample["short_audio_segments"][0]["wav"]]
                short_video = short_audio.replace(".wav", ".mp4")
            else:
                short_audio = AudioToAlias.get_random_name()
                short_video = short_audio.replace(".wav", ".mp4")
                merge_wav_files(
                    [short_segments_path / s["wav"] for s in sample["short_audio_segments"]],
                    audio_output_path / short_audio)
                if include_video:
                    merge_mp4_files(
                        [short_video_segments_path / s["wav"].replace(".wav", ".mp4")
                         for s in sample["short_audio_segments"]],
                        video_output_path / short_video)
            for lang in sample["langs"]:
                xml_src_sample = ET.SubElement(
                    xml_src_track[lang], "sample", attrib={'id': str(sample_id)})
                xml_src_sample_rand = ET.SubElement(
                    xml_src_track_rand[lang], "sample", attrib={'id': str(sample_id)})
                ET.SubElement(xml_src_sample, "audio_path").text = short_audio
                ET.SubElement(xml_src_sample_rand, "audio_path").text = short_audio
                if include_video:
                    ET.SubElement(xml_src_sample, "video_path").text = short_video
                    ET.SubElement(xml_src_sample_rand, "video_path").text = short_video
                ET.SubElement(xml_src_sample, "instruction").text = \
                    sample["langs"][lang]["instruction"]
                ET.SubElement(xml_src_sample_rand, "instruction").text = \
                    sample["langs"][lang]["random_instruction"]

                xml_ref_sample = ET.SubElement(
                    xml_ref_track[lang],
                    "sample",
                    attrib={
                        "id": str(sample_id),
                        "iid": sample["iid"],
                        "task": sample["task"],
                        "qa_type": sample["type"],
                        "qa_origin": sample["origin"]})
                ET.SubElement(xml_ref_sample, "audio_path").text = short_audio
                if include_video:
                    ET.SubElement(xml_ref_sample, "video_path").text = short_video
                ET.SubElement(xml_ref_sample, "reference").text = \
                    sample["langs"][lang]["reference"]
            sample_id += 1

    for lang in {"en"}.union(TGT_LANGS):
        tree_src = ET.ElementTree(xml_src[lang])
        tree_src_rand = ET.ElementTree(xml_src_rand[lang])
        tree_ref = ET.ElementTree(xml_ref[lang])
        ET.indent(tree_src)
        ET.indent(tree_src_rand)
        ET.indent(tree_ref)
        tree_src.write(
            output_path / f"IWSLT2025.IF.short.{lang}.src.fixedprompt.xml",
            encoding="utf-8",
            xml_declaration=True)
        tree_src_rand.write(
            output_path / f"IWSLT2025.IF.short.{lang}.src.randomprompt.xml",
            encoding="utf-8",
            xml_declaration=True)
        tree_ref.write(
            output_path / f"IWSLT2025.IF.short.{lang}.ref.xml",
            encoding="utf-8",
            xml_declaration=True)

    for original_name in audio_to_alias:
        shutil.copyfile(
            short_segments_path / original_name, audio_output_path / audio_to_alias[original_name])
        if include_video:
            shutil.copyfile(
                short_video_segments_path / original_name.replace(".wav", ".mp4"),
                video_output_path / audio_to_alias[original_name].replace(".wav", ".mp4"))


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
    parser.add_argument(
        "--include-video",
        action="store_true",
        default=False,
        help="add video clips",
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        default=False,
        help="add text source",
    )
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    source_path = Path(args.source_dir)
    # we set the seed to make reproducible the test set generation even though
    # there are random choices
    random.seed(3)  # in read_test_elements we select a random segment fon NA questions
    test_elements = read_test_elements(source_path, args.include_video)
    # shuffle test elements to avoid clear patterns in instructions
    random.seed(42)
    random.shuffle(test_elements)
    # write the XML test definition and reference
    long_audio_map = long_track(
        test_elements, source_path, output_path, args.include_video, args.include_text)
    short_track(test_elements, long_audio_map, source_path, output_path, args.include_video)


if __name__ == "__main__":
    cli_script()
