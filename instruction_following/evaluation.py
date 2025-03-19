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
import json
import os
import re
import shutil
import string
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import namedtuple
from pathlib import Path
from typing import Dict, List

import bert_score
import jiwer
from comet import download_model, load_from_checkpoint


CHAR_LEVEL_LANGS = {"zh"}

ReferenceSample = namedtuple('ReferenceSample', ['sample_ids', 'reference'])


class MwerSegmenter:
    """
    Executes the mWERSegmenter tool introduced in `"Evaluating Machine Translation Output
    with Automatic Sentence Segmentation" by Matusov et al. (2005)
    <https://aclanthology.org/2005.iwslt-1.19/>`_.

    The tool can be downloaded at:
    https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
    """
    def __init__(self, character_level=False):
        self.mwer_command = "mwerSegmenter"
        self.character_level = character_level
        if shutil.which(self.mwer_command) is None:
            mwerSegmenter_root = os.getenv("MWERSEGMENTER_ROOT")
            assert mwerSegmenter_root is not None, \
                f"{self.mwer_command} is not in PATH and no MWERSEGMENTER_ROOT environment " \
                "variable is set"
            self.mwer_command = mwerSegmenter_root + "/mwerSegmenter"

    def __call__(self, prediction: str, reference_sentences: List[str]) -> List[str]:
        """
        Segments the prediction based on the reference sentences using the edit distance algorithm.
        """
        tmp_pred = tempfile.NamedTemporaryFile(mode="w", delete=False)
        tmp_ref = tempfile.NamedTemporaryFile(mode="w", delete=False)
        if self.character_level:
            # If character-level evaluation, add spaces for resegmentation
            prediction = " ".join(prediction)
            reference_sentences = [" ".join(reference) for reference in reference_sentences]
        try:
            tmp_pred.write(prediction)
            tmp_ref.writelines(ref + '\n' for ref in reference_sentences)
            tmp_pred.flush()
            tmp_ref.flush()
            subprocess.run([
                self.mwer_command,
                "-mref",
                tmp_ref.name,
                "-hypfile",
                tmp_pred.name,
                "-usecase",
                "1"])
            # mwerSegmenter writes into the __segments file of the current working directory
            with open("__segments") as f:
                segments = []
                for line in f.readlines():
                    if self.character_level:
                        # If character-level evaluation, remove only spaces between characters
                        line = re.sub(r'(.)\s', r'\1', line)
                    segments.append(line.strip())
                return segments
        finally:
            tmp_pred.close()
            tmp_ref.close()
            os.unlink(tmp_pred.name)
            os.unlink(tmp_ref.name)
            os.unlink("__segments")


def read_hypo(hypo_path: Path, track: str, language: str) -> Dict[str, str]:
    xml = ET.parse(hypo_path)
    avail_tasks = []
    for task in xml.getroot().iter("task"):
        if task.attrib['track'] == track and task.attrib['text_lang'] == language:
            return {sample.attrib['id']: sample.text.strip() for sample in task.iter("sample")}
        avail_tasks.append((task.attrib['track'], task.attrib['text_lang']))
    raise Exception(
        f"Task '{track}' for language '{language}' not available in {hypo_path}. "
        f"Available tasks are: {avail_tasks}.")


def read_reference(
        ref_path: Path, track: str, language: str) -> Dict[str, Dict[str, ReferenceSample]]:
    xml = ET.parse(ref_path)
    avail_tasks = []
    for task in xml.getroot().iter("task"):
        if task.attrib['track'] == track and task.attrib['text_lang'] == language:
            samples_by_subtask = {}
            for sample in task.iter("sample"):
                if sample.attrib['task'] not in samples_by_subtask:
                    samples_by_subtask[sample.attrib['task']] = {}
                samples_by_subtask[sample.attrib['task']][sample.attrib['iid']] = ReferenceSample(
                    sample.attrib['id'].split(","), next(sample.iter('reference')).text)
            return samples_by_subtask
        avail_tasks.append((task.attrib['track'], task.attrib['text_lang']))
    raise Exception(
        f"Task '{track}' for language '{language}' not available in {ref_path}. "
        f"Available tasks are: {avail_tasks}.")


def score_asr(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, Dict[str, ReferenceSample]],
        lang: str) -> float:
    """
    Computes WER after removing punctuation and lowercasing. No tokenization is performed.
    """
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")

    def preprocess(text: str) -> str:
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # convert newlines to spaces and remove duplicated spaces
        text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()
        # Convert to lowercase
        return text.lower()

    refs, hypos = [], []
    for _, ref_sample in ref_dict["ASR"].items():
        hypo_components = []
        for sample_id in ref_sample.sample_ids:
            hypo_components.append(preprocess(hypo_dict[sample_id]))
        refs.append(preprocess(ref_sample.reference))
        hypos.append(" ".join(hypo_components))
    return jiwer.wer(refs, hypos)


def score_sqa(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, Dict[str, ReferenceSample]],
        lang: str) -> float:
    return bertscore(hypo_dict, ref_dict, lang, "SQA")


def score_ssum(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, Dict[str, ReferenceSample]],
        lang: str) -> float:
    return bertscore(hypo_dict, ref_dict, lang, "SSUM")


def bertscore(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, Dict[str, ReferenceSample]],
        lang: str,
        task: str) -> float:
    """
    Computes BERTScore.
    """
    refs, hypos = [], []
    for iid, ref_sample in ref_dict[task].items():
        assert len(ref_sample.sample_ids) == 1, \
            f"SQA reference (IID: {iid}) mapped to multiple samples ids: {ref_sample.sample_ids}"
        hypos.append(hypo_dict[ref_sample.sample_ids[0]])
        refs.append(ref_sample.reference)

    # since BERTScore supports a dedicated model for scientific text, we use it as it aligns with
    # out domain data. This should be revisited if used with different type of data
    if lang == "en":
        lang = "en-sci"
    P, R, F1 = bert_score.score(hypos, refs, lang=lang, rescale_with_baseline=True)
    return F1.mean().detach().item()


def comet_score(data: List[Dict[str, str]]) -> float:
    """
    Computes COMET starting from a List of Dictionary, each containing the "mt", "src", and "ref"
    keys.
    """
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model.eval()
    model_output = model.predict(data, batch_size=8, gpus=1)
    return model_output.system_score


def score_st(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, Dict[str, ReferenceSample]],
        lang: str) -> float:
    """
    Computes COMET.
    """
    comet_data = []
    mwer_segmeter = MwerSegmenter(character_level=(lang in CHAR_LEVEL_LANGS))
    for iid, ref_sample in ref_dict["ST"].items():
        ref_lines = ref_sample.reference.split("\n")
        src_lines = ref_dict["ASR"][iid].reference.split("\n")
        assert len(ref_lines) == len(src_lines), \
            f"ST reference (IID: {iid}) has mismatched number of target ({len(ref_lines)}) and " \
            f"source lines ({len(src_lines)})"
        hypo_components = []
        for sample_id in ref_sample.sample_ids:
            hypo_components.append(hypo_dict[sample_id])

        resegm_hypos = mwer_segmeter("\n".join(hypo_components), ref_lines)
        assert len(ref_lines) == len(resegm_hypos), \
            f"ST reference (IID: {iid}) has mismatched number of target ({len(resegm_hypos)}) " \
            f"and resegmented lines ({len(resegm_hypos)})"
        for hyp, ref, src in zip(resegm_hypos, ref_lines, src_lines):
            comet_data.append({
                "src": src.strip(),
                "mt": hyp.strip(),
                "ref": ref.strip()
            })
    return comet_score(comet_data)


def main(hypo_path: Path, ref_path: Path, track: str, lang: str) -> Dict[str, float]:
    """
    Main function computing all the scores and returning a Dictionary with the scores
    """
    hypo = read_hypo(hypo_path, track, lang)
    ref = read_reference(ref_path, track, lang)
    scores = {}
    # sanity checks for the IWSLT25 task
    if track == "short":
        assert len(ref.keys()) == 2
        assert "SQA" in ref.keys()
        scores["SQA-BERTScore"] = score_sqa(hypo, ref, lang)
        if lang == "en":
            assert "ASR" in ref.keys()
            scores["ASR-WER"] = score_asr(hypo, ref, lang)
        else:
            assert "ST" in ref.keys()
            scores["ST-COMET"] = score_st(hypo, ref, lang)
    else:
        assert len(ref.keys()) == 3
        assert "SQA" in ref.keys()
        assert "SSUM" in ref.keys()
        scores["SQA-BERTScore"] = score_sqa(hypo, ref, lang)
        scores["SSUM-BERTScore"] = score_ssum(hypo, ref, lang)
        if lang == "en":
            assert "ASR" in ref.keys()
            scores["ASR-WER"] = score_asr(hypo, ref, lang)
        else:
            assert "ST" in ref.keys()
            scores["ST-COMET"] = score_st(hypo, ref, lang)
    return scores


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
        '--hypothesis', '-s', type=str, required=True,
        help="the hypothesis to be scored")
    parser.add_argument(
        '--reference', '-r', type=str, required=True,
        help='the path to the folder containing the test set definition.')
    parser.add_argument(
        '--track', '-t', choices=["short", "long"], required=True,
        help="the track for the hypothesis")
    parser.add_argument(
        '--language', '-l', type=str, required=True,
        help="the target language to evaluate")
    args = parser.parse_args()
    try:
        hypo_path = Path(args.hypothesis)
        ref_path = Path(args.reference)
        scores = main(hypo_path, ref_path, args.track, args.language)
        print(json.dumps({
            "state": "OK",
            "scores": scores
        }))
    except Exception as e:  # noqa
        print(json.dumps({
            "state": "ERROR",
            "reason": str(e),
            "scores": {}
        }))


if __name__ == "__main__":
    cli_script()
