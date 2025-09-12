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
import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bert_score
import jiwer
from comet import download_model, load_from_checkpoint
from whisper_normalizer import english, basic


CHAR_LEVEL_LANGS = {"zh"}


@dataclass
class ReferenceSample:
    sample_ids: List[str]
    reference: str
    metadata: Dict[str, str] = None


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
            # if the prediction is empty mwerSegmenter returns a segmentation fault, so we put a
            # fake "." to avoid this issue
            if prediction.strip() == "":
                prediction = "."
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

    def read_text(xml_sample):
        if xml_sample.text is None:
            return ""
        return xml_sample.text.strip()

    xml = ET.parse(hypo_path)
    avail_tasks = []
    for task in xml.getroot().iter("task"):
        if task.attrib['track'] == track and task.attrib['text_lang'] == language:
            return {sample.attrib['id']: read_text(sample) for sample in task.iter("sample")}
        avail_tasks.append((task.attrib['track'], task.attrib['text_lang']))
    raise Exception(
        f"Task '{track}' for language '{language}' not available in {hypo_path}. "
        f"Available tasks are: {avail_tasks}.")


def read_reference(
        ref_path: Path,
        track: str,
        language: str,
        modality: Optional[str] = None) -> Dict[str, Dict[str, ReferenceSample]]:
    xml = ET.parse(ref_path)
    avail_tasks = []
    for task in xml.getroot().iter("task"):
        if task.attrib['track'] == track and task.attrib['text_lang'] == language:
            samples_by_subtask = {}
            for sample in task.iter("sample"):
                if modality is None or len(list(sample.iter(modality + '_path'))) > 0:
                    if sample.attrib['task'] not in samples_by_subtask:
                        samples_by_subtask[sample.attrib['task']] = {}
                    sample_ids = sample.attrib['id'].split(",")
                    sample_reference = next(sample.iter('reference')).text
                    sample_metadata = {}
                    for metadata in sample.iter('metadata'):
                        for metadata_field in metadata.iter():
                            sample_metadata[metadata_field.tag] = metadata_field.text
                    for field in ['qa_type', 'qa_origin']:
                        if field in sample.attrib:
                            sample_metadata[field] = sample.attrib[field]
                    samples_by_subtask[sample.attrib['task']][sample.attrib['iid']] = \
                        ReferenceSample(sample_ids, sample_reference, sample_metadata)
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
    if lang == "en":
        std = english.EnglishTextNormalizer()
    else:
        std = basic.BasicTextNormalizer()

    refs, hypos = [], []
    for _, ref_sample in ref_dict["ASR"].items():
        hypo_components = []
        for sample_id in ref_sample.sample_ids:
            hypo_components.append(std(hypo_dict[sample_id]))
        refs.append(std(ref_sample.reference))
        hypos.append(" ".join(hypo_components))
    return jiwer.wer(refs, hypos)


def score_sqa(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, Dict[str, ReferenceSample]],
        lang: str,
        breakdown_qa_types: bool) -> Tuple[float, Optional[Dict[str, float]]]:
    return bertscore(hypo_dict, ref_dict, lang, "QA", breakdown_qa_types)


def score_ssum(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, Dict[str, ReferenceSample]],
        lang: str) -> float:
    score, _ = bertscore(hypo_dict, ref_dict, lang, "SUM")
    return score


def bertscore(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, Dict[str, ReferenceSample]],
        lang: str,
        task: str,
        breakdown_qa_types: bool = False) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Computes BERTScore.
    """
    refs, hypos = [], []
    qa_types_indices = {}
    for i, (iid, ref_sample) in enumerate(ref_dict[task].items()):
        assert len(ref_sample.sample_ids) == 1, \
            f"{task} reference (IID: {iid}) mapped to multiple samples ids: " \
            f"{ref_sample.sample_ids}"
        hypos.append(hypo_dict[ref_sample.sample_ids[0]])
        refs.append(ref_sample.reference)
        if breakdown_qa_types:
            for field in ['qa_type', 'qa_origin']:
                qa_type = ref_sample.metadata[field]
                if qa_type not in qa_types_indices:
                    qa_types_indices[qa_type] = []
                qa_types_indices[qa_type].append(i)

    P, R, F1 = bert_score.score(hypos, refs, lang=lang, rescale_with_baseline=True)
    qa_types_scores = None
    if breakdown_qa_types:
        qa_types_scores = {
            qa_type: F1[indices].mean().detach().item()
            for qa_type, indices in qa_types_indices.items()
        }
    return F1.mean().detach().item(), qa_types_scores


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
    for iid, ref_sample in ref_dict["TRANS"].items():
        ref_lines = ref_sample.reference.split("\n")
        src_lines = ref_sample.metadata["transcript"].split("\n")
        assert len(ref_lines) == len(src_lines), \
            f"TRANS reference (IID: {iid}) has mismatched number of target ({len(ref_lines)}) " \
            f"and source lines ({len(src_lines)})"
        hypo_components = []
        for sample_id in ref_sample.sample_ids:
            hypo_components.append(hypo_dict[sample_id])

        resegm_hypos = mwer_segmeter("\n".join(hypo_components), ref_lines)
        assert len(ref_lines) == len(resegm_hypos), \
            f"TRANS reference (IID: {iid}) has mismatched number of target ({len(ref_lines)})" \
            f" and resegmented lines ({len(resegm_hypos)})"
        for hyp, ref, src in zip(resegm_hypos, ref_lines, src_lines):
            comet_data.append({
                "src": src.strip(),
                "mt": hyp.strip(),
                "ref": ref.strip()
            })
    return comet_score(comet_data)


def main(
        hypo_path: Path,
        ref_path: Path,
        track: str,
        lang: str,
        filter_modality: Optional[str],
        breakdown_qa_types: bool) -> Dict[str, float]:
    """
    Main function computing all the scores and returning a Dictionary with the scores
    """
    hypo = read_hypo(hypo_path, track, lang)
    ref = read_reference(ref_path, track, lang, modality=filter_modality)
    scores = {}
    assert "QA" in ref.keys()
    scores["QA-BERTScore"], qa_types_scores = score_sqa(hypo, ref, lang, breakdown_qa_types)
    if qa_types_scores is not None:
        for qa_type, score in qa_types_scores.items():
            scores[f"QA-{qa_type}-BERTScore"] = score

    # sanity checks for the IWSLT25 task
    if track == "short":
        assert len(ref.keys()) == 2
        if lang == "en":
            assert "ASR" in ref.keys()
            scores["ASR-WER"] = score_asr(hypo, ref, lang)
        else:
            assert "TRANS" in ref.keys()
            scores["TRANS-COMET"] = score_st(hypo, ref, lang)
    else:
        assert len(ref.keys()) == 3 or len(ref.keys()) == 2
        assert "SUM" in ref.keys()
        scores["SUM-BERTScore"] = score_ssum(hypo, ref, lang)
        if lang == "en":
            if "ASR" in ref.keys():
                scores["ASR-WER"] = score_asr(hypo, ref, lang)
        else:
            assert "TRANS" in ref.keys()
            scores["TRANS-COMET"] = score_st(hypo, ref, lang)
    return scores


def cli_script():
    """
    Script that evaluates the outputs of a system in XML format against the MCIF reference.
    By default, the evaluation is carried out on all the test elements, but the evaluation can be
    limited to the tasks/samples relevant for one modality by means of the --filter-modality param.
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
    parser.add_argument(
        '--filter-modality', '-m', choices=["audio", "text", "video"], default=None,
        help="consider only samples which have this modality")
    parser.add_argument(
        '--breakdown-qa-types', default=False, action='store_true',
        help="if set, print separate scores for different QA types")
    args = parser.parse_args()
    try:
        hypo_path = Path(args.hypothesis)
        ref_path = Path(args.reference)
        scores = main(
            hypo_path,
            ref_path,
            args.track,
            args.language,
            args.filter_modality,
            args.breakdown_qa_types)
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
