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
import xml.etree.ElementTree as ET
import wave
import contextlib
import os
from collections import defaultdict


def audio_seconds(audio_path):
    with contextlib.closing(wave.open(audio_path, 'r')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)


def compute_stats(reference, audio_dir, use_chars=False):
    tree = ET.parse(reference)
    root = tree.getroot()

    # === STORAGE ===
    stats = defaultdict(
        lambda: {'reference_count': 0, 'audio_duration_sec': 0.0, 'num_samples': 0})

    # === PROCESS EACH SAMPLE ===
    for sample in root.findall('.//sample'):
        task = sample.attrib.get('task')

        # Count words in reference
        reference_elem = sample.find('reference')
        reference = reference_elem.text.strip() if reference_elem is not None else ""
        if use_chars:
            ref_count = len(reference)
        else:
            ref_count = len(reference.split())

        stats[task]['reference_count'] += ref_count

        # in case of short track, consider each segment as a sample
        for audio_filename in sample.find('audio_path').text.split(","):
            duration_sec = audio_seconds(os.path.join(audio_dir, audio_filename))
            stats[task]['num_samples'] += 1
            stats[task]['audio_duration_sec'] += duration_sec

            if task == "QA":
                qa_type = sample.attrib.get('qa_type')
                qa_origin = sample.attrib.get('qa_origin')
                for subtask in [
                        task + "_" + qa_type,
                        task + "_" + qa_origin,
                        task + "_" + qa_type + "_" + qa_origin]:
                    stats[subtask]['num_samples'] += 1
                    stats[subtask]['reference_count'] += ref_count
                    stats[subtask]['audio_duration_sec'] += duration_sec
    return stats


def print_stats(stats, use_chars=False):
    if use_chars:
        unit = "Characters"
    else:
        unit = "Words"
    header = f"{'Task':<15} {'Num. Samples':>15} "
    for field in ['Audio Duration (sec)', f'Reference {unit}']:
        header += f"{'Total ' + field:>30} {'Avg. ' + field:>30} "
    print(header)
    print('-' * len(header))

    # === PRINT TABLE ROWS ===
    for task in sorted(stats.keys()):
        values = stats[task]
        output = f"{task:<15} {values['num_samples']:>15} "
        for field in ['audio_duration_sec', 'reference_count']:
            output += f"{values[field]:>30.2f} {values[field] / values['num_samples']:>30.2f} "
        print(output)


def cli_script():
    """
    Starting from the XML test set reference, this script computes statistics of the dataset
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--reference', '-r', type=str, required=True,
        help='the path to the the test set reference XML.')
    parser.add_argument(
        '--audio-dir', '-d', type=str, required=True,
        help='the path to the folder containing the test set definition.')
    parser.add_argument(
        "--use-chars",
        action="store_true",
        default=False,
        help="count characters instead of words",
    )
    args = parser.parse_args()
    stats = compute_stats(args.reference, args.audio_dir, args.use_chars)
    print_stats(stats, args.use_chars)


if __name__ == "__main__":
    cli_script()
