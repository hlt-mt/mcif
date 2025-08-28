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
from itertools import groupby
from pathlib import Path
from typing import Dict, List

from moviepy.video.io.VideoFileClip import VideoFileClip
import yaml


def read_shas(shas_definition) -> List[Dict[str, str]]:
    with open(shas_definition, 'r', encoding="utf8") as f:
        try:
            sentences = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)
    return sentences


def split_mp4s(source_path: Path, output_path: Path, shas_segm: List[Dict[str, str]]):
    for wav_filename, _seg_group in groupby(shas_segm, lambda x: x["wav"]):
        mp4_path = source_path / wav_filename.replace(".wav", ".mp4")
        seg_group = sorted(_seg_group, key=lambda x: float(x["offset"]))
        with VideoFileClip(mp4_path) as clip:
            for i, split in enumerate(seg_group):
                start = split["offset"]
                end = start + split["duration"]
                subclip = clip.subclipped(start, end)
                subclip.write_videofile(
                    output_path / wav_filename.replace(".wav", f"_{i}.mp4"), audio_codec='aac')


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
        help='the path to the folder containing the source videos.')
    parser.add_argument(
        '--shas-definition', '-d', type=str, required=True,
        help='the path to the SHAS yaml file containing the segmentation definition.')
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    source_path = Path(args.source_dir)

    shas_segm = read_shas(args.shas_definition)
    split_mp4s(source_path, output_path, shas_segm)


if __name__ == "__main__":
    cli_script()
