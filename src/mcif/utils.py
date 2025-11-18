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
import gzip
import logging
import tempfile
import urllib.request

from pathlib import Path
from typing import Optional

from mcif import __benchmark_version__, _HF_BASE_REPO


LOGGER = logging.getLogger('mcif.utils')


def resolve_reference(
        reference_path: Optional[str],
        language: str,
        track: str,
        version: Optional[str]) -> Path:
    if reference_path is not None:
        return Path(reference_path)
    version = version if version is not None else __benchmark_version__
    if version == '1.0':
        file_name = f"MCIF{version}.{track}.{language}.ref.xml.gz"
    else:
        file_name = f"MCIF.{track}.{language}.ref.xml.gz"
    url = _HF_BASE_REPO + f"resolve/{version}/{file_name}?download=true"
    LOGGER.info(f"Downloading reference from Huggingface ({url})...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        with urllib.request.urlopen(url) as response:
            with gzip.GzipFile(fileobj=response) as gz:
                tmp_file.write(gz.read())
        return Path(tmp_file.name)
