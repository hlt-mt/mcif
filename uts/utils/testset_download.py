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
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from itertools import product

from mcif.utils import resolve_reference


class TestResolveReference(unittest.TestCase):
    """Unit tests for the resolve_reference function."""

    # Define parameter sets
    LANGUAGES = ["it", "de", "zh", "en"]
    TRACKS = ["long", "short"]
    VERSIONS = [None, "1.0"]

    def setUp(self):
        """Keep track of all generated temp files for later cleanup."""
        self._temp_files = []

    def tearDown(self):
        """Clean up all temp files created during tests."""
        for path in self._temp_files:
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass  # Don’t fail cleanup

    def _check_xml_structure(self, xml_path: Path, language: str, track: str):
        """Helper: validate the XML file structure."""
        # File must exist
        self.assertTrue(xml_path.exists(), f"File not found: {xml_path}")

        # Parse XML
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as e:
            self.fail(f"Invalid XML returned by resolve_reference: {e}")

        root = tree.getroot()
        self.assertEqual(root.tag, "testset", f"Root element must be <testset>, got <{root.tag}>")

        tasks = root.findall("task")
        self.assertTrue(tasks, "No <task> elements found in XML")

        for task in tasks:
            self.assertEqual(
                task.get("track"), track, f"Expected track='{track}', got '{task.get('track')}'")
            self.assertEqual(
                task.get("text_lang"),
                language,
                f"Expected text_lang='{language}', got '{task.get('text_lang')}'")

    def test_resolve_reference_combinations(self):
        """Test resolve_reference with all combinations of parameters."""
        for language, track, version in product(self.LANGUAGES, self.TRACKS, self.VERSIONS):
            with self.subTest(language=language, track=track, version=version):
                result = resolve_reference(None, language, track, version)
                self._temp_files.append(result)
                self._check_xml_structure(result, language, track)


if __name__ == '__main__':
    unittest.main()
