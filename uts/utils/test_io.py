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
from io import StringIO, BytesIO

from mcif.io import write_output, OutputSample


class TestIO(unittest.TestCase):

    def test_xml_writing(self):
        samples = [OutputSample(1, "hello"), OutputSample(10, "world"), OutputSample(2, "!")]
        buffer = BytesIO()
        write_output(samples, "long", "en", "example", buffer)
        self.assertEqual(buffer.getvalue().decode("utf-8"), """<?xml version=\'1.0\' encoding=\'utf-8\'?>
<testset name="example" type="output">
  <task track="long" text_lang="en">
    <sample id="1">hello</sample>
    <sample id="10">world</sample>
    <sample id="2">!</sample>
  </task>
</testset>""")


if __name__ == '__main__':
    unittest.main()