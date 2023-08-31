# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for proto_util_test."""

from absl.testing import absltest
from absl.testing import parameterized
from saxml.server import proto_util


class ProtoUtilTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('?x?x?', '?x?x?', None),
      ('2', '2', 2),
      ('8x16', '8x16', 128),
      ('16x16', '16x16', 256),
      ('16x32', '16x32', 512),
      ('4x4x16', '4x4x16', 256),
      ('4x8x8', '4x8x8', 256),
      ('8x8x12', '8x8x12', 768),
      ('8x8x12_twisted', '8x8x12_twisted', 768),
  )
  def test_count_physical_chips(self, topology, chips):
    self.assertEqual(chips, proto_util.count_physical_chips(topology))


if __name__ == '__main__':
  absltest.main()
