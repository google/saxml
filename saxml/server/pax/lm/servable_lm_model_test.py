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
"""Tests for servable_lm_model."""
from absl.testing import absltest
from absl.testing import parameterized
from saxml.server.pax.lm import servable_lm_model


class ServableLmModelTest(parameterized.TestCase):

  @parameterized.parameters(
      ('1', True),
      ('true', True),
      ('t', True),
      ('True', True),
      ('0', False),
      ('false', False),
      ('False', False),
  )
  def test_string_to_bool(self, arg, expected):
    self.assertEqual(servable_lm_model._string_to_bool(arg), expected)

  @parameterized.parameters('hello', '100')
  def test_string_to_bool_invalid_arguments(self, arg):
    with self.assertRaisesRegex(
        ValueError, 'Non-boolean argument to boolean flag'
    ):
      servable_lm_model._string_to_bool(arg)


if __name__ == '__main__':
  absltest.main()
