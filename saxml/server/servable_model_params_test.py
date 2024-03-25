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
"""Tests for servable_model_params."""

from absl.testing import absltest
from saxml.server import servable_model_params


class ServableModelParamsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    servable_model_params.ServableModelParams.__abstractmethods__ = set()
    self.params = servable_model_params.ServableModelParams()

  def test_overrides(self):
    params = self.params
    params.INT_KEY = 42
    params.STR_KEY = "hi there"
    params.LIST_KEY = [128, 256]
    params.ANOTHER_LIST_KEY = [1, 2]
    params.apply_model_overrides(dict(
        INT_KEY="100",
        STR_KEY="\"foo\"",
        LIST_KEY="[55, 65, 75]",
    ))
    self.assertEqual(params.INT_KEY, 100)
    self.assertEqual(params.STR_KEY, "foo")
    self.assertEqual(params.LIST_KEY, [55, 65, 75])
    self.assertEqual(params.ANOTHER_LIST_KEY, [1, 2])

  def test_skip_on_missing_field(self):
    params = self.params
    params.INT_KEY = 42
    params.apply_model_overrides(dict(ANOTHER_INT_KEY="100",))
    self.assertEqual(params.INT_KEY, 42)

  def test_exception_on_different_type(self):
    params = self.params
    params.INT_KEY = 42
    self.assertRaises(ValueError, params.apply_model_overrides, dict(
        INT_KEY="false",))


if __name__ == "__main__":
  absltest.main()
