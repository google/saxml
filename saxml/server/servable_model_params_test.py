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

from saxml.server import servable_model_params
from google3.testing.pybase import googletest


class ServableModelParamsTest(googletest.TestCase):

  def test_overrides(self):
    servable_model_params.ServableModelParams.__abstractmethods__ = set()
    params = servable_model_params.ServableModelParams()
    params.INT_KEY = 42
    params.STR_KEY = "hi there"
    params.LIST_KEY = [128, 256]
    params.ANOTHER_LIST_KEY = [1, 2]
    params.apply_model_overrides(dict(
        INT_KEY="100",
        STR_KEY="foo",
        LIST_KEY="55,65,75"
    ))
    self.assertEqual(params.INT_KEY, 100)
    self.assertEqual(params.STR_KEY, "foo")
    self.assertEqual(params.LIST_KEY, [55, 65, 75])
    self.assertEqual(params.ANOTHER_LIST_KEY, [1, 2])


if __name__ == "__main__":
  googletest.main()
