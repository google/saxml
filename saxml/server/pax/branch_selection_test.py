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
"""Tests for branch_selection."""

from absl.testing import absltest
from saxml.server.pax import branch_selection


class BranchSelectionTest(absltest.TestCase):

  def test_multiple_user_keys(self):
    seq_lens = [4, 8]
    branch_selector = branch_selection.BranchSelector(keys=seq_lens)
    self.assertEqual(branch_selector.branch_keys, [4, 8])
    self.assertTrue(branch_selector.has_multiple_branches())
    self.assertEqual(branch_selector.get_branch_index(key=2), 0)
    self.assertEqual(branch_selector.get_branch_index(key=4), 0)
    self.assertEqual(branch_selector.get_branch_index(key=5), 1)
    self.assertEqual(branch_selector.get_branch_index(key=8), 1)

  def test_multiple_user_keys_tf(self):
    seq_lens = [4, 8]
    branch_selector = branch_selection.BranchSelector(keys=seq_lens)
    self.assertEqual(branch_selector.branch_keys, [4, 8])
    self.assertTrue(branch_selector.has_multiple_branches())
    self.assertEqual(branch_selector.get_branch_index_tf(key=2), 0)
    self.assertEqual(branch_selector.get_branch_index_tf(key=4), 0)
    self.assertEqual(branch_selector.get_branch_index_tf(key=5), 1)
    self.assertEqual(branch_selector.get_branch_index_tf(key=8), 1)


if __name__ == '__main__':
  absltest.main()
