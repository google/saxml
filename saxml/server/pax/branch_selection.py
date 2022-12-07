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

"""Helper class for branch selection."""

from typing import List
import numpy as np


class BranchSelector(object):
  """Util class to help select branch index.

  typical usage:

  seq_lens = [8, 16]
  selector = BranchSelector(keys=seq_lens)

  # branch_keys [8, 16]
  branch_keys = selector.branch_keys

  seq_len = 5
  # branch_index == 0
  branch_index = selector.get_branch_index(seq_len)
  """

  def __init__(self, keys: List[int]):
    keys.sort()
    self._keys = keys

  @property
  def branch_keys(self) -> List[int]:
    return self._keys

  def has_multiple_branches(self) -> bool:
    return len(self._keys) > 1

  def get_branch_index(self, key: int) -> int:
    return min(np.searchsorted(self._keys, key), len(self._keys) - 1)
