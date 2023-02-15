# Copyright 2023 Google LLC
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

"""Tests for servable_lm_common."""

import os

from absl import flags
from absl.testing import absltest
from praxis import test_utils
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_common

FLAGS = flags.FLAGS


def create_tokenizer_params():
  p = lm_tokenizer.LMTokenizer.HParams(
      spm_model=os.path.join(
          FLAGS.test_srcdir,
          'google3/learning/multipod/sax/server/lm/test_data',
          'meena_0611.32000.model',
      ),
      slice_left=False,
      target_sos_id=0,
      target_eos_id=1,
  )
  return p


class ServableLmCommonTest(test_utils.TestCase):

  def test_score_tf_tokenize_inputs(self):
    p = create_tokenizer_params()
    tokenizer = p.Instantiate()
    prefixes = ['ABCDEFGHIJKLMN', 'Hi']
    suffixes = ['XX!!', 'YY!!']
    result = servable_lm_common.score_tf_tokenize_inputs(
        prefixes,
        suffixes,
        tokenizer,
        max_prefix_seq_len=5,
        max_suffix_seq_len=5,
        include_eos=True,
    )
    ids = result[0].numpy().tolist()
    labels = result[0].numpy().tolist()
    expected_strings = ['JKLMN XX!!', 'Hi YY!!']
    self.assertArraysEqual(
        expected_strings,
        [x.numpy().decode() for x in tokenizer.IdsToStrings(ids)],
    )
    self.assertArraysEqual(
        expected_strings,
        [x.numpy().decode() for x in tokenizer.IdsToStrings(labels)],
    )


if __name__ == '__main__':
  absltest.main()
