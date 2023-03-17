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
"""Tests for servable_lm_common."""

import os

from absl import flags
import numpy as np
from praxis import py_utils
from praxis import test_utils
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_common
import tensorflow as tf

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


class ServableLmCommonTest(tf.test.TestCase, test_utils.TestCase):

  def setUp(self):
    super().setUp()
    self.tokenizer = create_tokenizer_params().Instantiate()

  def test_score_tf_tokenize_inputs(self):
    prefixes = ['ABCDEFGHIJKLMN', 'Hi']
    suffixes = ['XX!!', 'YY!!']
    result = servable_lm_common.score_tf_tokenize_inputs(
        prefixes,
        suffixes,
        self.tokenizer,
        max_prefix_seq_len=5,
        max_suffix_seq_len=5,
        include_eos=True,
    )
    ids = result[0].numpy().tolist()
    labels = result[0].numpy().tolist()
    expected_strings = ['JKLMN XX!!', 'Hi YY!!']
    self.assertArraysEqual(
        expected_strings,
        [x.numpy().decode() for x in self.tokenizer.IdsToStrings(ids)],
    )
    self.assertArraysEqual(
        expected_strings,
        [x.numpy().decode() for x in self.tokenizer.IdsToStrings(labels)],
    )

  def test_decode_post_processing_decoder_only(self):
    max_length = 8
    strs = ['hello world', 'the quick brown fox jumps']
    ids, _, paddings = self.tokenizer.StringsToIds(strs, max_length)
    computed_outputs = py_utils.NestedMap(
        # First sequence has last dim padded.
        output_ids=tf.expand_dims(ids, 0).numpy(),
        # Non-paddings as decoded length for each tensor.
        decode_lengths=tf.expand_dims(
            tf.math.reduce_sum(
                tf.cast(tf.equal(paddings, 0), tf.int32), axis=-1
            ),
            0,
        ).numpy(),
        scores=np.asarray([[[0.1], [0.2]]]),
    )

    out = servable_lm_common.decode_tf_post_processing(
        computed_outputs, self.tokenizer, include_prefix_in_result=True
    )

    self.assertContainsExactSubsequence(
        [s.numpy().decode() for s in tf.squeeze(out['topk_decoded'])], strs
    )
    self.assertArraysEqual(
        np.asarray([3, 6]), tf.squeeze(out['topk_decode_lengths']).numpy()
    )

  def test_decode_post_processing_encoder_decoder(self):
    max_length = 8
    strs = ['hello world', 'the quick brown fox jumps']
    ids, _, _ = self.tokenizer.StringsToIds(strs, max_length)
    computed_outputs = py_utils.NestedMap(
        # First sequence has last dim padded.
        output_ids=ids.numpy(),
        # None as the convention from decode_fetch_output.
        decode_lengths=None,
        scores=np.asarray([0.1, 0.2]),
    )

    out = servable_lm_common.decode_tf_post_processing(
        computed_outputs, self.tokenizer, encoder_decoder_model=True
    )

    self.assertContainsExactSubsequence(
        [s.numpy().decode() for s in tf.squeeze(out['topk_decoded'])], strs
    )
    self.assertAllEqual([0, 0], tf.squeeze(out['topk_decode_lengths']))


if __name__ == '__main__':
  tf.test.main()
