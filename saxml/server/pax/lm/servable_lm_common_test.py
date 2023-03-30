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
from saxml.server.jax import np_tf_sess_wrapper
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_common
import tensorflow as tf

FLAGS = flags.FLAGS


def create_tokenizer_params():
  p = lm_tokenizer.LMTokenizer.HParams(
      spm_model=os.path.join(
          FLAGS.test_srcdir,
          '__main__/saxml/server/pax/lm/test_data',
          'test_model.model',
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

  @test_utils.parameterized.parameters((True,), (False,))
  def test_decode_post_processing_decoder_only(self, use_wrapper: bool):
    max_length = 8
    strs = ['Hello world', 'This is a test']
    ids, _, paddings = self.tokenizer.StringsToIds(strs, max_length)
    # This SPM doesn't support padding, decoding 0 to <unk>, so trim the output.
    ids = tf.slice(ids, [0, 1], [-1, -1])
    paddings = tf.slice(paddings, [0, 1], [-1, -1])
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

    def decode_tf_post_processing(*args):
      return servable_lm_common.decode_tf_post_processing(
          *args,
          tokenizer=self.tokenizer,
          t5_model=False,
          include_prefix_in_result=True,
      )

    if use_wrapper:
      decode_tf_post_processing = np_tf_sess_wrapper.wrap_tf_session(
          decode_tf_post_processing
      )

    out = decode_tf_post_processing(computed_outputs)

    self.assertContainsExactSubsequence(
        [s.numpy().decode() for s in tf.squeeze(out['topk_decoded'])], strs
    )
    self.assertArraysEqual(
        np.asarray([4, 6]), tf.squeeze(out['topk_decode_lengths']).numpy()
    )


if __name__ == '__main__':
  tf.test.main()
