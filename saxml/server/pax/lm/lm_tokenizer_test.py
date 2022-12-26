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

"""Tests for lm_tokenizer."""

import os

from absl import flags
from absl.testing import parameterized
from praxis import base_layer
from saxml.server.pax.lm import lm_tokenizer
import tensorflow as tf

instantiate = base_layer.instantiate
FLAGS = flags.FLAGS


def _CreateParams():
  p = lm_tokenizer.LMTokenizer.HParams(
      spm_model=os.path.join(
          FLAGS.test_srcdir,
          'google3/learning/multipod/sax/server/lm/test_data',
          'meena_0611.32000.model'),
      target_sos_id=0,
      target_eos_id=1)
  return p


class LMTokenizerTest(tf.test.TestCase, parameterized.TestCase):

  def testStringsToIds(self):
    p = _CreateParams()
    tokenizer = p.Instantiate()
    max_length = 5
    strs = ['hello world', 'the quick brown fox jumps']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[0, 9873, 640, 0, 0], [0, 261, 1242, 3350, 9806]], ids)
    self.assertAllEqual(
        [[9873, 640, 1, 0, 0], [261, 1242, 3350, 9806, 1]], labels)
    self.assertAllEqual([[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                        paddings)

  def testStringsToIdsLongMaxLen(self):
    p = _CreateParams()
    tokenizer = instantiate(p)
    max_length = 8
    strs = ['hello world', 'the quick brown fox jumps']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[0, 9873, 640, 0, 0, 0, 0, 0],
                         [0, 261, 1242, 3350, 9806, 11144, 0, 0]], ids)
    self.assertAllEqual([[9873, 640, 1, 0, 0, 0, 0, 0],
                         [261, 1242, 3350, 9806, 11144, 1, 0, 0]], labels)
    self.assertAllEqual([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]], paddings)

  def testStringsToIdsShortMaxLenSliceLeft(self):
    p = _CreateParams()
    tokenizer = instantiate(p)
    max_length = 4
    strs = ['hello world', 'the quick brown fox jumps']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[0, 9873, 640, 0], [0, 261, 1242, 3350]], ids)
    self.assertAllEqual([[9873, 640, 1, 0], [261, 1242, 3350, 1]], labels)
    self.assertAllEqual([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], paddings)

  def testStringsToIdsShortMaxLenSliceRight(self):
    p = _CreateParams()
    p.slice_left = False
    tokenizer = instantiate(p)
    max_length = 4
    strs = ['hello world', 'the quick brown fox jumps']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)
    self.assertAllEqual([[0, 9873, 640, 0], [0, 3350, 9806, 11144]], ids)
    self.assertAllEqual([[9873, 640, 1, 0], [3350, 9806, 11144, 1]], labels)
    self.assertAllEqual([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], paddings)

  def testIdsToStrings(self):
    p = _CreateParams()
    tokenizer = instantiate(p)
    ids = [[9873, 640, 1, 0, 0, 0, 0, 0, 0, 0],
           [261, 1242, 3350, 9806, 11144, 1, 0, 0, 0, 0]]
    strs = tokenizer.IdsToStrings(ids)
    self.assertEqual([b'hello world', b'the quick brown fox jumps'], list(strs))

  def testStreamWhiteSpaces(self):
    p = _CreateParams()
    tokenizer = instantiate(p)
    state = tokenizer.InitStream(2)

    strs, state = tokenizer.DecodeOnStream(state, [[0], [261]])
    self.assertEqual([b'', b'the'], list(strs))
    strs, state = tokenizer.DecodeOnStream(state, [[9873], [1242]])
    self.assertEqual([b'hello', b' quick'], list(strs))
    strs, state = tokenizer.DecodeOnStream(state,
                                           [[640, 1, 0], [3350, 9806, 11144]])
    self.assertEqual([b' world', b' brown fox jumps'], list(strs))
    strs, state = tokenizer.DecodeOnStream(state, [[0], [1]])
    self.assertEqual([b'', b''], list(strs))
    strs = tokenizer.FinishStream(state)
    self.assertEqual([b'', b''], list(strs))

  def testStreamBytes(self):
    p = _CreateParams()
    tokenizer = instantiate(p)
    byte_ids = list(
        tokenizer._vocab.tf_tokenizer.string_to_id(
            #   g           h           i
            [b'<0x67>', b'<0x68>', b'<0x69>']))
    state = tokenizer.InitStream(2)

    strs, state = tokenizer.DecodeOnStream(state, [[9873], [261]])
    self.assertEqual([b'hello', b'the'], list(strs))
    strs, state = tokenizer.DecodeOnStream(state,
                                           [[byte_ids[0]], [byte_ids[1]]])
    self.assertEqual([b'', b''], list(strs))
    strs, state = tokenizer.DecodeOnStream(state, [[byte_ids[2]], [1242]])
    self.assertEqual([b'', b'h quick'], list(strs))
    strs, state = tokenizer.DecodeOnStream(
        state, [[byte_ids[2], 640], [byte_ids[0], byte_ids[1]]])
    self.assertEqual([b'gii world', b''], list(strs))
    strs = tokenizer.FinishStream(state)
    self.assertEqual([b'', b'gh'], list(strs))


if __name__ == '__main__':
  tf.test.main()
