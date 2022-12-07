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

from absl.testing import parameterized
from lingvo import compat as tf
from lingvo.core import test_utils
from praxis import base_layer

from saxml.server.pax.lm import lm_tokenizer

instantiate = base_layer.instantiate
FLAGS = tf.flags.FLAGS


def _CreateParams():
  p = lm_tokenizer.LMTokenizer.HParams(
      spm_model=os.path.join(
          FLAGS.test_srcdir,
          'google3/learning/multipod/sax/server/lm/test_data',
          'meena_0611.32000.model'),
      target_sos_id=0,
      target_eos_id=1)
  return p


class LMTokenizerTest(test_utils.TestCase, parameterized.TestCase):

  def testStringsToIds(self):
    p = _CreateParams()
    tokenizer = p.Instantiate()
    max_length = 5
    strs = ['hello world', 'the quick brown fox jumps']
    ids, labels, paddings = tokenizer.StringsToIds(strs, max_length)

    with self.session() as sess:
      ids, labels, paddings = sess.run([ids, labels, paddings])
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

    with self.session() as sess:
      ids, labels, paddings = sess.run([ids, labels, paddings])
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

    with self.session() as sess:
      ids, labels, paddings = sess.run([ids, labels, paddings])
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

    with self.session() as sess:
      ids, labels, paddings = sess.run([ids, labels, paddings])
    self.assertAllEqual([[0, 9873, 640, 0], [0, 3350, 9806, 11144]], ids)
    self.assertAllEqual([[9873, 640, 1, 0], [3350, 9806, 11144, 1]], labels)
    self.assertAllEqual([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]], paddings)

  def testIdsToStrings(self):
    p = _CreateParams()
    tokenizer = instantiate(p)
    ids = [[9873, 640, 1, 0, 0, 0, 0, 0, 0, 0],
           [261, 1242, 3350, 9806, 11144, 1, 0, 0, 0, 0]]
    strs = tokenizer.IdsToStrings(ids)
    with self.session() as sess:
      strs = sess.run(strs)

    self.assertEqual([b'hello world', b'the quick brown fox jumps'], list(strs))


if __name__ == '__main__':
  tf.test.main()
