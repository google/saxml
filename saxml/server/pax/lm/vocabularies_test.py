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
"""Tests for vocabularies."""
import os
from absl import flags
from saxml.server.pax.lm import vocabularies
import tensorflow as tf

FLAGS = flags.FLAGS


def gpt2bpe_vocab():
  vocabulary_path = os.path.join(
      FLAGS.test_srcdir,
      "google3/third_party/py/saxml/server/pax/lm/test_data",
      "gpt2bpe",
  )

  return vocabularies.GPT2BPEVocabulary(vocabulary_path)


class GPT2BPEVocabularyTest(tf.test.TestCase):
  TEST_BATCH_STRINGS = ["Hello", "world"]
  TEST_BATCH_TOKENS = [[0], [1]]

  TEST_BATCH_EMPTY_STRINGS = ["", ""]
  TEST_BATCH_EMPTY_TOKENS = [[], []]

  def test_vocab(self):
    vocab = gpt2bpe_vocab()
    # The test vocab.json is a modified version of the vocab.json.
    # The HuggingFace Tokenizer vocab_size does not include the `added_tokens`
    # and is equivalent to Seqio Vocabulary _base_vocab_size.
    self.assertEqual(vocab.tokenizer.vocab_size, vocab._base_vocab_size)
    self.assertEqual(vocab.tokenizer.vocab_size, 3)
    # The HF number of added_tokens are equivalent to Seqio extra_ids.
    # Seqio vocab_size includes the _base_vocab_size and extra_ids.
    self.assertEqual(vocab.vocab_size, 6)

    self.assertEqual(vocab.pad_id, 5)
    self.assertEqual(vocab.bos_id, 2)
    self.assertEqual(vocab.eos_id, 2)
    self.assertEqual(vocab.unk_id, 2)

  def test_encode_tf(self):
    vocab = gpt2bpe_vocab()
    actual_batch_encode_tf = vocab.encode_tf(
        tf.constant(self.TEST_BATCH_STRINGS)
    )
    self.assertEqual(actual_batch_encode_tf.shape[0], 2)

  def test_decode_tf(self):
    vocab = gpt2bpe_vocab()
    ids = tf.ragged.constant(self.TEST_BATCH_TOKENS, dtype=tf.int32)
    actual_batch_decode_tf = vocab.decode_tf(ids)
    self.assertEqual(actual_batch_decode_tf.shape[0], 2)

  def test_encode_tf_empty_strings(self):
    vocab = gpt2bpe_vocab()
    actual_batch_encode_tf = vocab.encode_tf(
        tf.constant(self.TEST_BATCH_EMPTY_STRINGS)
    )
    expected_batch_encode_tf = tf.ragged.constant(
        self.TEST_BATCH_EMPTY_TOKENS, dtype=tf.int32
    )
    self.assertEqual(
        actual_batch_encode_tf.shape, expected_batch_encode_tf.shape
    )
    self.assertTrue(
        tf.math.reduce_all(
            tf.equal(actual_batch_encode_tf, expected_batch_encode_tf)
        )
    )

  def test_equal(self):
    vocab1 = gpt2bpe_vocab()
    vocab2 = gpt2bpe_vocab()
    self.assertEqual(vocab1, vocab2)


if __name__ == "__main__":
  tf.test.main()
