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
"""Tokenizer for language models."""
from __future__ import annotations

import dataclasses
import os
from typing import Any, List, Tuple

from praxis import base_hyperparams
from saxml.server.pax.lm import vocabularies
import tensorflow as tf

StreamState = Tuple[tf.Tensor, tf.Tensor, tf.Tensor]


class LMTokenizer(base_hyperparams.FiddleBaseParameterizable):
  """Tokenizer for language models.

  Attributes:
    append_eos: Whether to append </s> at the end and treat it as a non-padded
      label, always set to True.
    prepend_sos: Whether to prepend sos at the beginning and treat it as a
      non-padded label, default is set to True.
    spm_model: File name for a sentencepiece model. This is used to construct a
      vocabularies.SentencePieceVocabulary object when vocabulary_class is None.
    target_sos_id: Start of sentence id.
    target_eos_id: End of sentence id.
    slice_left: If true, keep the left part of the sequence if it is too long.
      Otherwise, keep the right part of the sequence.
    streaming_whitespace_preserving_prefix: A prefix added to each non-SOS
      streaming decoding step to prevent the leading whitespace from being
      removed by sentencepiece; after decoding the step, it will be removed from
      the string result. It must be a regular token in the vocabulary.
    tokenized_input: Whether to skip the input tokenization. This is useful when
      the input are in tokens instead of texts
    tokenized_output: Output tokens instead of texts.
    eos_padding_and_no_sos: Do not use EOS or SOS for id or label, and use EOS
      for padding. This is a special tokenization specific to GPTJ MLPerf
      inference implementation. Only used when tokenized=True.
    vocabulary_class: The name of the vocabulary class. If None, a
      vocabularies.SentencePieceVocabulary object is constructed from spm_model;
      otherwise, vocabulary_path also needs to be specified.
    vocabulary_path: The path to the directory or file, if a vocabulary_class is
      specified.
  """

  prepend_sos: bool = True
  append_eos: bool = True
  spm_model: str = None
  target_sos_id: int = 0
  target_eos_id: int = 1
  slice_left: bool = True
  streaming_whitespace_preserving_prefix: str = 'a'
  tokenized_input: bool = False
  tokenized_output: bool = False
  eos_padding_and_no_sos: bool = False
  vocabulary_class: str = None
  vocabulary_path: str = None

  _vocab: vocabularies.Vocabulary = dataclasses.field(init=False, repr=False)

  def __post_init__(self):
    assert self.append_eos
    if self.vocabulary_class is None:
      assert self.spm_model is not None
      self._vocab = vocabularies.SentencePieceVocabulary(
          self.hparams.spm_model, 0
      )
    else:
      assert self.vocabulary_path is not None
      assert self.spm_model is None
      vocab_cls = getattr(vocabularies, self.vocabulary_class)
      vocabulary_path = self.vocabulary_path
      if vocabulary_path.startswith('gs://'):
        # Need to copy the vocabulary_path if in GCS to local, since HuggingFace
        # does not support loading tokenizer from GCS.
        local_vocabulary_path = os.path.join(
            '/tmp/vocabulary/' + vocabulary_path.split('/')[-1]
        )

        if not os.path.exists(local_vocabulary_path):
          os.makedirs(local_vocabulary_path)

        for filename in tf.io.gfile.listdir(vocabulary_path):
          src = os.path.join(vocabulary_path, filename)
          dst = os.path.join(local_vocabulary_path, filename)
          if not os.path.exists(dst):
            tf.io.gfile.copy(src, dst, overwrite=False)

        vocabulary_path = local_vocabulary_path

      self._vocab = vocab_cls(vocabulary_path)

  @property
  def Vocabulary(self) -> vocabularies.Vocabulary:
    """Get the vocabulary."""
    return self._vocab

  def StringsToIdsTokenized(
      self, strs: tf.Tensor, max_length: int
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    p = self.hparams
    batch = tf.shape(strs)[0]

    # `strs` may contain empty string elements. This happens either when
    # the user explicitly sends them to indicate empty inputs, or when
    # dummy inputs defined with empty strings are given to the model.
    # tf.strings.split would generate [b''] for empty string elements,
    # which tf.strings.to_number cannot handle. Convert [b''] to [] to be
    # consistent with the not p.tokenized path.
    labels_in_str = tf.strings.split(strs, sep=',', maxsplit=-1)
    empty_str_tensor = tf.constant([], dtype=tf.string)
    labels_in_str = tf.map_fn(
        lambda x: empty_str_tensor if x.shape == [1] and x[0] == b'' else x,  # pylint: disable=g-explicit-bool-comparison
        labels_in_str,
    )
    labels = tf.strings.to_number(labels_in_str, out_type=tf.int32)
    if p.slice_left:
      labels = labels[:, :max_length]
    else:
      labels = labels[:, -(max_length):]
    # Get the shape of each ragged tensor and drop the dimension of the shape.
    padding_indices = max_length - (
        tf.map_fn(tf.squeeze, tf.map_fn(tf.shape, labels).to_tensor())
    )
    # Convert to tensor and pad at the same time.
    lengths = labels.row_lengths()
    to_pad_as_flat_tensor = tf.repeat(
        self.target_eos_id, repeats=tf.reduce_sum(max_length - lengths)
    )
    to_pad = tf.RaggedTensor.from_row_lengths(
        to_pad_as_flat_tensor, max_length - lengths
    )
    labels = tf.concat([labels, to_pad], axis=1).to_tensor()

    padding_indices = tf.stack([padding_indices] * max_length, axis=1)
    indices = tf.repeat([tf.range(max_length)], batch, axis=0)

    paddings = tf.where(
        indices >= padding_indices,
        tf.zeros_like(indices, tf.float32),
        tf.ones_like(indices, tf.float32),
    )
    return labels, labels, tf.reverse(paddings, axis=[1])

  def StringsToIds(
      self, strs: tf.Tensor, max_length: int
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Tokenizes strs into vocab ids.

    Args:
      strs: A 1D tensor of strings.
      max_length: An int providing the max_length for strs.
      unused_args: Some not used arguments from base class.

    Returns:
      A tuple (ids, labels, paddings) with the same shape [batch, maxlen].

      - ids[i, j] is the input token id of i-th sample for j-th step.
      - labels[i, j] is the target token id of i-th sample for j-th step.
      - paddings[i, j] is 1 iff i-th sample's j-th step is padded.
    """
    p = self.hparams
    assert max_length is not None

    if p.tokenized_input and p.eos_padding_and_no_sos:
      return self.StringsToIdsTokenized(strs, max_length)

    batch = tf.shape(strs)[0]
    # labels is a ragged Tensor.
    if p.tokenized_input:
      # `strs` may contain empty string elements. This happens either when
      # the user explicitly sends them to indicate empty inputs, or when
      # dummy inputs defined with empty strings are given to the model.
      # tf.strings.split would generate [b''] for empty string elements,
      # which tf.strings.to_number cannot handle. Convert [b''] to [] to be
      # consistent with the not p.tokenized path.
      labels_in_str = tf.strings.split(strs, sep=',', maxsplit=-1)
      empty_str_tensor = tf.constant([], dtype=tf.string)
      labels_in_str = tf.map_fn(
          lambda x: empty_str_tensor if x.shape == [1] and x[0] == b'' else x,  # pylint: disable=g-explicit-bool-comparison
          labels_in_str,
      )
      labels = tf.strings.to_number(labels_in_str, out_type=tf.int32)
    else:
      labels = self._vocab.encode_tf(strs)
    if p.slice_left:
      labels = labels[:, : max_length - 1]
    else:
      labels = labels[:, -(max_length - 1) :]

    if p.prepend_sos:
      sos_ids = tf.fill(
          [batch, 1], tf.constant(p.target_sos_id, dtype=tf.int32)
      )
      ids = tf.concat([sos_ids, labels], axis=1)
    else:
      ids = tf.identity(labels)
    eos_ids = tf.fill([batch, 1], tf.constant(p.target_eos_id, dtype=tf.int32))
    labels = tf.concat([labels, eos_ids], axis=1)
    # Convert raggedtensor to padded tensor.
    ids = ids.to_tensor()
    labels = labels.to_tensor()

    def _pad(x: tf.Tensor, shape: List[int]) -> tf.Tensor:
      """Helper function to pad tensor to the desired shape."""
      pad = shape - tf.minimum(tf.shape(x), shape)
      zeros = tf.zeros_like(pad)
      # If dim_i is less than shape[i], pads after contents.
      paddings = tf.stack([zeros, pad], axis=1)
      # If dim_i is larger than shape[i], we slice [0:shape[i]] for dim_i.
      slice_begin = zeros
      x = tf.pad(x, paddings)
      x = tf.slice(x, slice_begin, shape)

      return tf.reshape(x, shape)

    # Pad ids and labels to the desired shape.
    shape = [batch, max_length]
    ids = _pad(ids, shape)
    labels = _pad(labels, shape)

    # Calculate paddings for each example based on eos_id locations.
    eos_indices = tf.argmax(
        tf.equal(labels, p.target_eos_id), axis=1, output_type=tf.int32
    )
    eos_indices = tf.stack([eos_indices] * max_length, axis=1)
    indices = tf.repeat([tf.range(max_length)], batch, axis=0)
    if p.prepend_sos:
      paddings = tf.where(
          indices <= eos_indices,
          tf.zeros_like(indices, tf.float32),
          tf.ones_like(indices, tf.float32),
      )
    else:
      paddings = tf.where(
          indices < eos_indices,
          tf.zeros_like(indices, tf.float32),
          tf.ones_like(indices, tf.float32),
      )

    return ids, labels, paddings

  def IdsToStrings(self, ids: tf.Tensor, *unused_args: Any) -> tf.Tensor:
    """Converts ids back to strings.

    Decoding stops at padding or eos token.

    Args:
      ids: A matrix of shape [batch, seqlen]. ids[i, :] is the i-th sample's
        ids.
      unused_args: Some not used arguments for API use.

    Returns:
      sequences - A vector of shape [batch]. The converted string sequence.
    """
    p = self.hparams

    if p.tokenized_output:
      ids_as_string = tf.strings.as_string(ids)
      reduced_ids_as_string = tf.strings.reduce_join(
          ids_as_string, separator=',', axis=-1
      )
      return reduced_ids_as_string

    return self._vocab.decode_tf(ids)

  def IdToString(self, ids: tf.Tensor) -> tf.Tensor:
    """Converts each token ID to a token string.

    Args:
      ids: A tensor of shape [batch, seqlen] and int32 data type.
        ids[n, i] is the token ID at decoding step i for the n-th sample.

    Returns:
      A tensor of token strings with the same shape as the input ids.
    """
    return self._vocab.id_to_string_tf(ids)

  def InitStream(self, batch_size: tf.Tensor) -> StreamState:
    """Create the initial state for streaming.

    Args:
      batch_size: A scalar or 1D tensor representing stream formation, usually
        batch or [batch, num_samples].

    Returns:

      A tuple of 3 elements:
        - A tensor of shape [np.prod(batch_size), seqlen], containing
          unprocessed prefix IDs. Left-aligned if they have differen lengths.
        - A tensor of shape [np.prod(batch_size)], indicating the valid length
          in the unprocessed prefix IDs.
        - A boolean tensor of shape batch_size indicating if any prefix strings
          have been generated.
    """
    nrows = tf.math.reduce_prod(batch_size)
    started = tf.fill(batch_size, False)
    return (
        tf.zeros([nrows, 0], dtype=tf.int32),
        tf.zeros([nrows], dtype=tf.int32),
        started,
    )

  def DecodeOnStream(
      self, new_ids: tf.Tensor, stream_state: StreamState
  ) -> Tuple[tf.Tensor, StreamState]:
    """Converts new chunks of IDs on decoding streams.

    Args:
      new_ids: A matrix of shape [batch, ..., new_chunk_len] containing IDs
        newly generated from streaming.
      stream_state: Stream state. See description in InitStream.

    Returns:
      A tuple of (newly decoded strings, updated stream state).
    """
    p = self.hparams
    if p.tokenized_input or self.vocabulary_class is not None:
      raise NotImplementedError(
          'DecodeOnStream does not support when TOKENIZE_INPUT=True, nor when'
          'VOCABULARY_CLASS is specified other than SentencePieceVocabulary.'
      )
    assert p.spm_model

    # Merge all leading N - 1 dimensions of new_ids and started to match the
    # flattened ragged tensor.
    unprocessed_prefix_ids, unprocessed_prefix_len, started = stream_state
    batch_size = tf.shape(started)
    nrows = tf.math.reduce_prod(batch_size)
    new_ids = tf.reshape(new_ids, [nrows, -1])
    started = tf.reshape(started, [nrows])

    # Hint the rank of prefix ids/len for `tf.RaggedTensor.from_tensor`, which
    # is useful when TF can't identify their ranks on its own.
    unprocessed_prefix_ids = tf.ensure_shape(
        unprocessed_prefix_ids, [None, None]
    )
    unprocessed_prefix_len = tf.ensure_shape(unprocessed_prefix_len, [None])

    # Find byte-encoded IDs, represented as "<0x??>", where ? is [0-9,A-F].
    new_ids_shape = tf.shape(new_ids)
    b, new_seqlen = new_ids_shape[0], new_ids_shape[1]
    new_pieces = self._vocab.id_to_string_tf(new_ids)
    # Extend every string tensor element length by 6.
    spaces = tf.broadcast_to(tf.constant(' ' * 6), new_ids_shape)
    new_pieces = tf.strings.join([new_pieces, spaces])
    # Byte-encoded IDs should have length 6 + 6 = 12.
    new_pieces_len = tf.equal(tf.strings.length(new_pieces), 12)
    # Byte-encoded IDs should start with "<0x".
    start = tf.constant('<0x')
    new_pieces_start = tf.equal(tf.strings.substr(new_pieces, 0, 3), start)
    # Byte-encoded IDs should end with ">".
    end = tf.constant('>')
    new_pieces_end = tf.equal(tf.strings.substr(new_pieces, 5, 1), end)
    # Whether every ID is a byte-encoded ID, not testing two middle characters.
    is_byte = new_pieces_len & new_pieces_start & new_pieces_end
    # Remove trailing bytes.
    trailing_byte_count = tf.reduce_sum(
        tf.cast(
            tf.equal(
                tf.cumsum(1 - tf.cast(is_byte, tf.int32), axis=1, reverse=True),
                0,
            ),
            tf.int32,
        ),
        axis=1,
    )
    without_trailing_bytes = tf.RaggedTensor.from_tensor(
        new_ids, new_seqlen - trailing_byte_count
    )
    is_all_bytes = tf.equal(trailing_byte_count, new_seqlen)

    # Add a fake prefix to preserve leading whitespace if earlier prefix was
    # generated.
    fake_prefix_str = p.streaming_whitespace_preserving_prefix
    fake_prefix = self._vocab.encode_tf([fake_prefix_str]).to_tensor()
    fake_prefix = tf.repeat(fake_prefix, b, axis=0)
    fake_prefix_len = tf.fill([b], tf.shape(fake_prefix)[1])
    fake_prefix_str_len = tf.fill(
        [b], tf.constant(len(fake_prefix_str), dtype=tf.int32)
    )
    fake_prefix_len = tf.where(started, fake_prefix_len, 0)
    fake_prefix_str_len = tf.where(started, fake_prefix_str_len, 0)
    fake_prefix = tf.RaggedTensor.from_tensor(fake_prefix, fake_prefix_len)

    # Decode with prefix.
    unprocessed_prefix_ids_ragged = tf.RaggedTensor.from_tensor(
        unprocessed_prefix_ids, unprocessed_prefix_len
    )
    to_process = tf.concat(
        [fake_prefix, unprocessed_prefix_ids_ragged, without_trailing_bytes],
        axis=1,
    )
    new_strs = self._vocab.decode_tf(to_process)
    # Remove fake prefix.
    new_strs = tf.strings.substr(
        new_strs, fake_prefix_str_len, tf.fill([b], -1)
    )
    new_strs = tf.where(is_all_bytes, tf.constant(''), new_strs)

    new_started = tf.logical_or(started, tf.strings.length(new_strs) > 0)

    trailing_bytes = tf.RaggedTensor.from_tensor(
        tf.reverse(new_ids, axis=[1]), trailing_byte_count
    )
    trailing_bytes = tf.reverse(trailing_bytes, axis=[1])

    remaining_prefix_len = tf.where(is_all_bytes, unprocessed_prefix_len, 0)
    remaining_prefix = tf.RaggedTensor.from_tensor(
        unprocessed_prefix_ids, remaining_prefix_len
    )
    new_unprocessed_prefix_ids = tf.concat(
        [remaining_prefix, trailing_bytes], axis=1
    )

    # Reshape output back to [batch_size, ...] before returning.
    new_strs = tf.reshape(new_strs, batch_size)
    new_started = tf.reshape(new_started, batch_size)
    return new_strs, (
        new_unprocessed_prefix_ids.to_tensor(),
        tf.cast(new_unprocessed_prefix_ids.row_lengths(), dtype=tf.int32),
        new_started,
    )

  def FinishStream(self, stream_state: StreamState) -> tf.Tensor:
    """Finishes the streams by decoding any remaining tokens."""
    p = self.hparams
    assert p.spm_model
    _, _, started = stream_state
    batch_size = tf.shape(started)

    eos_ids = tf.fill(batch_size, tf.constant(p.target_eos_id, dtype=tf.int32))
    eos_ids = tf.expand_dims(eos_ids, axis=-1)
    new_strs, _ = self.DecodeOnStream(eos_ids, stream_state)
    return new_strs
