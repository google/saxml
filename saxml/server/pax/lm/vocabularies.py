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
"""Custom tokenizer for language models."""

import json
import os
import pathlib
from typing import Dict, Iterator, List, Optional, Sequence, cast

from absl import logging
import seqio
import tensorflow as tf


class Vocabulary(seqio.Vocabulary):
  """Abstract class for all vocabularies."""

  def decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Detokenizes int32 batched Tensor."""
    return self._decode_tf(ids)

  def id_to_string_tf(self, ids: tf.Tensor) -> tf.Tensor:
    raise NotImplementedError


class SentencePieceVocabulary(Vocabulary, seqio.SentencePieceVocabulary):
  """Wrapper for nlp/sentencepiece encoder."""

  def id_to_string_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Converts vocabulary id into a token.

    Args:
      ids: An arbitrary tensor of int32 representing the token IDs.

    Returns:
      A tensor of string with the same shape as input.
    """
    return self.tf_tokenizer.id_to_string(ids)


class GPT2BPEVocabulary(Vocabulary):
  """The HuggingFace GPT2BPEVocabulary Class.

  The class wraps the HuggingFace `tokenizers` Tokenizer.

  Special tokens (bos, eos, unk, pad) are read from `tokenizer_config.json`
  in the same directory as `tokenizer.json`.
  """

  def __init__(self, tokenizer_name_or_path: str):
    """Vocabulary constructor.

    Args:
      tokenizer_name_or_path: Can be either: a path to a directory containing
        a `tokenizer.json`, or a model id hosted on huggingface.co (OSS only).
    """
    super().__init__()

    # pylint: disable=g-import-not-at-top
    from tokenizers import Tokenizer

    if os.path.isdir(tokenizer_name_or_path):
      tokenizer_path = os.path.join(tokenizer_name_or_path, "tokenizer.json")
      self._tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
      self._tokenizer = Tokenizer.from_pretrained(tokenizer_name_or_path)
      tokenizer_path = None
    self._vocab = self._tokenizer.get_vocab()

    # Read special token names from tokenizer_config.json.
    config_path = os.path.join(tokenizer_name_or_path, "tokenizer_config.json")
    config = {}
    if os.path.isfile(config_path):
      with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    def _get_special(key: str) -> Optional[str]:
      val = config.get(key)
      if val is None:
        return None
      if isinstance(val, str):
        return val
      if isinstance(val, dict):
        return val.get("content")
      return None

    self._bos_token = _get_special("bos_token") or "<|endoftext|>"
    self._eos_token = _get_special("eos_token") or "<|endoftext|>"
    self._unk_token = _get_special("unk_token") or "<|endoftext|>"
    self._pad_token = _get_special("pad_token")

    def _token_id(
        token: Optional[str],
    ) -> Optional[int]:
      if token is None:
        return None
      return self._tokenizer.token_to_id(token)

    self._bos_token_id = _token_id(self._bos_token)
    self._eos_token_id = _token_id(self._eos_token)
    self._unk_token_id = _token_id(self._unk_token)
    self._pad_token_id = _token_id(self._pad_token)

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including _base_vocab_size and extra_ids."""
    return len(self._vocab)

  def _encode(self, s: str) -> Sequence[int]:
    """Encode a python string as a list of integers.

    Args:
      s: a string

    Returns:
      a list of integers
    """
    return self._tokenizer.encode(s).ids

  def _decode(self, ids: Sequence[int]) -> str:
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers

    Returns:
      a string
    """
    return self._tokenizer.decode(list(ids), skip_special_tokens=True)

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    """Encode a string tf.Tensor.

    Args:
      s: a tf.Scalar or a 1d tf.Tensor in [batch_size] with dtype tf.string

    Returns:
      a 1d tf.Tensor or
      a tf.RaggedTensor in [batch_size, None] with dtype tf.int32
    """

    def _encode_py_func(text) -> Sequence[int]:
      text = tf.compat.as_text(text.numpy().decode("UTF-8"))
      result = self._tokenizer.encode(text).ids
      result = tf.expand_dims(result, 0)
      return tf.cast(result, tf.int32)

    def _encode_func():
      result = tf.py_function(func=_encode_py_func, inp=[s], Tout=tf.int32)
      result = tf.squeeze(result, 0)
      result.set_shape([None])
      return result

    def _batch_encode_py_func(strs) -> Sequence[Sequence[int]]:
      texts = []
      for s in strs.numpy():
        texts.append(tf.compat.as_text(s.decode("UTF-8")))
      encodings = self._tokenizer.encode_batch(texts)
      input_ids = [enc.ids for enc in encodings]
      return tf.ragged.constant(input_ids, dtype=tf.int32)

    def _batch_encode_func():
      return tf.py_function(
          func=_batch_encode_py_func,
          inp=[s],
          Tout=tf.RaggedTensorSpec(shape=[s.shape[0], None], dtype=tf.int32),
      )

    # pylint: disable=g-explicit-length-test
    if len(s.shape) > 0:
      return _batch_encode_func()
    return _encode_func()

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Decode in TensorFlow.

    Args:
      ids: a 1d or 2d (in [batch_size, None]) tf.Tensor with dtype tf.int32

    Returns:
      a tf.Scaler or 1d (in [batch_size]) tf.Tensor with dtype tf.string
    """

    def _decode_py_func(token_ids: Sequence[int]) -> str:
      result = self._tokenizer.decode(list(token_ids), skip_special_tokens=True)
      return result

    def _decode_func():
      result = tf.py_function(func=_decode_py_func, inp=[ids], Tout=tf.string)
      return result

    def _batch_decode_py_func(token_ids) -> str:
      result = self._tokenizer.decode_batch(
          [list(seq) for seq in token_ids.numpy()], skip_special_tokens=True
      )
      result = tf.expand_dims(result, 0)
      return result

    def _batch_decode_func():
      result = tf.py_function(
          func=_batch_decode_py_func, inp=[ids], Tout=tf.string
      )
      result = tf.squeeze(result, 0)
      result.set_shape([None])
      return result

    if len(ids.shape) > 1:
      return _batch_decode_func()
    return _decode_func()

  @property
  def tokenizer(self):
    """Returns the HuggingFace tokenizer object."""
    return self._tokenizer

  @property
  def tf_tokenizer(self):
    raise NotImplementedError(
        "This is a HuggingFace tokenizer, it does not have a tf_tokenizer."
    )

  @property
  def vocab(self):
    """Returns the vocab dictionary."""
    return self._vocab

  @property
  def unk(self) -> str:
    return self._unk_token

  @property
  def bos(self) -> str:
    return self._bos_token

  @property
  def eos(self) -> str:
    return self._eos_token

  @property
  def pad(self) -> Optional[str]:
    return self._pad_token

  @property
  def unk_id(self) -> Optional[int]:
    return self._unk_token_id

  @property
  def bos_id(self) -> Optional[int]:
    return self._bos_token_id

  @property
  def eos_id(self) -> Optional[int]:
    return self._eos_token_id

  @property
  def pad_id(self) -> Optional[int]:
    return self._pad_token_id

  @property
  def _base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including BOS/EOS/UNK/PAD."""
    return self._tokenizer.get_vocab_size(with_added_tokens=False)

  def __eq__(self, other):
    if not isinstance(other, GPT2BPEVocabulary):
      return False
    return (
        self.extra_ids == other.extra_ids
        and self.unk_id == other.unk_id
        and self.bos_id == other.bos_id
        and self.eos_id == other.eos_id
        and self.pad_id == other.pad_id
        and self.vocab_size == other.vocab_size
        and self.vocab == other.vocab
    )


class LLama3Vocabulary(Vocabulary):
  """The LLama3 Tiktoken Vocabulary Class.

  Adapted from https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py
  """

  special_tokens: Dict[str, int]

  num_reserved_special_tokens = 256

  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # pylint: disable=line-too-long

  def __init__(self, tokenizer_path: str):
    """Vocabulary constructor.

    Args:
      tokenizer_path: The path to the Tiktoken model file.
    """
    super(LLama3Vocabulary, self).__init__()

    # pylint: disable=g-import-not-at-top
    import tiktoken
    from tiktoken.load import load_tiktoken_bpe  # pylint: disable=g-importing-member

    mergeable_ranks = load_tiktoken_bpe(tokenizer_path)
    num_base_tokens = len(mergeable_ranks)
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|reserved_special_token_2|>",
        "<|reserved_special_token_3|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|reserved_special_token_4|>",
        "<|eot_id|>",  # end of turn
    ] + [
        f"<|reserved_special_token_{i}|>"
        for i in range(5, self.num_reserved_special_tokens - 5)
    ]
    self.special_tokens = {
        token: num_base_tokens + i for i, token in enumerate(special_tokens)
    }
    # TODO(sax-dev): replace with native tiktoken Encoding once it is supported
    # in google3.
    self.model = tiktoken.Encoding(
        name=pathlib.Path(tokenizer_path).name,
        pat_str=self.pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens=self.special_tokens,
    )
    logging.info("Reloaded tiktoken model from %s", tokenizer_path)

    self._n_words: int = self.model.n_vocab
    # BOS / EOS token IDs

    self._bos_token_id: int = self.special_tokens["<|begin_of_text|>"]
    self._eos_token_id: int = self.special_tokens["<|end_of_text|>"]
    self._pad_token_id: int = -1
    self._stop_tokens = {
        self.special_tokens["<|end_of_text|>"],
        self.special_tokens["<|eot_id|>"],
    }
    logging.info(
        "#words: %d - BOS ID: %d - EOS ID: %d",
        self._n_words,
        self._bos_token_id,
        self._eos_token_id,
    )

  @property
  def _base_vocab_size(self) -> int:
    """Vocabulary size, excluding extra ids but including BOS/EOS/UNK/PAD."""
    return self.model.n_vocab

  @property
  def vocab_size(self) -> int:
    """Vocabulary size, including _base_vocab_size and extra_ids."""
    return self.model.n_vocab + len(self.special_tokens)

  def _encode(self, s: str) -> Sequence[int]:
    """Encode a python string as a list of integers.

    Args:
      s: a string

    Returns:
      a list of integers
    """
    # The tiktoken tokenizer can handle <=400k chars without
    # pyo3_runtime.PanicException.
    TIKTOKEN_MAX_ENCODE_CHARS = 400_000  # pylint: disable=invalid-name

    # https://github.com/openai/tiktoken/issues/195
    # Here we iterate over subsequences and split if we exceed the limit
    # of max consecutive non-whitespace or whitespace characters.
    MAX_NO_WHITESPACES_CHARS = 25_000  # pylint: disable=invalid-name

    # pylint: disable=g-complex-comprehension
    substrs = (
        substr
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
        for substr in self._split_whitespaces_or_nonwhitespaces(
            s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
        )
    )
    t: List[int] = []
    for substr in substrs:
      t.extend(
          self.model.encode(
              substr,
          )
      )
    return t

  def _decode(self, ids: Sequence[int]) -> str:
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers

    Returns:
      a string
    """
    return self.model.decode(cast(List[int], ids))

  def _encode_tf(self, s: tf.Tensor) -> tf.Tensor:
    """Encode a string tf.Tensor.

    Args:
      s: a tf.Scalar or a 1d tf.Tensor in [batch_size] with dtype tf.string

    Returns:
      a 1d tf.Tensor or
      a tf.RaggedTensor in [batch_size, None] with dtype tf.int32
    """

    def _encode_py_func(text) -> Sequence[int]:
      text = tf.compat.as_text(text.numpy().decode("UTF-8"))
      result = self.encode(text)
      result = tf.expand_dims(result, 0)
      return tf.cast(result, tf.int32)

    def _encode_func():
      result = tf.py_function(func=_encode_py_func, inp=[s], Tout=tf.int32)
      result = tf.squeeze(result, 0)
      result.set_shape([None])
      return result

    def _batch_encode_py_func(strs) -> Sequence[Sequence[int]]:
      texts = []
      for s in strs.numpy():
        texts.append(tf.compat.as_text(s.decode("UTF-8")))
      results = self.model.encode_batch(texts)
      return tf.ragged.constant(results, dtype=tf.int32)

    def _batch_encode_func():
      return tf.py_function(
          func=_batch_encode_py_func,
          inp=[s],
          Tout=tf.RaggedTensorSpec(shape=[s.shape[0], None], dtype=tf.int32),
      )

    # pylint: disable=g-explicit-length-test
    if len(s.shape) > 0:
      return _batch_encode_func()
    return _encode_func()

  def _decode_tf(self, ids: tf.Tensor) -> tf.Tensor:
    """Decode in TensorFlow.

    Args:
      ids: a 1d or 2d (in [batch_size, None]) tf.Tensor with dtype tf.int32

    Returns:
      a tf.Scaler or 1d (in [batch_size]) tf.Tensor with dtype tf.string
    """

    def _decode_py_func(token_ids: Sequence[int]) -> str:
      result = self.decode(token_ids)
      return result

    def _decode_func():
      result = tf.py_function(func=_decode_py_func, inp=[ids], Tout=tf.string)
      return result

    def _batch_decode_py_func(token_ids) -> str:
      result = self.model.decode_batch(token_ids.numpy())
      result = tf.expand_dims(result, 0)
      return result

    def _batch_decode_func():
      result = tf.py_function(
          func=_batch_decode_py_func, inp=[ids], Tout=tf.string
      )
      result = tf.squeeze(result, 0)
      result.set_shape([None])
      return result

    if len(ids.shape) > 1:
      return _batch_decode_func()
    return _decode_func()

  @property
  def unk_id(self) -> int:
    return -1

  @property
  def bos_id(self) -> int:
    return self._bos_token_id

  @property
  def eos_id(self) -> int:
    return self._eos_token_id

  @property
  def pad_id(self) -> int:
    return self._pad_token_id

  @staticmethod
  def _split_whitespaces_or_nonwhitespaces(
      s: str, max_consecutive_slice_len: int
  ) -> Iterator[str]:
    """Splits the string `s`.

    Args:
      s: The string to split.
      max_consecutive_slice_len: The maximum number of consecutive whitespaces
        or consecutive non-whitespaces in each substring.

    Yields:
      The substrings.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if s else False
    slice_start = 0

    for i in range(len(s)):
      is_now_space = s[i].isspace()

      if current_slice_is_space ^ is_now_space:
        current_slice_len = 1
        current_slice_is_space = is_now_space
      else:
        current_slice_len += 1
        if current_slice_len > max_consecutive_slice_len:
          yield s[slice_start:i]
          slice_start = i
          current_slice_len = 1
    yield s[slice_start:]
