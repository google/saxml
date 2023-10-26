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
"""Common function calls for language model methods."""

import dataclasses
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import jax
from jax import numpy as jnp
import numpy as np
import orbax.export as oex
from praxis import py_utils
from praxis import pytypes
from saxml.server.pax import branch_selection
from saxml.server.pax import servable_model
import tensorflow as tf


NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
NestedTfTensor = pytypes.Nested[tf.Tensor]
NestedNpOrTfTensor = Union[NestedNpTensor, NestedTfTensor]
HostTensors = servable_model.HostTensors
ShapesAndDtypes = servable_model.ShapesAndDtypes
TensorSpec = Union[tf.TensorSpec, oex.TensorSpecWithDefault]
NpOrTfTensor = Union[pytypes.NpTensor, tf.Tensor]


@dataclasses.dataclass(eq=True, frozen=True)
class InputShapeInfo(servable_model.InputShapeInfo):
  """Input shape information."""

  batch_size: int = -1
  seq_len: int = -1


def decode_tf_tokenize_inputs(
    texts: tf.Tensor,
    tokenizer: Any,
    max_input_seq_len: int,
    t5_model: bool = False,
) -> Tuple[
    tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]
]:
  """Tokenize inputs for decoding."""
  ids, labels, paddings = tokenizer.StringsToIds(texts, max_input_seq_len)

  if t5_model:
    # TODO(wangtao): consider change behavior of tokenizer. Encoder-decoder
    # model needs EOS at the end of the sequence, BOS is not needed at the
    # beginning of the sequence.
    return labels, None, None, None

  # weights, prefix_lengths computation is specific for decoder model.
  weights = 1.0 - paddings
  prefix_lengths = tf.reduce_sum(1.0 - paddings, axis=-1)
  if hasattr(tokenizer, 'hparams') and not tokenizer.hparams.append_eos:
    # Use labels prepended with SOS as IDs.
    ids = tf.concat([ids[:, 0:1], labels[:, :-1]], axis=1)
    prefix_lengths += 1

  return ids, paddings, prefix_lengths, weights


def tf_post_processing_per_token_logprobs(
    outputs: NestedNpOrTfTensor,
    include_prefix_in_result: bool,
    output_ids: NpOrTfTensor,
    decode_lengths: NpOrTfTensor,
    remove_prefix: Callable[[Tuple[NpOrTfTensor, NpOrTfTensor]], tf.Tensor],
    tokenizer: Any,
) -> Dict[str, tf.Tensor]:
  """Post-process the per_token_logprobs related outputs using TF ops.

  Because JAX JIT AOT compilation requires the inputs and outputs to be of
  static shapes, the per_token_logprobs related outputs have fixed shapes with
  unnecessary paddings. This function removes those unnecessary paddings, thus
  reducing the output size, and consequently may reduce the network bandwidth
  and RPC latency between upper-level application systems and the model servers.

  Apart from removing paddings, this function also uses TF tokenizer to convert
  token IDs to UTF-8 bytes.

  Args:
    outputs: the output tensors returned by `decode_fetch_output`.
    include_prefix_in_result: whether to include prefixes in results. It's used
      to determine where does the sampling actually begins.
    output_ids: if include_prefix_in_result is False, this is the original
      output_ids without prefixes, otherwise, it's the same as the original one.
    decode_lengths: the output_ids' decoding lengths.
    remove_prefix: a function that takes (sequence, prefix_length), removes
      prefixes from each sequence, and adds zero right paddings to the sequence.
    tokenizer: a TF tokenizer for converting token IDs to UTF-8 bytes.

  Returns:
    A mapping from tensor names to TF tensors, the keys are:
      - sampled_tokens_per_step
      - sampled_logprobs_per_step
      - top_candidate_tokens_per_step
      - top_candidate_logprobs_per_step
    - If `num_per_token_logprobs` < 0, the tensors' shapes are:
      - sampled_tokens_per_step and sampled_logprobs_per_step:
        [batch, 0, 0]
      - top_candidate_tokens_per_step and top_candidate_logprobs_per_step:
        [batch, 0, 0, 0]
    - If `num_per_token_logprobs` == 0, the tensors' shapes are:
      - sampled_tokens_per_step and sampled_logprobs_per_step:
        [batch, num_samples, max_decode_length]
      - top_candidate_tokens_per_step and top_candidate_logprobs_per_step:
        [batch, 0, 0, 0]
    - If `num_per_token_logprobs` > 0, the tensors' shapes are:
      - sampled_tokens_per_step and sampled_logprobs_per_step:
        [batch, num_samples, max_decode_length]
      - top_candidate_tokens_per_step and top_candidate_logprobs_per_step:
        [batch, num_samples, max_decode_length, num_per_token_logprobs]
  """
  assert hasattr(outputs, 'top_candidate_ids')
  assert hasattr(outputs, 'top_candidate_logprobs')
  assert hasattr(outputs, 'logprobs')
  assert hasattr(outputs, 'num_per_token_logprobs')

  num_per_token_logprobs = outputs.num_per_token_logprobs
  top_candidate_ids = outputs.top_candidate_ids
  top_candidate_logprobs = outputs.top_candidate_logprobs

  # Make output_logprobs's shape be consistent with output_ids'.
  if include_prefix_in_result:
    output_logprobs = outputs.logprobs
  else:
    output_logprobs = tf.map_fn(
        remove_prefix,
        (outputs.logprobs, outputs.prefix_lengths),
        fn_output_signature=tf.float32,
    )

  # To save network bandwidth and reduce RPC latency, we want to return as few
  # data as possible, thus, we don't want to directly return the output tensors
  # from device, which has a fixed shape with unnecessary paddings.
  # But TF serving cannot return tf.RaggedTensor, so we compute the maximum
  # required size for the dimensions to be truncated.
  max_decode_length = tf.reduce_max(decode_lengths)
  max_num_candidates = tf.reduce_max(num_per_token_logprobs)

  def _top_candidate_tokens_and_logprobs():
    return (
        tokenizer.IdToString(
            top_candidate_ids[:, :, :max_decode_length, :max_num_candidates]
        ),
        top_candidate_logprobs[:, :, :max_decode_length, :max_num_candidates]
    )

  def _sampled_tokens_and_logprobs():
    return (
        tokenizer.IdToString(output_ids[:, :, :max_decode_length]),
        output_logprobs[:, :, :max_decode_length],
    )

  # Some serving system requires that the output tensors' 0th dimension must be
  # equal to the input tensors' batch size.
  # The input's batch size can be dynamic, we cannot create a static tensor
  # with a dimension of dynamic size.
  # So, instead of representing empty tensors as tf.constant([]), we represent
  # them as tensors with the same number of dimensions as ids or logprobs, but
  # all non-0th dimensions have size 0.

  def _empty_top_candidate_tokens_and_logprobs():
    return (
        tokenizer.IdToString(top_candidate_ids[:, :0, :0, :0]),
        top_candidate_logprobs[:, :0, :0, :0],
    )

  def _empty_sampled_tokens_and_logprobs():
    return (
        tokenizer.IdToString(output_ids[:, :0, :0]),
        output_logprobs[:, :0, :0],
    )

  top_candidate_tokens, top_candidate_logprobs = tf.cond(
      tf.reduce_all(num_per_token_logprobs > 0),
      _top_candidate_tokens_and_logprobs,
      _empty_top_candidate_tokens_and_logprobs,
  )

  sampled_tokens, sampled_logprobs = tf.cond(
      tf.reduce_all(num_per_token_logprobs >= 0),
      _sampled_tokens_and_logprobs,
      _empty_sampled_tokens_and_logprobs,
  )

  return {
      'sampled_tokens_per_step': sampled_tokens,
      'sampled_logprobs_per_step': sampled_logprobs,
      'top_candidate_tokens_per_step': top_candidate_tokens,
      'top_candidate_logprobs_per_step': top_candidate_logprobs,
  }


def decode_tf_post_processing(
    compute_outputs: NestedNpOrTfTensor,
    tokenizer: Any,
    t5_model: bool = False,
    include_prefix_in_result: bool = False,
) -> Dict[str, tf.Tensor]:
  """Post-process the outputs using TF ops.

  This also implements `ExportableToSavedModel.tf_post_processing`.

  Args:
    compute_outputs: the outputs of the model function.
    tokenizer: tokenizer to decode ids to strings.
    t5_model: if it is T5 encoder_decoder_model.
    include_prefix_in_result: if include prefix in result or not.

  Returns:
    A mapping that contains the decoded tensors, scores and ids of the topk
    results.
    Return mean entropy of tokens when available.
    Return per token logprobs when `num_per_token_logprobs` is specified.
  """
  assert isinstance(compute_outputs, py_utils.NestedMap)
  prefix_lengths = None
  per_token_logprobs_results = None
  if t5_model:
    # Post process for the encoder decoder model.
    # output_ids: [b, seqlen]
    # scores: [b]
    # decode_lengths: [b]
    decoded = tf.map_fn(
        tokenizer.IdsToStrings,
        compute_outputs.output_ids,
        fn_output_signature=tf.string,
    )
    decoded = tf.expand_dims(decoded, axis=-1)
    output_ids = tf.expand_dims(compute_outputs.output_ids, axis=-1)
    scores = tf.expand_dims(compute_outputs.scores, axis=-1)
    # Place holder since decode_lengths is None from decode_fetch_output.
    decode_lengths = tf.zeros_like(scores)
    mean_entropy = None
  else:
    # prefix_lengths: [b]
    # decode_lengths: [b, num_samples]
    # output_ids: [b, num_samples, seqlen]
    # scores: [b, num_samples]
    # logprobs: [b, num_samples, seqlen]
    # top_candidate_ids and top_candidate_logprobs:
    #   [b, num_samples, seqlen, sample_decode.MAX_NUM_PER_TOKEN_LOGPROBS]

    def remove_prefix(ids_and_prefix_length):
      ids, prefix_length = ids_and_prefix_length
      return tf.pad(ids[:, prefix_length:], [[0, 0], [0, prefix_length]])

    if include_prefix_in_result:
      output_ids = compute_outputs.output_ids
      decode_lengths = compute_outputs.decode_lengths
    else:
      output_ids = tf.map_fn(
          remove_prefix,
          (compute_outputs.output_ids, compute_outputs.prefix_lengths),
          fn_output_signature=tf.int32,
      )
      decode_lengths = compute_outputs.decode_lengths - tf.expand_dims(
          compute_outputs.prefix_lengths, axis=-1
      )

    def decode(ids_and_lens):
      ids, lens = ids_and_lens
      return tokenizer.IdsToStrings(
          tf.RaggedTensor.from_tensor(ids, lens), lens
      )

    decoded = tf.map_fn(
        decode, (output_ids, decode_lengths), fn_output_signature=tf.string
    )
    scores = compute_outputs.scores
    mean_entropy = None
    if hasattr(compute_outputs, 'mean_entropy'):
      mean_entropy = compute_outputs.mean_entropy
    if hasattr(compute_outputs, 'prefix_lengths'):
      prefix_lengths = compute_outputs.prefix_lengths

    if hasattr(compute_outputs, 'num_per_token_logprobs'):
      per_token_logprobs_results = tf_post_processing_per_token_logprobs(
          compute_outputs,
          include_prefix_in_result,
          output_ids,
          decode_lengths,
          remove_prefix,
          tokenizer,
      )

  ret = {
      'topk_decoded': decoded,
      'topk_scores': scores,
      'topk_ids': output_ids,
      'topk_decode_lengths': decode_lengths,
  }
  if mean_entropy is not None:
    ret['mean_entropy'] = mean_entropy
  if prefix_lengths is not None:
    ret['prefix_lengths'] = prefix_lengths
  if per_token_logprobs_results is not None:
    ret.update(per_token_logprobs_results)
  return ret


def decode_get_scores(result: NestedMap, t5_model: bool = False, host=False):
  """Get scores from decoding results."""
  if t5_model:
    return result.logprobs

  if hasattr(result, 'scores'):
    return result.scores

  np_op = np if host else jnp

  if 'suffix_prompt_lengths' in result and 'suffix_lengths' in result:
    # Get scores for suffix rating ids.
    is_valid_output = np_op.logical_and(
        np_op.arange(result.output_ids.shape[-1])
        >= result.decode_lengths[:, :, None]
        + result.suffix_prompt_lengths[:, :, None]
        - 1,
        np_op.arange(result.output_ids.shape[-1])
        < result.decode_lengths[:, :, None]
        + result.suffix_lengths[:, :, None]
        - 1,
    )
  else:
    is_valid_output = np_op.logical_and(
        np_op.arange(result.output_ids.shape[-1])
        >= result.prefix_lengths[:, None, None],
        np_op.arange(result.output_ids.shape[-1])
        < result.decode_lengths[:, :, None],
    )
  # [batch_size, num_samples, seqlen]
  scores = np_op.where(
      is_valid_output, result.logprobs, np_op.zeros_like(result.logprobs)
  )
  # Scores are computed by excluding the prefix and padding.
  # [batch_size, num_samples]
  return np_op.sum(scores, axis=-1)


def decode_get_mean_entropy(result: NestedMap,
                            decode_lengths: NestedJTensor,
                            prefix_lengths: NestedJTensor,
                            host=False):
  """Get mean of entropy from decoding results."""
  np_op = np if host else jnp
  output_length = decode_lengths - np_op.expand_dims(prefix_lengths, axis=-1)
  output_length = np_op.where(output_length > 0, output_length, 1)
  return np_op.sum(result.entropy, axis=-1) / output_length


def fetch_per_token_logprobs_related_outputs(
    device_input: NestedJTensor,
    device_output: NestedJTensor,
    result: NestedMap,
):
  """Fetches per_token_logprobs related tensors from device inputs and outputs.

  This function will be run on device. Its returned tensors will be passed to
  host for post processing.

  Args:
    device_input: input tensors to the device computation.
    device_output: output tensors from the device computation.
    result: the fetched tensors are inserted into this result.
  """
  assert hasattr(device_input, 'num_per_token_logprobs')
  assert hasattr(device_output, 'top_candidate_ids')
  assert hasattr(device_output, 'top_candidate_logprobs')
  assert hasattr(device_output, 'logprobs')
  result.num_per_token_logprobs = device_input.num_per_token_logprobs
  result.top_candidate_ids = device_output.top_candidate_ids
  result.top_candidate_logprobs = device_output.top_candidate_logprobs
  result.logprobs = device_output.logprobs


def decode_fetch_output(
    model_fn_outputs: NestedJTensor,
    model_fn_inputs: NestedJTensor,
    t5_model=False,
    fetch_prefix_length_from_inputs: bool = False,
) -> NestedJTensor:
  """Fetch output for decode."""
  assert len(model_fn_outputs[0]) == 3
  # Extract the per example outputs and discard weighted scalars and metrics.
  _, result, _ = model_fn_outputs[0]
  output_ids = result.output_ids  # [batch_size, num_samples, seqlen].
  scores = decode_get_scores(result, t5_model)

  if t5_model:
    decode_lengths = None
    prefix_lengths = None
  else:
    # [batch_size, num_samples]
    decode_lengths = result.decode_lengths
    # [batch_size]
    if fetch_prefix_length_from_inputs:
      # Special handle google3/learning/multipod/pax/core/flaxformer_models.py
      prefix_lengths = model_fn_inputs.prefix_lengths  # pytype: disable=attribute-error  # jax-ndarray
    else:
      prefix_lengths = result.prefix_lengths

  ret = NestedMap(
      output_ids=output_ids,
      decode_lengths=decode_lengths,
      prefix_lengths=prefix_lengths,
      scores=scores,
  )
  if hasattr(result, 'entropy'):
    ret.mean_entropy = decode_get_mean_entropy(
        result, decode_lengths, prefix_lengths)
  if (not t5_model) and hasattr(model_fn_inputs, 'num_per_token_logprobs'):
    fetch_per_token_logprobs_related_outputs(model_fn_inputs, result, ret)
  return ret


def tf_tokenize_inputs(
    prefixes: tf.Tensor,
    suffixes: tf.Tensor,
    tokenizer: Any,
    max_prefix_seq_len: int,
    max_suffix_seq_len: int,
    include_eos: bool,
) -> NestedMap:
  """Tokenize inputs."""
  seqlen = max_prefix_seq_len + max_suffix_seq_len
  if not max_suffix_seq_len:
    max_suffix_seq_len = max_prefix_seq_len
  output_shape = [None, seqlen]
  prefix_shape = [None, max_prefix_seq_len]
  suffix_shape = [None, max_suffix_seq_len]
  pfx_ids, pfx_labels, pfx_paddings = tokenizer.StringsToIds(
      prefixes, max_length=max_prefix_seq_len
  )
  (pfx_ids, pfx_labels, pfx_paddings) = (
      tf.ensure_shape(pfx_ids, prefix_shape),
      tf.ensure_shape(pfx_labels, prefix_shape),
      tf.ensure_shape(pfx_paddings, prefix_shape),
  )
  _, sfx_labels, sfx_paddings = tokenizer.StringsToIds(
      suffixes, max_length=max_suffix_seq_len
  )
  (sfx_labels, sfx_paddings) = (
      tf.ensure_shape(sfx_labels, suffix_shape),
      tf.ensure_shape(sfx_paddings, suffix_shape),
  )
  # Lengths are with EOS in labels, and SOS in ids (will be adjusted if not).
  pfx_lengths = max_prefix_seq_len - tf.cast(
      tf.reduce_sum(pfx_paddings, 1), dtype=tf.int32
  )
  sfx_lengths = max_suffix_seq_len - tf.cast(
      tf.reduce_sum(sfx_paddings, 1), dtype=tf.int32
  )

  pfx_score_masks = pfx_paddings
  sfx_score_masks = tf.ones_like(sfx_paddings, dtype=sfx_paddings.dtype)
  if hasattr(tokenizer, 'hparams') and tokenizer.hparams.append_eos:
    # Left-shift to exclude prefix EOS.
    pfx_score_masks = tf.pad(
        pfx_score_masks[:, 1:], [[0, 0], [0, 1]], constant_values=1.0
    )
    full_lengths = pfx_lengths + sfx_lengths - 1
  else:
    # pfx_ids are not complete. Reconstruct by appending SOS to labels.
    pfx_ids = tf.concat([pfx_ids[:, :1], pfx_labels[:, :-1]], axis=1)
    full_lengths = pfx_lengths + sfx_lengths + 1
    pfx_lengths += 1
    sfx_lengths += 1
    assert not include_eos, 'tokenizer cannot append EOS'

  # Remove SOS from suffix
  sfx_ids = sfx_labels
  pfx_inputs_indicator = tf.cast(
      tf.less(tf.range(max_prefix_seq_len, dtype=tf.int32),
              pfx_lengths[:, tf.newaxis]), tf.int32)
  sfx_inputs_indicator = tf.zeros_like(sfx_paddings, dtype=tf.int32)

  def _combine(pfx, pfx_lens, sfx, sfx_lens):
    r_pfx = tf.RaggedTensor.from_tensor(pfx, pfx_lens)
    r_sfx = tf.RaggedTensor.from_tensor(sfx, sfx_lens)
    # truncate left.
    r_combined = tf.concat([r_pfx, r_sfx], axis=1)[:, -seqlen:]
    return r_combined.to_tensor(shape=output_shape)

  # Do not include suffix EOS in ids.
  ids = _combine(pfx_ids, pfx_lengths, sfx_ids, sfx_lengths - 1)
  # Do not include prefix EOS in labels.
  labels = _combine(pfx_labels, pfx_lengths - 1, sfx_labels, sfx_lengths)
  score_masks = _combine(
      pfx_score_masks, pfx_lengths - 1, sfx_score_masks, sfx_lengths
  )
  inputs_indicator = _combine(
      pfx_inputs_indicator, pfx_lengths, sfx_inputs_indicator, sfx_lengths - 1
  )

  paddings = tf.cast(
      tf.greater_equal(
          tf.range(seqlen, dtype=tf.int32), full_lengths[:, tf.newaxis]
      ),
      pfx_paddings.dtype,
  )

  weights = 1.0 - paddings
  return NestedMap(
      ids=ids,
      labels=labels,
      paddings=paddings,
      weights=weights,
      score_masks=score_masks,
      inputs_indicator=inputs_indicator,
  )


def bucketize_tokenized_inputs(
    bucket_keys: List[int],
    inputs: NestedMap,
    branch_index: tf.Tensor | None = None,
) -> NestedMap:
  """Bucketize tokenized input tensors.

  Args:
    bucket_keys: a bucket of sequence lengths.
    inputs: the tokenized tf.Tensors.
    branch_index: if set and in [0, len(bucket_keys)), choose this index in
      `bucket_keys` regardless of the input paddings.

  Returns:
    A NestedMap of tensors padded to the nearest length in the bucket greater
    than or equal to the longest input sequence length in the batch.
  """
  if len(bucket_keys) == 1 or 'paddings' not in inputs:
    return inputs

  branch_selector = branch_selection.BranchSelector(bucket_keys)
  assert branch_selector.has_multiple_branches()
  seq_lengths = tf.cast(
      tf.math.reduce_sum(1.0 - inputs['paddings'], axis=-1), tf.int32
  )
  branch_key = tf.math.reduce_max(seq_lengths)
  if branch_index is not None:
    branch_index = tf.squeeze(branch_index[0])
    branch_idx = tf.cond(
        tf.math.logical_and(
            branch_index >= 0, branch_index < len(bucket_keys)
        ),
        lambda: branch_index,
        lambda: branch_selector.get_branch_index_tf(branch_key),
    )
  else:
    branch_idx = branch_selector.get_branch_index_tf(branch_key)
  seqlen = tf.constant(branch_selector.branch_keys)[branch_idx]

  def _slice_fn(x):
    return x[:, :seqlen] if len(x.shape) == 2 else x

  return jax.tree_util.tree_map(_slice_fn, inputs)


def extra_inputs_to_tf_signature(
    sample_extra_inputs: Optional[Mapping[str, Any]],
    batch_size: Optional[int],
    sample_extra_inputs_dtypes: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, TensorSpec]:
  """Generate input signature from sample extra inputs.

  All extra inputs will have default values repeated to shape `(batch_size,
  ...)`, or `(1, ...)` if batch_size is None.

  Args:
    sample_extra_inputs: Extra inputs defined in the model.
    batch_size: Batch size of the model. `None` batch_size means batch
      polymorphic, it usually comes when there is a list of batch sizes in SAX
      config.
    sample_extra_inputs_dtypes: Extra inputs' data types.

  Returns:
    A dict mapping extra inputs' names to converted TensorSpecs and default
    values.
  """
  extra_tensor_specs = {}
  sample_extra_inputs_dtypes = sample_extra_inputs_dtypes or {}
  if sample_extra_inputs:
    for name, val in sample_extra_inputs.items():
      val_tf = tf.convert_to_tensor(
          val, sample_extra_inputs_dtypes.get(name, None)
      )
      default_val = tf.expand_dims(val_tf, axis=0)
      if batch_size:
        default_val = tf.repeat(default_val, batch_size, axis=0)
      extra_tensor_specs[name] = oex.TensorSpecWithDefault(
          tf.TensorSpec(
              [batch_size, *val_tf.shape.as_list()], val_tf.dtype, name=name
          ),
          default_val,
      )
  return extra_tensor_specs


def deserialize_input_shape(
    unpadded_shape_str: str, dummy_bucket_key: int = -1
) -> InputShapeInfo:
  """Deserialize input shape from a str."""
  unpadded_shape_dict = json.loads(unpadded_shape_str)
  seq_len = unpadded_shape_dict.get('seq_len', dummy_bucket_key)
  return InputShapeInfo(
      batch_size=unpadded_shape_dict['batch_size'], seq_len=seq_len
  )


def get_padded_input_seq_len(
    seq_len: int, sorted_seq_lens: Sequence[int]
) -> int:
  """Get padded input shape.

  Args:
    seq_len: Unpadded sequence length.
    sorted_seq_lens: Sorted sequence lengths.

  Returns:
    Padded sequence length.
  Raises:
    ValueError if sequence length too large.
  """
  for sl in sorted_seq_lens:
    if sl >= seq_len:
      return sl

  raise ValueError(
      f'Sequence length larger than maximum: {seq_len} vs {sorted_seq_lens[-1]}'
  )


def get_max_seq_len_in_batch(
    inputs: HostTensors,
    dummy_bucket_key: int = -1,
    bucket_keys: Optional[Sequence[int]] = None,
) -> int:
  """Get unpadded seq_len for inputs.

  Args:
    inputs: Host tensors.
    dummy_bucket_keys: Dummy bucket key when bucket_keys is not defined.
    bucket_keys: Bucket_keys of the method.

  Returns:
    Unpadded sequence length for inputs.
  """
  if inputs is None or bucket_keys is None:
    return dummy_bucket_key
  paddings = getattr(inputs, 'paddings', None)
  if isinstance(inputs, tuple):
    for item in inputs:
      if 'paddings' in item:
        paddings = item['paddings']
        break
  if paddings is None:
    return dummy_bucket_key
  prefix_lengths = np.sum(1.0 - paddings, axis=-1).astype(np.int32)
  return np.max(prefix_lengths).item()


def handle_host_input_with_input_shape(
    batch_input: HostTensors, input_shape: InputShapeInfo
) -> HostTensors:
  """Pad or slice the host input with the input shape.

  Args:
    batch_input: Host tensors.
    input_shape: The desired input_shape for the batch_input.

  Returns:
  """

  def _slice_fn(x):
    """The function to slice at sequence dimension."""
    if (
        not isinstance(x, np.ndarray)
        or not hasattr(input_shape, 'seq_len')
        or input_shape.seq_len == -1
        or len(x.shape) < 2
    ):
      return x

    if x.shape[1] >= input_shape.seq_len:
      if len(x.shape) == 2:
        return x[:, : input_shape.seq_len]
      if len(x.shape) == 3:
        return x[:, : input_shape.seq_len, :]
    return x

  return jax.tree_util.tree_map(_slice_fn, batch_input)


def resize_host_array(
    x: np.ndarray,
    global_input_shape_dtype: ShapesAndDtypes,
    unpadded_input_shape: InputShapeInfo,
) -> HostTensors:
  """Resize host array to the deired shape.

  Args:
    x: Host tensor.
    global_input_shape_dtype: Global input shape and dtype for this tensor.
    unpadded_input_shape: Unpadded input shape.

  Returns:
    host array after padding or slice of x.
  """
  global_shape, _ = global_input_shape_dtype
  if unpadded_input_shape.seq_len != -1 and (
      len(x.shape) == 2 or len(x.shape) == 3
  ):
    # x's shape has the longest sequence length with trailing 0s.
    # Slice sequence which is the 2nd dim to have the desired sequence len.
    l = x.shape[1]
    full_l = global_shape[2]
    if l != full_l:
      assert l >= full_l
      if len(x.shape) == 2:
        x = x[:, :full_l]
      if len(x.shape) == 3:
        x = x[:, :full_l, :]
  return x
