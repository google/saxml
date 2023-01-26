"""Common function calls for language model methods."""

from typing import Any, Optional, Tuple, Union, Mapping

import jax
from jax import numpy as jnp
import numpy as np
from praxis import py_utils
from praxis import pytypes
from saxml.server.pax import branch_selection
import tensorflow as tf


NestedMap = py_utils.NestedMap
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
NestedTfTensor = pytypes.Nested[tf.Tensor]
NestedNpOrTfTensor = Union[NestedNpTensor, NestedTfTensor]


def decode_tf_tokenize_inputs(
    texts: tf.Tensor,
    tokenizer: Any,
    max_input_seq_len: int,
    encoder_decoder_model: bool = False,
) -> Tuple[
    tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor], Optional[tf.Tensor]
]:
  """Tokenize inputs for decoding."""
  ids, labels, paddings = tokenizer.StringsToIds(texts, max_input_seq_len)

  if encoder_decoder_model:
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


def decode_tf_post_processing(
    compute_outputs: NestedNpOrTfTensor,
    tokenizer: Any,
    encoder_decoder_model: bool = False,
    include_prefix_in_result: bool = False,
) -> NestedNpOrTfTensor:
  """Post-process the outputs using TF ops.

  This also implements `ExportableToSavedModel.tf_post_processing`.

  Args:
    compute_outputs: the outputs of the model function.
    tokenizer: tokenizer to decode ids to strings.
    encoder_decoder_model: if it is encoder_decoder_model.
    include_prefix_in_result: if include prefix in result or not.

  Returns:
    A mapping that contains the decoded tensors, scores and ids of the topk
    results.
  """
  assert isinstance(compute_outputs, py_utils.NestedMap)
  if encoder_decoder_model:
    # Post process for the encoder decoder model.
    # output_ids: [b, seqlen]
    # scores: [b]
    decoded = tf.map_fn(
        tokenizer.IdsToStrings,
        compute_outputs.output_ids,
        fn_output_signature=tf.string,
    )
    decoded = tf.expand_dims(decoded, axis=-1)
    output_ids = tf.expand_dims(compute_outputs.output_ids, axis=-1)
    scores = tf.expand_dims(compute_outputs.scores, axis=-1)
  else:
    # prefix_lengths: [b]
    # decode_lengths: [b, num_samples]
    # output_ids: [b, num_samples, seqlen]
    # scores: [b, num_samples]
    if include_prefix_in_result:
      output_ids = compute_outputs.output_ids
      decode_lengths = compute_outputs.decode_lengths
    else:

      def remove_prefix(ids_and_prefix_length):
        ids, prefix_length = ids_and_prefix_length
        return tf.pad(ids[:, prefix_length:], [[0, 0], [0, prefix_length]])

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
  return {
      'topk_decoded': decoded,
      'topk_scores': scores,
      'topk_ids': output_ids,
  }


def decode_get_scores(
    result: NestedMap, encoder_decoder_model: bool = False, host=False
):
  """Get scores from decoding results."""
  if encoder_decoder_model:
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


def decode_fetch_output(
    model_fn_outputs: NestedJTensor,
    model_fn_inputs: NestedJTensor,
    encoder_decoder_model=False,
) -> NestedJTensor:
  """Fetch output for decode."""
  assert len(model_fn_outputs[0]) == 3
  # Extract the per example outputs and discard weighted scalars and metrics.
  _, result, _ = model_fn_outputs[0]
  output_ids = result.output_ids  # [batch_size, num_samples, seqlen].
  scores = decode_get_scores(result, encoder_decoder_model)

  if encoder_decoder_model:
    decode_lengths = None
    prefix_lengths = None
  else:
    # [batch_size, num_samples]
    decode_lengths = result.decode_lengths
    # [batch_size]
    prefix_lengths = model_fn_inputs.prefix_lengths

  return NestedMap(
      output_ids=output_ids,
      decode_lengths=decode_lengths,
      prefix_lengths=prefix_lengths,
      scores=scores,
  )


def score_tf_tokenize_inputs(
    prefixes: tf.Tensor, suffixes: tf.Tensor, tokenizer: Any,
    max_seq_len: int, include_eos: bool,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Tokenize inputs for scoring."""
  seqlen = max_seq_len
  shape = [None, seqlen]
  pfx_ids, pfx_labels, pfx_paddings = tokenizer.StringsToIds(
      prefixes, max_length=seqlen
  )
  (pfx_ids, pfx_labels, pfx_paddings) = (
      tf.ensure_shape(pfx_ids, shape),
      tf.ensure_shape(pfx_labels, shape),
      tf.ensure_shape(pfx_paddings, shape),
  )
  sfx_ids, sfx_labels, sfx_paddings = tokenizer.StringsToIds(
      suffixes, max_length=seqlen
  )
  (sfx_ids, sfx_labels, sfx_paddings) = (
      tf.ensure_shape(sfx_ids, shape),
      tf.ensure_shape(sfx_labels, shape),
      tf.ensure_shape(sfx_paddings, shape),
  )
  # Lengths are with EOS in labels, and SOS in ids (will be adjusted if not).
  pfx_lengths = seqlen - tf.cast(tf.reduce_sum(pfx_paddings, 1), dtype=tf.int32)
  sfx_lengths = seqlen - tf.cast(tf.reduce_sum(sfx_paddings, 1), dtype=tf.int32)

  score_masks = pfx_paddings
  if tokenizer.hparams.append_eos:
    # Left-shift to exclude prefix EOS.
    score_masks = tf.pad(
        score_masks[:, 1:], [[0, 0], [0, 1]], constant_values=1.0
    )
    full_lengths = pfx_lengths + sfx_lengths - 1
  else:
    # pfx_ids are not complete. Reconstruct by appending SOS to labels.
    pfx_ids = tf.concat([pfx_ids[:, :1], pfx_labels[:, :-1]], axis=1)
    full_lengths = pfx_lengths + sfx_lengths + 1
    pfx_lengths += 1
    sfx_lengths += 1
    assert (
        not include_eos
    ), 'tokenizer cannot append EOS'

  # Remove SOS from suffix
  sfx_ids = sfx_labels

  def _combine(pfx, pfx_lens, sfx, sfx_lens):
    r_pfx = tf.RaggedTensor.from_tensor(pfx, pfx_lens)
    r_sfx = tf.RaggedTensor.from_tensor(sfx, sfx_lens)
    return tf.concat([r_pfx, r_sfx], axis=1).to_tensor(shape=pfx.shape)

  # Do not include suffix EOS in ids.
  ids = _combine(pfx_ids, pfx_lengths, sfx_ids, sfx_lengths - 1)
  # Do not include prefix EOS in ids.
  labels = _combine(pfx_labels, pfx_lengths - 1, sfx_labels, sfx_lengths)

  paddings = tf.cast(
      tf.greater_equal(
          tf.range(seqlen, dtype=tf.int32), full_lengths[:, tf.newaxis]
      ),
      pfx_paddings.dtype,
  )
  inputs_indicator = tf.cast(
      tf.less(tf.range(seqlen, dtype=tf.int32), pfx_lengths[:, tf.newaxis]),
      tf.int32,
  )
  weights = 1.0 - paddings
  return ids, labels, paddings, weights, score_masks, inputs_indicator


def bucketize_tokenized_inputs(
    bucket_keys: list[int], inputs: NestedMap
) -> NestedMap:
  """Bucketize tokenized input tensors.

  Args:
    bucket_keys: a bucket of sequence lengths.
    inputs: the tokenized tf.Tensors.

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
  branch_idx = branch_selector.get_branch_index_tf(branch_key)
  seqlen = tf.constant(branch_selector.branch_keys)[branch_idx]

  def _slice_fn(x):
    return x[:, :seqlen] if len(x.shape) == 2 else x

  return jax.tree_util.tree_map(_slice_fn, inputs)


def extra_inputs_to_tf_signature(
    sample_extra_inputs: Optional[Mapping[str, Any]], batch_size: Optional[int]
) -> Mapping[str, tf.TensorSpec]:
  """Generate input signature from sample extra inputs."""
  extra_tensor_specs = {}
  if sample_extra_inputs:
    for name, val in sample_extra_inputs.items():
      val_tf = tf.convert_to_tensor(val)
      extra_tensor_specs[name] = tf.TensorSpec(
          [batch_size, *val_tf.shape.as_list()], val_tf.dtype, name=name
      )
  return extra_tensor_specs
