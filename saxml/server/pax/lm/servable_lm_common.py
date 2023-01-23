"""Common function calls for language model methods."""

from typing import Any, Optional, Tuple, Union

from jax import numpy as jnp
import numpy as np
from praxis import py_utils
from praxis import pytypes
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
