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
"""Wraps a model with LMService APIs."""

import abc
import dataclasses
import functools
import json
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import host_callback as hcb
import numpy as np
from paxml import checkpoint_pb2
from praxis import base_layer
from praxis import base_model
from praxis import decoder_hparams
from praxis import decoder_utils
from praxis import py_utils
from praxis import pytypes
from saxml.server.jax import np_tf_sess_wrapper
from saxml.server.pax import branch_selection
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params
from saxml.server.services import lm_service
import tensorflow as tf

CheckpointType = checkpoint_pb2.CheckpointType
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedTfTensor = pytypes.Nested[tf.Tensor]
NestedNpOrTfTensor = Union[NestedNpTensor, NestedTfTensor]
LMMethodName = lm_service.LMMethodName
HostTensors = servable_model.HostTensors
ShapesAndDtypes = servable_model.ShapesAndDtypes


@dataclasses.dataclass(eq=True, frozen=True)
class InputShapeInfo(servable_model.InputShapeInfo):
  """Input shape information."""
  batch_size: int = -1
  seq_len: int = -1


class ScoreHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM score method.

  Attributes:
    max_input_seq_len: static sequence length dimension size. Inputs are padded
      or truncated to this size.
    include_eos_score: whether to add EOS score to the result.
  """
  max_input_seq_len: int = 0
  include_eos_score: bool = False


class DecodeHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM sample decode method.

  Attributes:
    max_input_seq_len: static sequence length dimension size. Inputs are padded
      or truncated to this size.
    decoder: decoder params.
    include_prefix_in_result: whether to include the input prefix in the result.
    encoder_decoder_model: whether this is an encoder decoder model.
  """
  max_input_seq_len: int = 0
  decoder: decoder_hparams.DecoderHParams = decoder_hparams.DecoderHParams()
  include_prefix_in_result: bool = False
  encoder_decoder_model: bool = False
  stream_interval_steps: int = 1


class TextToEmbeddingHParams(servable_model_params.ServableMethodParams):
  """HParameters for TextToEmbedding method.

  Attributes:
    max_input_seq_len: static sequence length dimension size. Inputs are padded
      or truncated to this size.
    output_embedding_name: The name of the embedding to use from the model's
      outputs.  Required.
  """
  max_input_seq_len: int = 0
  output_embedding_name: Optional[str] = None


class ServableLMModelParams(
    servable_model_params.ServableModelParams, metaclass=abc.ABCMeta):
  """A base class that each LM model config needs to implement for serving."""

  @abc.abstractmethod
  def serving_tokenizer(self):
    """Tokenizer params used by serving."""

  def methods(self) -> Dict[str, servable_model_params.ServableMethodParams]:
    methods = {}
    score = self.score()  # pylint: disable=assignment-from-none
    if score is not None:
      methods[LMMethodName.SCORE] = score
    generate = self.generate()  # pylint: disable=assignment-from-none
    if generate is not None:
      methods[LMMethodName.GENERATE] = generate
    generate_stream = self.generate_stream()  # pylint: disable=assignment-from-none
    if generate_stream is not None:
      methods[LMMethodName.GENERATE_STREAM] = generate_stream
    text_to_embedding = self.text_to_embedding()  # pylint: disable=assignment-from-none
    if text_to_embedding is not None:
      methods[LMMethodName.EMBED] = text_to_embedding
    return methods

  def score(self) -> Optional[ScoreHParams]:
    """Returns the params for the score method."""
    return None

  def generate(self) -> Optional[DecodeHParams]:
    """Returns the params for the decode method."""
    return None

  def generate_stream(self) -> Optional[DecodeHParams]:
    """Returns the params for the decode method."""
    return None

  def text_to_embedding(self) -> Optional[TextToEmbeddingHParams]:
    return None

  def create_model(self, primary_process_id: int) -> 'ServableLMModel':
    return ServableLMModel(self, primary_process_id, self.get_checkpoint_type())


class ServableLMMethod(servable_model.ServableMethod):
  """Implements common method of LM."""

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  @property
  def sorted_seq_lens(self) -> List[int]:
    """A list of sorted supported (ascending order) sequence lengths."""
    return sorted(self._bucket_keys) if self._bucket_keys else [-1]

  def get_sorted_input_shapes(self) -> List[InputShapeInfo]:
    result = []
    for batch_size in self._sorted_batch_sizes:
      for seq_len in self.sorted_seq_lens:
        result.append(InputShapeInfo(batch_size, seq_len))
    return result

  def deserialize_input_shape(self, unpadded_shape_str: str) -> InputShapeInfo:
    """Deserialize input shape from a str."""
    unpadded_shape_dict = json.loads(unpadded_shape_str)
    seq_len = unpadded_shape_dict.get('seq_len', self._dummy_bucket_key)
    return InputShapeInfo(
        batch_size=unpadded_shape_dict['batch_size'], seq_len=seq_len)

  def get_unpadded_shape(self, unpadded_batch_size,
                         inputs: HostTensors) -> InputShapeInfo:
    return InputShapeInfo(unpadded_batch_size,
                          self.get_max_seq_len_in_batch(inputs))

  def get_padded_input_shape(self,
                             unpadded_shape: InputShapeInfo) -> InputShapeInfo:
    """Get padded input shape.

    Args:
      unpadded_shape: Unpadded shape information contains batch size or sequence
        length.

    Returns:
      Padded input shape.
    Raises:
      ValueError if unpadded batch size or sequence length too large.
    """
    padded_shape = super().get_padded_input_shape(unpadded_shape)
    if self._bucket_keys is None:
      return InputShapeInfo(padded_shape.batch_size)

    seq_len = -1
    sorted_seq_lens = self.sorted_seq_lens
    for sl in sorted_seq_lens:
      if sl >= unpadded_shape.seq_len:
        seq_len = sl
        break

    if seq_len == -1:
      raise ValueError(
          f'Sequence length larger than maximum: {unpadded_shape.seq_len} vs '
          f'{sorted_seq_lens[-1]}')
    return InputShapeInfo(padded_shape.batch_size, seq_len)

  def get_max_seq_len_in_batch(self, inputs: HostTensors) -> int:
    """Get unpadded seq_len for inputs.

    Args:
      inputs: Host tensors.

    Returns:
      Unpadded sequence length for inputs.
    """
    if inputs is None or self._bucket_keys is None:
      return self._dummy_bucket_key
    paddings = getattr(inputs, 'paddings', None)
    if isinstance(inputs, tuple):
      for item in inputs:
        if 'paddings' in item:
          paddings = item['paddings']
          break
    if paddings is None:
      return self._dummy_bucket_key
    prefix_lengths = np.sum(1.0 - paddings, axis=-1).astype(np.int32)
    return np.max(prefix_lengths).item()

  def get_dummy_inputs(self, input_shape: InputShapeInfo) -> HostTensors:
    """Returns host tensors with dummy data at a batch size."""
    batched_input = self.pre_processing([self._dummy_input_sample] *
                                        input_shape.batch_size)

    def _slice_fn(x):
      """The function to slice at sequence dimension."""
      if (not isinstance(x, np.ndarray) or
          not hasattr(input_shape, 'seq_len') or
          input_shape.seq_len == self._dummy_bucket_key):
        return x
      if len(x.shape) == 2 and x.shape[1] >= input_shape.seq_len:
        return x[:, :input_shape.seq_len]
      return x

    return jax.tree_util.tree_map(_slice_fn, batched_input)

  def resize_host_array(
      self,
      x: np.ndarray,
      global_input_shape_dtype: ShapesAndDtypes,
      unpadded_input_shape: InputShapeInfo,
  ):
    """Resizes x to the desired shape.

    Args:
      x: Host tensor.
      global_input_shape_dtype: Global input shape and dtype for this tensor.
      unpadded_input_shape: Unpadded input shape.

    Returns:
      host array after padding or slice of x.
    """
    global_shape, _ = global_input_shape_dtype
    if unpadded_input_shape.seq_len != self._dummy_bucket_key and len(
        x.shape) == 2:
      # x's shape has the longest sequence length with trailing 0s.
      # Slice sequence which is the 2nd dim to have the desired sequence length.
      l = x.shape[1]
      full_l = global_shape[2]
      if l != full_l:
        assert l >= full_l
        x = x[:, :full_l]

    # Let the parent class handle the batch dim.
    x = super().resize_host_array(
        x, global_input_shape_dtype, unpadded_input_shape
    )
    return x

  def _get_longest_seqlen(self, inputs: NestedNpTensor) -> int:
    """Gets the longest sequence length in a batch."""
    if 'paddings' in inputs:
      prefix_lengths = np.sum(
          1.0 - inputs['paddings'], axis=-1).astype(np.int32)  # pytype: disable=attribute-error
      return np.max(prefix_lengths).item()
    return inputs['ids'].shape[1]

  def get_unpadded_branch_key(self, inputs: NestedNpTensor) -> int:
    return self._get_longest_seqlen(inputs)

  def get_branch_inputs(self, inputs: NestedJTensor,
                        branch_key: int) -> NestedJTensor:
    """Returns the inputs for a branch key.

    Args:
      inputs: inputs with padded sequence lengths.
      branch_key: branch_key is seqlen.

    Returns:
      Tensors sliced at sequence length dimension.
    """
    seqlen = branch_key

    def _slice_fn(x):
      """The function to slice at sequence dimension."""
      if not isinstance(x, JTensor):
        return x
      if len(x.shape) == 2 and x.shape[1] >= seqlen:
        return jax.lax.slice(x, [0, 0], [x.shape[0], seqlen])
      return x

    return jax.tree_util.tree_map(_slice_fn, inputs)

  def _bucketize_tf_preprocessed_inputs(self, inputs: NestedMap) -> NestedMap:
    if len(self.sorted_seq_lens) == 1 or 'paddings' not in inputs:
      return inputs

    branch_selector = branch_selection.BranchSelector(self.sorted_seq_lens)
    assert branch_selector.has_multiple_branches()
    prefix_lengths = tf.cast(
        tf.math.reduce_sum(1.0 - inputs['paddings'], axis=-1), tf.int32
    )
    branch_key = tf.math.reduce_max(prefix_lengths)
    branch_idx = branch_selector.get_branch_index_tf(branch_key)
    seqlen = tf.constant(branch_selector.branch_keys)[branch_idx]

    def _slice_fn(x):
      return x[:, :seqlen] if len(x.shape) == 2 else x

    return jax.tree_util.tree_map(_slice_fn, inputs)

  def get_maxlen(self) -> int:
    """Gets the max input sequence lengths."""
    raise NotImplementedError('get_maxlen not implemented')

  def output_seq_dim(self) -> int:
    """Gets the sequence dim in the output result."""
    raise NotImplementedError('output_seq_dim not implemented')

  def extra_pad_result(self, result: NestedJTensor,
                       branch_key: int) -> NestedJTensor:
    """Special paddings for some tensors."""
    return result

  def pad_result(self, result: NestedJTensor, pad_len: int,
                 seq_dim: int) -> NestedJTensor:
    """Pads the result at sequence dimension."""

    def _pad_fn(x):
      if not isinstance(x, JTensor) or len(x.shape) < seq_dim + 1:
        return x
      paddings = [[0, 0]] * len(x.shape)
      paddings[seq_dim] = [0, max(0, pad_len)]
      padded = jnp.pad(x, paddings)
      return padded

    return jax.tree_map(_pad_fn, result)

  def post_process_branch_outputs(self, outputs: NestedJTensor,
                                  branch_key: int) -> NestedJTensor:
    """Post process branch outputs."""
    seqlen = branch_key
    maxlen = self.get_maxlen()
    result, state = outputs
    padded_result = self.pad_result(result, maxlen - seqlen,
                                    self.output_seq_dim())
    padded_result = self.extra_pad_result(padded_result, branch_key)
    padded_state = self.pad_result(state, maxlen - seqlen, 1)
    return padded_result, padded_state

  @property
  def model_fn_input_polymorphic_shape(self) -> pytypes.Nested[str]:
    """Returns a batch polymorphic shape for jax2tf."""
    batched_host_dummy = self.get_dummy_inputs(InputShapeInfo(self.batch_size))
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy,
        self.batch_size,
        [self.default_extra_inputs] * self.batch_size,
    )

    batch_pattern = 'b' if len(self.sorted_batch_sizes) > 1 else '_'
    if len(self.sorted_seq_lens) > 1:
      seq_pattern = f'{batch_pattern}, t'
    else:
      seq_pattern = f'{batch_pattern}, _'
    return jax.tree_util.tree_map(
        lambda x: seq_pattern if len(x.shape) == 2 else f'{batch_pattern}, ...',
        batched_host_dummy,
    )


class LMScoreMethod(ServableLMMethod):
  """Implements the score method of an LM."""

  def __init__(self,
               model: base_model.BaseModel,
               model_state: servable_model.ServableModelState,
               prng_key: PRNGKey,
               score_params: ScoreHParams,
               tokenizer_p: Any,
               exportable: bool = False):
    self._tokenizer = tokenizer_p.Instantiate()
    self._score_params = score_params
    dummy_input_sample = ('', [''])
    logging.info('Using np_tf_sess_wrapper on LMScoreMethod.tf_pre_processing')
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    super().__init__(
        model,
        'compute_predictions',
        model_state,
        score_params,
        prng_key,
        dummy_input_sample,
        exportable=exportable)

  def fetch_output(self, model_fn_outputs: NestedJTensor,
                   model_fn_inputs: NestedJTensor) -> NestedJTensor:
    if 'scores' in model_fn_outputs[0]:
      # Custom scores.
      return model_fn_outputs[0]['scores']
    # per_token_xent or per_example_xnent is -logprobs. We return the negative
    # value so that higher score is better.
    if 'per_token_xent' not in model_fn_outputs[0]:
      assert 'per_example_xent' in model_fn_outputs[0]
      assert model_fn_outputs[0].per_example_xent.ndim == 1
      return -model_fn_outputs[0].per_example_xent
    assert len(model_fn_outputs[0].per_token_xent.shape) > 1
    xnent_len = model_fn_outputs[0].per_token_xent.shape[1]
    assert xnent_len == model_fn_inputs.ids.shape[1]
    per_token_logprobs = -model_fn_outputs[0].per_token_xent
    non_paddings = 1.0 - model_fn_inputs.paddings
    if (not self._score_params.include_eos_score and
        self._tokenizer.hparams.append_eos):
      non_paddings = jnp.pad(
          # TODO(b/263808957): change back to non_paddings[:, 1:] once the bug
          # is fixed.
          jax.lax.dynamic_slice_in_dim(
              non_paddings, 1, non_paddings.shape[1] - 1, axis=1),
          [[0, 0], [0, 1]],
      )
    return jnp.sum(
        per_token_logprobs * model_fn_inputs.score_masks * non_paddings,
        axis=-1,
        keepdims=True)

  def get_maxlen(self) -> int:
    return self._score_params.max_input_seq_len

  def output_seq_dim(self) -> int:
    return 1

  def _tf_tokenize_inputs(
      self, prefixes: tf.Tensor, suffixes: tf.Tensor
  ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    seqlen = self._score_params.max_input_seq_len
    shape = [None, seqlen]
    pfx_ids, pfx_labels, pfx_paddings = self._tokenizer.StringsToIds(
        prefixes, max_length=seqlen)
    (pfx_ids, pfx_labels, pfx_paddings) = (tf.ensure_shape(pfx_ids, shape),
                                           tf.ensure_shape(pfx_labels, shape),
                                           tf.ensure_shape(pfx_paddings, shape))
    sfx_ids, sfx_labels, sfx_paddings = self._tokenizer.StringsToIds(
        suffixes, max_length=seqlen)
    (sfx_ids, sfx_labels, sfx_paddings) = (tf.ensure_shape(sfx_ids, shape),
                                           tf.ensure_shape(sfx_labels, shape),
                                           tf.ensure_shape(sfx_paddings, shape))
    # Lengths are with EOS in labels, and SOS in ids (will be adjusted if not).
    pfx_lengths = seqlen - tf.cast(
        tf.reduce_sum(pfx_paddings, 1), dtype=tf.int32)
    sfx_lengths = seqlen - tf.cast(
        tf.reduce_sum(sfx_paddings, 1), dtype=tf.int32)

    score_masks = pfx_paddings
    if self._tokenizer.hparams.append_eos:
      # Left-shift to exclude prefix EOS.
      score_masks = tf.pad(
          score_masks[:, 1:], [[0, 0], [0, 1]], constant_values=1.0)
      full_lengths = pfx_lengths + sfx_lengths - 1
    else:
      # pfx_ids are not complete. Reconstruct by appending SOS to labels.
      pfx_ids = tf.concat([pfx_ids[:, :1], pfx_labels[:, :-1]], axis=1)
      full_lengths = pfx_lengths + sfx_lengths + 1
      pfx_lengths += 1
      sfx_lengths += 1
      assert not self._score_params.include_eos_score, (
          'tokenizer cannot append EOS')

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
            tf.range(seqlen, dtype=tf.int32), full_lengths[:, tf.newaxis]),
        pfx_paddings.dtype)
    inputs_indicator = tf.cast(
        tf.less(tf.range(seqlen, dtype=tf.int32), pfx_lengths[:, tf.newaxis]),
        tf.int32)
    weights = 1.0 - paddings
    return ids, labels, paddings, weights, score_masks, inputs_indicator

  def pre_processing(self,
                     raw_inputs: List[Tuple[str, List[str]]]) -> NestedNpTensor:
    prefixes = np.array([prefix for prefix, _ in raw_inputs])
    for _, suffix in raw_inputs:
      assert len(suffix) <= 1, (
          'Only one suffix score is supported in lm.score')
    suffixes = np.array([suffix[0] for _, suffix in raw_inputs])
    return self._tf_sess_pre_processing(prefixes, suffixes)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[float]:
    assert isinstance(compute_outputs, pytypes.NpTensor)
    scores = list(compute_outputs.astype(float))
    return scores

  def tf_pre_processing(
      self,
      prefixes: NestedNpOrTfTensor,
      suffixes: NestedNpOrTfTensor,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `prefixes` and `suffixes` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`.

    Args:
      prefixes: the prefix text batch of shape [batch_size].
      suffixes: the suffix text batch of shape [batch_size].
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    (ids, labels, paddings, weights, score_masks,
     inputs_indicator) = self._tf_tokenize_inputs(prefixes, suffixes)

    preprocessed = py_utils.NestedMap(
        ids=ids,
        labels=labels,
        paddings=paddings,
        weights=weights,
        score_masks=score_masks,
        inputs_indicator=inputs_indicator,
    )
    if bucketize_inputs:
      return self._bucketize_tf_preprocessed_inputs(preprocessed)
    return preprocessed

  def tf_post_processing(
      self, compute_outputs: NestedNpOrTfTensor) -> NestedNpOrTfTensor:
    """Implements `ExportableToSavedModel.tf_post_processing`."""
    return {'scores': compute_outputs}

  def input_signature(self, batch_size: Optional[int]) -> list[tf.TensorSpec]:
    """Implements `ExportableToSavedModel.input_signature`."""
    return [
        tf.TensorSpec([batch_size], dtype=tf.string, name='prefixes'),
        tf.TensorSpec([batch_size], dtype=tf.string, name='suffixes')
    ]

  @property
  def extra_trackables(self) -> Any:
    """Implements `ExportableToSavedModel.extra_trackables`."""
    return None


class LMDecodeMethod(ServableLMMethod):
  """Base decode method of an LM."""

  def __init__(self,
               model: base_model.BaseModel,
               model_state: servable_model.ServableModelState,
               prng_key: PRNGKey,
               method_hparams: DecodeHParams,
               tokenizer_p: Any,
               exportable: bool = False,
               streamable: bool = False):
    self._tokenizer = tokenizer_p.Instantiate()
    self._method_hparams = method_hparams
    dummy_input_sample = ''
    if isinstance(method_hparams, DecodeHParams):
      self._include_prefix_in_result = method_hparams.include_prefix_in_result
    logging.info('Using np_tf_sess_wrapper on LMDecodeMethod.tf_pre_processing')
    self._tf_sess_pre_processing = np_tf_sess_wrapper.wrap_tf_session(
        # `bucketize_inputs` is only used in SavedModel export. The sax-native
        # serving has an equivalent bucketization after `pre_processing`.
        lambda *args: self.tf_pre_processing(*args, bucketize_inputs=False)
    )
    logging.info(
        'Using np_tf_sess_wrapper on LMDecodeMethod.tf_post_processing')
    self._tf_sess_post_processing = np_tf_sess_wrapper.wrap_tf_session(
        self.tf_post_processing, False)
    self._streamable = streamable
    logging.info('Initialize LMDecodeMethod to be streamable=%s.', streamable)

    def _init_stream_and_decode(new_ids):
      batch_size = tf.shape(new_ids)[:-1]
      return self._tokenizer.DecodeOnStream(
          new_ids, self._tokenizer.InitStream(batch_size)
      )

    self._tf_sess_first_stream_step = np_tf_sess_wrapper.wrap_tf_session(
        _init_stream_and_decode, False
    )
    self._tf_sess_stream_step = np_tf_sess_wrapper.wrap_tf_session(
        self._tokenizer.DecodeOnStream, False
    )
    self._tf_sess_stream_finish = np_tf_sess_wrapper.wrap_tf_session(
        self._tokenizer.FinishStream, False
    )

    super().__init__(
        model,
        'decode',
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
        exportable=exportable,
    )

  def call_model_function(self, inputs, mdl_vars, prng_key):
    k1, k2 = prng_key

    kwargs = {}
    if self.streamable:

      def callback_fn(x, _):
        assert self.model_state.is_primary_host
        self.enqueue_stream_output(x)

      kwargs['result_callback'] = decoder_utils.StreamingResultCallback(
          functools.partial(
              hcb.id_tap, callback_fn, device_index=self.callback_device_index),
          interval_steps=self._method_hparams.stream_interval_steps)

    outputs = self._model.apply(
        mdl_vars,
        input_batch=inputs,
        method=self._model.decode_with_params,
        mutable=[
            base_layer.NON_TRAINABLE,
            base_layer.DECODE_CACHE,
            base_layer.PREFIX_DECODE_CACHE,
        ],
        rngs={
            base_layer.PARAMS: k1,
            base_layer.RANDOM: k2,
        },
        decoder_params=self._method_hparams.decoder,
        **kwargs,
    )
    return outputs

  @property
  def streamable(self) -> bool:
    return self._streamable

  def fetch_output(self, model_fn_outputs: NestedJTensor,
                   model_fn_inputs: NestedJTensor) -> NestedJTensor:
    assert len(model_fn_outputs[0]) == 3
    # Extract the per example outputs and discard weighted scalars and metrics.
    _, result, _ = model_fn_outputs[0]
    output_ids = result.output_ids  # [batch_size, num_samples, seqlen].
    scores = self.get_scores(result)

    if self._method_hparams.encoder_decoder_model:
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

  def _tf_tokenize_inputs(
      self, texts: tf.Tensor
  ) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor],
             Optional[tf.Tensor]]:
    ids, labels, paddings = self._tokenizer.StringsToIds(
        texts, self._method_hparams.max_input_seq_len)

    if self._method_hparams.encoder_decoder_model:
      # TODO(wangtao): consider change behavior of tokenizer. Encoder-decoder
      # model needs EOS at the end of the sequence, BOS is not needed at the
      # beginning of the sequence.
      return labels, None, None, None

    # weights, prefix_lengths computation is specific for decoder model.
    weights = 1.0 - paddings
    prefix_lengths = tf.reduce_sum(1.0 - paddings, axis=-1)
    if (hasattr(self._tokenizer, 'hparams') and
        not self._tokenizer.hparams.append_eos):
      # Use labels prepended with SOS as IDs.
      ids = tf.concat([ids[:, 0:1], labels[:, :-1]], axis=1)
      prefix_lengths += 1
    return ids, paddings, prefix_lengths, weights

  def pre_processing(self, raw_inputs: List[str]) -> NestedNpTensor:
    texts = np.array(raw_inputs)
    return self._tf_sess_pre_processing(texts)

  def get_maxlen(self) -> int:
    return self._method_hparams.max_input_seq_len

  def output_seq_dim(self) -> int:
    return 2

  def extra_pad_result(self, result: NestedJTensor,
                       branch_key: int) -> NestedJTensor:
    """Extra pad result from decoding."""
    seqlen = branch_key

    def _pad_fn(sub_result):
      paddings = [[0, 0], [0, self.get_maxlen() - seqlen]]
      for key in {'paddings', 'weights', 'ids'}:
        if key in sub_result:
          sub_result[key] = jnp.pad(sub_result[key], paddings)
      return sub_result

    return tuple([_pad_fn(sub_result) for sub_result in result])

  def post_processing(
      self,
      compute_outputs: NestedNpTensor) -> List[Tuple[List[str], List[float]]]:
    # A list of results for the inputs. Each element has multiple samples from
    # the decoding algorithm, which has a list of strings and a list of scores.
    post_processed = self._tf_sess_post_processing(compute_outputs)
    # post_processed = self.tf_post_processing(compute_outputs)
    batched_decoded = post_processed['topk_decoded']
    batched_scores = post_processed['topk_scores']
    return [([d.decode()
              for d in decoded], list(scores))
            for decoded, scores in zip(batched_decoded, batched_scores)]

  def post_processing_stream(
      self,
      compute_outputs: Optional[NestedNpTensor] = None,
      stream_state: Optional[Any] = None,
  ) -> Tuple[List[Tuple[List[str], List[float]]], Optional[Any]]:
    if compute_outputs is None and stream_state is None:
      raise ValueError('compute_outputs and stream_state cannot both be None')

    if compute_outputs is None:
      batch_decoded = self._tf_sess_stream_finish(stream_state)
      stream_state = None
      scores = np.zeros(batch_decoded.shape)
    elif stream_state is None:
      batch_decoded, stream_state = self._tf_sess_first_stream_step(
          compute_outputs['output_ids']
      )
      scores = compute_outputs['scores']
    else:
      batch_decoded, stream_state = self._tf_sess_stream_step(
          compute_outputs['output_ids'], stream_state
      )
      scores = compute_outputs['scores']

    return [(d, s) for (d, s) in zip(batch_decoded, scores)], stream_state

  def get_scores(self, result: NestedMap, host=False):
    """Get scores from decoding results."""
    if self._method_hparams.encoder_decoder_model:
      return result.logprobs

    if hasattr(result, 'scores'):
      return result.scores

    np_op = np if host else jnp

    if 'suffix_prompt_lengths' in result and 'suffix_lengths' in result:
      # Get scores for suffix rating ids.
      is_valid_output = np_op.logical_and(
          np_op.arange(result.output_ids.shape[-1]) >=
          result.decode_lengths[:, :, None] +
          result.suffix_prompt_lengths[:, :, None] - 1,
          np_op.arange(
              result.output_ids.shape[-1]) < result.decode_lengths[:, :, None] +
          result.suffix_lengths[:, :, None] - 1)
    else:
      is_valid_output = np_op.logical_and(
          np_op.arange(result.output_ids.shape[-1]) >=
          result.prefix_lengths[:, None, None],
          np_op.arange(
              result.output_ids.shape[-1]) < result.decode_lengths[:, :, None])
    # [batch_size, num_samples, seqlen]
    scores = np_op.where(is_valid_output, result.logprobs,
                         np_op.zeros_like(result.logprobs))
    # Scores are computed by excluding the prefix and padding.
    # [batch_size, num_samples]
    return np_op.sum(scores, axis=-1)

  def tf_pre_processing(
      self,
      texts: NestedNpOrTfTensor,
      extra_inputs: Optional[Mapping[str, Any]] = None,
      bucketize_inputs: bool = True,
  ) -> NestedTfTensor:
    """Tokenizes `texts` using TF ops.

    This also implements `ExportableToSavedModel.tf_pre_processing`. If extra
    inputs are provided in the input signature, the exported
    method will take a batched tensor too. See also the `input_signature` method
    of this class.

    Args:
      texts: the input text of shape [batch_size].
      extra_inputs: optional mapping of extra input key to tensor or tensor spec
        of shape [batch_size].
      bucketize_inputs: whether to bucketize the preprocessed inputs based on
        max sequence length in the batch.

    Returns:
      A NestedMap of preprocessed tensors.
    """
    ids, paddings, prefix_lengths, weights = self._tf_tokenize_inputs(texts)

    if self._method_hparams.encoder_decoder_model:
      # Preprocess for the encoder decoder model.
      batch_size = tf.shape(ids)[0]
      target_length = self._method_hparams.decoder.seqlen
      preprocessed = py_utils.NestedMap(
          encoder_input_tokens=ids,
          decoder_input_tokens=tf.ones((batch_size, target_length)))
    else:
      preprocessed = py_utils.NestedMap(
          ids=ids,
          paddings=paddings,
          prefix_lengths=tf.cast(prefix_lengths, tf.int32),
          weights=weights)

    if bucketize_inputs:
      preprocessed = self._bucketize_tf_preprocessed_inputs(preprocessed)

    if extra_inputs:
      preprocessed.update(extra_inputs)

    return preprocessed

  def tf_post_processing(
      self, compute_outputs: NestedNpOrTfTensor) -> NestedNpOrTfTensor:
    """Post-process the outputs using TF ops.

    This also implements `ExportableToSavedModel.tf_post_processing`.

    Args:
      compute_outputs: the outputs of the model function.

    Returns:
      A mapping that contains the decoded tensors, scores and ids of the topk
      results.
    """
    assert isinstance(compute_outputs, py_utils.NestedMap)
    if self._method_hparams.encoder_decoder_model:
      # Post process for the encoder decoder model.
      # output_ids: [b, seqlen]
      # scores: [b]
      decoded = tf.map_fn(
          self._tokenizer.IdsToStrings,
          compute_outputs.output_ids,
          fn_output_signature=tf.string)
      decoded = tf.expand_dims(decoded, axis=-1)
      output_ids = tf.expand_dims(compute_outputs.output_ids, axis=-1)
      scores = tf.expand_dims(compute_outputs.scores, axis=-1)
    else:
      # prefix_lengths: [b]
      # decode_lengths: [b, num_samples]
      # output_ids: [b, num_samples, seqlen]
      # scores: [b, num_samples]
      if self._include_prefix_in_result:
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
            compute_outputs.prefix_lengths, axis=-1)

      def decode(ids_and_lens):
        ids, lens = ids_and_lens
        return self._tokenizer.IdsToStrings(
            tf.RaggedTensor.from_tensor(ids, lens), lens)

      decoded = tf.map_fn(
          decode, (output_ids, decode_lengths), fn_output_signature=tf.string)
      scores = compute_outputs.scores
    return {
        'topk_decoded': decoded,
        'topk_scores': scores,
        'topk_ids': output_ids
    }

  def input_signature(
      self, batch_size: Optional[int]
  ) -> tuple[tf.TensorSpec, Mapping[str, tf.TensorSpec]]:
    """Implements `ExportableToSavedModel.input_signature`."""
    extra_tensor_specs = {}
    if self._extra_inputs:
      for name, val in self._extra_inputs.items():
        extra_tensor_specs[name] = tf.TensorSpec(
            [batch_size], tf.convert_to_tensor(val).dtype, name=name)
    return (
        tf.TensorSpec([batch_size], dtype=tf.string, name='text'),
        extra_tensor_specs
    )

  @property
  def extra_trackables(self) -> Any:
    """Implements `ExportableToSavedModel.extra_trackables`."""
    return None


class TextToEmbedding(servable_model.ServableMethod):
  """Implements text embedding method."""

  def __init__(self, model: base_model.BaseModel, model_fn_name: str,
               model_state: servable_model.ServableModelState,
               method_hparams: TextToEmbeddingHParams, prng_key: PRNGKey,
               dummy_input_sample: Any, model_config: Any):
    self._model_config = model_config
    self._model_config.init_for_serving()
    self._max_length = method_hparams.max_input_seq_len
    self._embedding_name = method_hparams.output_embedding_name
    super().__init__(model, model_fn_name, model_state, method_hparams,
                     prng_key, dummy_input_sample)

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  def fetch_output(self, model_fn_outputs: NestedJTensor,
                   model_fn_inputs: NestedJTensor) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    return py_utils.NestedMap(
        text_embedding=model_fn_outputs[0][self._embedding_name],)

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    ids, labels, weights, paddings = self._model_config.tokenize(
        np.array(raw_inputs), self._max_length)
    return py_utils.NestedMap(
        ids=np.array(ids),
        labels=np.array(labels),
        weights=np.array(weights),
        paddings=np.array(paddings))

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    return list(compute_outputs['text_embedding'])


class ServableLMModel(servable_model.ServableModel):
  """Represents an implementation for the LM service, backed by a model.

  This class is responsible for model loading, batch padding, etc.
  """

  def init_method(self, method: str, model: base_model.BaseModel,
                  model_state: servable_model.ServableModelState,
                  method_params: servable_model_params.ServableMethodParams,
                  prng_key: PRNGKey) -> servable_model.ServableMethod:
    assert isinstance(self.model_config, ServableLMModelParams)
    tokenizer_p = self.model_config.serving_tokenizer()
    if method == LMMethodName.SCORE:
      assert isinstance(method_params, ScoreHParams)
      return LMScoreMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True)
    elif method == LMMethodName.GENERATE:
      assert isinstance(method_params, DecodeHParams)
      return LMDecodeMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=True)
    elif method == LMMethodName.GENERATE_STREAM:
      assert isinstance(method_params, DecodeHParams)
      return LMDecodeMethod(
          model,
          model_state,
          prng_key,
          method_params,
          tokenizer_p,
          exportable=False,
          streamable=True)
    elif method == LMMethodName.EMBED:
      assert isinstance(method_params, TextToEmbeddingHParams)
      assert method_params.output_embedding_name is not None
      return TextToEmbedding(
          model,
          'compute_text_embedding',
          model_state,
          method_params,
          prng_key=prng_key,
          dummy_input_sample='test',
          model_config=self.model_config)
    else:
      raise NotImplementedError(f'method {method} not implemented')

  def supports_dummy_compute_on_primary(self) -> bool:
    if self.methods is None or not isinstance(self.methods, Dict):
      return True
    for method in list(self.methods.values()):
      has_multiple_seq_lens = (
          hasattr(method, 'sorted_seq_lens') and
          method.sorted_seq_lens is not None and
          len(method.sorted_seq_lens) > 1)
      if has_multiple_seq_lens:
        return False
    return True
