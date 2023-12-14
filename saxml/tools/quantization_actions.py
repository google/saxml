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
"""Unstack and quantize a model."""

import dataclasses
import logging
import re
from typing import Optional

from jax import numpy as jnp
import quantization_configs
import tensorstore_util


@dataclasses.dataclass
class OptAction:
  """Optimization action."""

  input_dir: str
  source_name: str
  target_name: str
  num_layers: int
  layer_id: Optional[int] = None
  dtype: Optional[str] = None
  quantize_axis: Optional[list[int]] = None
  quantize_factor: float = 1.0
  transpose_embedding: bool = False
  number_bit: int = 8
  pack_dim: int = 0
  use_optimization: bool = False
  optimization_p_value: float = 1.0
  per_channel_clipping: bool = False
  use_int4_packed_weights: bool = True
  int4_packed_dtype: jnp.dtype = jnp.int32
  use_fp: bool = False
  add_scale_eps: bool = False
  sharding_indices: Optional[list[int]] = None


def get_sharding_indexes(
    reader: tensorstore_util.TensorStoreReader, tname: str
) -> list[int]:
  """Gets overall sharding for a tensor."""
  var_shape = reader.get_variable_shape(tname)
  shard_shape = reader.get_variable_chunks(tname)
  assert len(var_shape) == len(shard_shape)
  sharding_indices = [a // b for a, b in zip(var_shape, shard_shape)]
  return sharding_indices


def _get_unrepeated_var_name(
    var_name: str,
    layer_id: int,
    sub_layer_id: int,
    num_sub_layers: int,
    stacked: bool = False,
) -> str:
  """Gets the variable name of the corresponding unrepeated model."""
  if stacked:
    return var_name
  target_var_name = var_name
  if '.repeat.' in var_name:
    prefix, postfix = var_name.split('repeat.sub.x_layers_%d' % sub_layer_id, 2)
    target_var_name = ''.join([
        prefix,
        'x_layers_%d' % (sub_layer_id + layer_id * num_sub_layers),
        postfix,
    ])
  return target_var_name


def create_actions(
    input_dir: str,
    config: quantization_configs.QuantizationConfigs,
    transpose: bool = False,
    quantize_embedding: bool = False,
    number_bit: int = 8,
    prune_params_regex: Optional[str] = None,
    skip_params_regex: Optional[str] = None,
    lm_var_prefix: str = '',
    use_optimization: bool = False,
    optimization_p_value: float = 1.0,
    per_channel_clipping: bool = False,
    use_int4_packed_weights: bool = True,
    int4_packed_dtype: jnp.dtype = jnp.int32,
    stacked: bool = False,
    use_fp: bool = False,
    add_scale_eps: bool = False,
    quantize_ngrammer_embedding: bool = False,
    model_var_tags: str = 'mdl_vars.params',
    preserve_shardings: bool = False,
) -> list[OptAction]:
  """Create quantization actions to run."""

  def _get_num_layers(var_name: str, stacked: bool = False) -> int:
    if stacked:
      return 1
    if '.repeat.' in var_name:
      return reader.get_variable_shape(var_name)[0]
    return 1

  def _get_num_sub_layers(var_names: list[str], stacked: bool = False) -> int:
    if stacked:
      return 1
    highest = 0
    for var_name in var_names:
      regex = r'[.]repeat[.]sub[.]x_layers_(\d+)'
      match = re.search(regex, var_name)
      if match is not None:
        highest = max(highest, int(match.group(1)))
    return highest + 1

  def _get_sub_layer(var_name: str) -> int:
    regex = r'[.]repeat[.]sub[.]x_layers_(\d+)'
    match = re.search(regex, var_name)
    if match is not None:
      return int(match.group(1))
    return -1

  prune_pattern = None
  if prune_params_regex:
    prune_pattern = re.compile(prune_params_regex)
  skip_pattern = None
  if skip_params_regex:
    skip_pattern = re.compile(skip_params_regex)
  reader = tensorstore_util.LowRamTensorStoreReader(input_dir)
  if not any(
      model_var_tags in source_name for source_name in reader.get_variables()
  ):
    raise ValueError(
        f'Cannot find any model variables under directory {input_dir}. '
        'Make sure input_dir contains variables, e.g.,'
        ' .../checkpoint_00010000/state'
    )

  actions = []
  num_sub_layers = _get_num_sub_layers(reader.get_variables(), stacked=stacked)
  for source_name in reader.get_variables():
    layer_wise_num_bits = number_bit
    if model_var_tags not in source_name and 'step' not in source_name:
      # Convert only mdl_vars.
      continue
    if prune_pattern and prune_pattern.search(source_name):
      continue

    num_layers = _get_num_layers(source_name, stacked=stacked)
    sub_layer_id = _get_sub_layer(source_name)
    for layer_id in range(num_layers):
      target_name = _get_unrepeated_var_name(
          source_name,
          layer_id,
          sub_layer_id,
          num_sub_layers,
          stacked=stacked,
      )
      target_name = target_name.replace(
          'mdl_vars.params.lm', f'mdl_vars.params.lm{lm_var_prefix}'
      )
      axis = None
      var_dtype = 'bfloat16'
      transpose_embedding = False
      quantize_factor = 1.0
      pack_dim = 0
      curr_config = config.get_quantize_axis_and_factor(source_name)
      if source_name == 'step':
        # case 1. For 'step', use uint32.
        var_dtype = 'uint32'
      elif source_name == 'mdl_vars.params.lm.softmax.logits_ffn.bias.b':
        # case 2. For embedding bias, rename if transpose is enabled.
        if transpose:
          target_name = f'mdl_vars.params.lm{lm_var_prefix}.softmax.bias.b'
      elif (
          source_name.startswith(
              'mdl_vars.params.lm.ngrammer.ngram_layer.ngram_table_'
          )
          and source_name.endswith('emb_var')
          and quantize_ngrammer_embedding
      ):
        # case 3. For ngrammer embedding weight, quantize if
        # quantize_ngrammer_embedding.
        var_dtype = 'int8'
        axis = [1]
        quantize_factor = config.factor  # pytype: disable=attribute-error
      elif (
          source_name
          == 'mdl_vars.non_trainable.lm.ngrammer.input_id_to_cluster_id_cache'
      ):
        # case 4. For
        # 'mdl_vars.non_trainable.lm.ngrammer.input_id_to_cluster_id_cache', use
        # int32.
        var_dtype = 'int32'
      elif curr_config:
        # case 5. For rest of the tensors, quantize or skip depends on 'config'.
        var_dtype = 'int8'
        axis, quantize_factor, pack_dim, bits = curr_config

        # override number_bit if necessary
        if bits != -1:
          if isinstance(bits, int):
            layer_wise_num_bits = bits
          else:
            assert num_layers == len(bits)
            layer_wise_num_bits = bits[layer_id]

      # Because of deprecated quantize_embedding arg, we need to additionally
      # keep these parts.
      if source_name in (
          'mdl_vars.params.lm.softmax.logits_ffn.linear.w',
          'mdl_vars.params.lm.softmax.subs_0.logits_ffn.linear.w',
          'mdl_vars.params.lm.softmax.subs_1.logits_ffn.linear.w',
      ):
        # case 6. For embedding weight, rename and transpose if transpose is
        # enabled. Quantize if quantize_embedding.
        if transpose:
          transpose_embedding = True
          if 'subs_0' in source_name:
            target_name = f'mdl_vars.params.lm{lm_var_prefix}.softmax.subs_0.w'
          elif 'subs_1' in source_name:
            # We don't transpose multimodal embedding because it already seems
            # to be transposed
            transpose_embedding = False
            target_name = source_name
          else:
            target_name = f'mdl_vars.params.lm{lm_var_prefix}.softmax.w'
        if quantize_embedding:
          logging.info('quantize_embedding is deprecated in favor of configs')
          var_dtype = 'int8'
          axis = [1] if transpose_embedding else [0]
          quantize_factor = config.factor  # pytype: disable=attribute-error
      elif source_name in (
          'mdl_vars.params.lm.softmax.subs_1.embedding_lookup.emb_var',
          'mdl_vars.params.lm.softmax.subs_1.embedding_lookup.extra_emb_var',
      ):
        # case 7. Quantize multimodal embeddings, we do not quantize projection
        # matrix yet and do not support transpose embeddings.
        if quantize_embedding:
          logging.info('quantize_embedding is deprecated in favor of configs')
          var_dtype = 'int8'
          axis = [1]  # we are quantizing embedding along the outer axis here
          quantize_factor = config.factor  # pytype: disable=attribute-error

      # Setting quantization configs back to non-quantize when the source name
      # matches the skip pattern.
      if skip_pattern and skip_pattern.search(source_name):
        var_dtype = 'bfloat16'
        axis = None
        quantize_factor = 1.0
        pack_dim = 0

      # get sharding sharding_indices
      sharding_indices = None
      if preserve_shardings:
        sharding_indices = get_sharding_indexes(reader, source_name)
        if not stacked and num_layers > 1:
          # assumes index 0 is layer index.
          assert sharding_indices
          sharding_indices = sharding_indices[1:]
      actions.append(
          OptAction(
              input_dir=input_dir,
              source_name=source_name,
              # If var_name is a gcs path, it might contain trailing '/'.
              target_name=target_name.rstrip('/'),
              num_layers=num_layers,
              layer_id=layer_id,
              dtype=var_dtype,
              quantize_axis=axis,
              quantize_factor=quantize_factor,
              transpose_embedding=transpose_embedding,
              number_bit=layer_wise_num_bits,
              pack_dim=pack_dim,
              use_optimization=use_optimization,
              optimization_p_value=optimization_p_value,
              per_channel_clipping=per_channel_clipping,
              use_int4_packed_weights=use_int4_packed_weights,
              int4_packed_dtype=int4_packed_dtype,
              use_fp=use_fp,
              add_scale_eps=add_scale_eps,
              sharding_indices=sharding_indices,
          )
      )
  return actions
