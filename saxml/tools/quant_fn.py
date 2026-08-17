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
"""Quantization function class."""

from typing import Any, Iterable, Optional

import apache_beam as beam
from jax import numpy as jnp
from jax import lax
import numpy as np
import quantization_actions
import quantization_configs
import tensorstore_util

from flax.linen import fp8_ops


def _split_with_shard(
    var: np.ndarray,
    sharding_indices: list[int],
) -> Iterable[Any]:
  """Split the var into shards specified in sharding_indices."""
  assert sharding_indices
  gen_index = (
      np.r_[: var.shape[i] + 1 : var.shape[i] // block]
      for i, block in enumerate(sharding_indices)
  )

  reshape = [[np.s_[a:b] for a, b in zip(r[:-1], r[1:])] for r in gen_index]

  for idxs in np.ndindex(*sharding_indices):
    bounds = tuple(reshape[j][idxs[j]] for j in range(len(sharding_indices)))
    yield var[bounds]


def _write_with_shard(
    writer: tensorstore_util.TensorStoreWriter,
    tname: str,
    var: np.ndarray,
    sharding_indices: list[int],
) -> None:
  """Split the var and write to tensorstore."""
  shards = _split_with_shard(var, sharding_indices)
  for index, shard in enumerate(shards):
    writer.write_chunk_variable(tname, shard, list(var.shape), index)


class QuantFn(beam.DoFn):
  """ParDo function that unstacks a GDA tensor and optionally quantize it."""

  def __init__(
      self,
      input_dirs: list[str],
      output_dir: str,
      symmetric: bool,
  ):
    self._input_dirs = input_dirs
    self._output_dir = output_dir
    self._symmetric = symmetric

  def setup(self):
    self._readers = {}
    for input_dir in self._input_dirs:
      self._readers[input_dir] = tensorstore_util.LowRamTensorStoreReader(
          input_dir, max_num_bytes_per_read=8 * 2**30
      )
    self._writer = tensorstore_util.TensorStoreWriter(self._output_dir)

  def _write_quantized_tensor(
      self,
      action: quantization_actions.OptAction,
      number_bit: int,
      var: np.ndarray,
      scale: np.ndarray,
      zp: np.ndarray | None,
      suffix: str = '',
      sharding_indices: Optional[list[int]] = None,
      scale_act: np.ndarray | None = None,
  ) -> None:
    if number_bit == 4 and action.use_int4_packed_weights:
      # Extra pack needed for 4 bit.
      var = quantization_configs.pack_4bit(
          var, action.pack_dim, action.int4_packed_dtype
      )
    target_name = action.target_name + suffix
    if sharding_indices:
      _write_with_shard(self._writer, target_name, var, sharding_indices)
    else:
      self._writer.write_variable(target_name, var)
    scale_name = action.target_name + '_quantized_scale' + suffix
    self._writer.write_variable(scale_name, scale)
    if zp:
      zp_name = action.target_name + '_quantized_zp' + suffix
      self._writer.write_variable(zp_name, zp)
    scale_act_name = action.target_name + '_quantized_act_scale' + suffix
    self._writer.write_variable(scale_act_name, scale_act)

  def process(self, action: quantization_actions.OptAction):
    target_var = self._readers[action.input_dir].read_variable(
        action.source_name, action.layer_id, action.num_layers
    )

    if action.transpose_embedding:
      target_var = jnp.transpose(target_var)

    if action.quantize_axis:
      quantize_axis = action.quantize_axis
      quantize_factor = action.quantize_factor
      number_of_bits = action.number_bit
      optimization_on_bound = False
      p_value = 1.0
      per_channel = False
      scale_act = None
      if action.number_bit == 4 and action.use_optimization:
        optimization_on_bound = True
        p_value = action.optimization_p_value
        per_channel = False
      if action.per_channel_clipping:
        optimization_on_bound = True
        p_value = action.optimization_p_value
        per_channel = True

      if self._symmetric:
        if action.use_fp and number_of_bits == 8:
          assert per_channel == False, f'fp8 only supports per-tensor quantization.'
          scale_act_name = action.source_name[:-1] + 'einsum.input_scale'
          scale_kernel_name = action.source_name[:-1] + 'einsum.kernel_scale'
          scale_act = self._readers[action.input_dir].read_variable(
            scale_act_name, action.layer_id, action.num_layers
          )
          scale = self._readers[action.input_dir].read_variable(
            scale_kernel_name, action.layer_id, action.num_layers
          )
          compute_dtype = target_var.dtype
          target_var = fp8_ops.quantize(target_var, jnp.float8_e4m3fn, scale, compute_dtype)
          # This is needed since fp8 cannot be saved.
          target_var = lax.bitcast_convert_type(target_var, new_dtype=jnp.int8)
        else:
          target_var, scale = quantization_configs.quantize_tensor(
              target_var,
              quantize_axis,
              quantize_factor,
              True,
              number_of_bits,
              use_fp=action.use_fp,
              add_scale_eps=action.add_scale_eps,
              optimization_on_bound=optimization_on_bound,
              p_value=p_value,
              per_channel=per_channel,
          )
        zp = None
      else:
        target_var, scale, zp = quantization_configs.quantize_tensor(
            target_var,
            quantize_axis,
            quantize_factor,
            False,
            number_of_bits,
            use_fp=action.use_fp,
            add_scale_eps=action.add_scale_eps,
            optimization_on_bound=optimization_on_bound,
            p_value=p_value,
            per_channel=per_channel,
        )
      self._write_quantized_tensor(
          action,
          number_of_bits,
          target_var,
          scale,
          zp,
          sharding_indices=action.sharding_indices,
          scale_act = scale_act,
      )
    else:
      # no quantization.
      if action.sharding_indices:
        _write_with_shard(
            self._writer,
            action.target_name,
            target_var,
            sharding_indices=action.sharding_indices,
        )
      else:
        self._writer.write_variable(action.target_name, target_var)
