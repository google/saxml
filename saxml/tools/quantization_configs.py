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
"""Quantization configs and utils."""

import abc
from typing import Optional, Tuple, Union

from jax import numpy as jnp
import numpy as np
from praxis.layers.quantization import operations
from praxis.layers.quantization import utils


class QuantizationConfigs(abc.ABC):
  """Base Quantization configs."""

  @property
  @abc.abstractmethod
  def configs(
      self,
  ) -> dict[str, tuple[list[int], float, int, Union[int, list[int]]]]:
    raise NotImplementedError

  def get_quantize_axis_and_factor(
      self,
      var_name: str,
  ) -> Optional[Tuple[list[int], float, int, Union[int, list[int]]]]:
    """Get quantization axis and factor for variable.

    Args:
      var_name: Name of the variable.

    Returns:
      Optionally a list of axis, factor, and number of bits for the variable. -1
      bits value means use default bit num for quantization.
    """

    for suffix, axis_and_factor in self.configs.items():
      # If var_name is a gcs path, it might contain trailing '/'.
      if var_name.endswith(suffix) or var_name.rstrip('/').endswith(suffix):
        return axis_and_factor
    return None


class QuantizationConfigsGPTJ(QuantizationConfigs):
  """Quantization config for GPTJ model."""

  factor = 1.0
  configs = {
      'ff_layer.ffn_layer1.linear.w': ([0], factor, 0, -1),
      'ff_layer.ffn_layer1_gate.linear.w': ([0], factor, 0, -1),
      'ff_layer.ffn_layer2.linear.w': ([0], factor, 0, -1),
      'self_attention.combined_qkv.w': ([1], factor, 1, -1),
      'self_attention.post.w': ([1, 2], factor, 0, -1),
  }


class QuantizationConfigsGPTJStacked(QuantizationConfigs):
  """Quantization config for GPTJ model."""

  factor = 1.0
  configs = {
      'ff_layer.ffn_layer1.linear.w': ([1], factor, 0, -1),
      'ff_layer.ffn_layer1_gate.linear.w': ([1], factor, 0, -1),
      'ff_layer.ffn_layer2.linear.w': ([1], factor, 0, -1),
      'self_attention.combined_qkv.w': ([2], factor, 1, -1),
      'self_attention.post.w': ([2, 3], factor, 0, -1),
  }


class QuantizationConfigsGemma2B(QuantizationConfigs):
  """Quantization config for Gemma 2B."""

  factor = 1.0
  configs = {
      'ff_layer.ffn_layer1.linear.w': ([0], factor, 0, -1),
      'ff_layer.ffn_layer1_gate.linear.w': ([0], factor, 0, -1),
      'ff_layer.ffn_layer2.linear.w': ([0], factor, 0, -1),
      'self_attention.post.w': ([1, 2], factor, 0, -1),
      'self_attention.key.w': ([0], factor, 0, -1),
      'self_attention.query.w': ([0], factor, 0, -1),
      'self_attention.value.w': ([0], factor, 0, -1),
  }


class QuantizationConfigsGemma7B(QuantizationConfigsGPTJ):
  """Quantization config for Gemma 7B."""


class QuantizationConfigsLLaMA70BWeightLinearOnlyInt8(QuantizationConfigs):
  """Quantization config for LLaMA70B model."""

  factor = 1.0
  configs = {
      'ff_layer.ffn_layer1.linear.w': ([0], factor, 0, 8),
      'ff_layer.ffn_layer1_gate.linear.w': ([0], factor, 0, 8),
      'ff_layer.ffn_layer2.linear.w': ([0], factor, 0, 8),
  }


class QuantizationConfigsLLaMA70BStackedWeightLinearOnlyInt8(
    QuantizationConfigs
):
  """Quantization config for LLaMA70B model."""

  factor = 1.0
  configs = {
      'ff_layer.ffn_layer1.linear.w': ([1], factor, 0, 8),
      'ff_layer.ffn_layer1_gate.linear.w': ([1], factor, 0, 8),
      'ff_layer.ffn_layer2.linear.w': ([1], factor, 0, 8),
  }


def quantize_tensor(
    var: np.ndarray,
    axis: list[int],
    factor: float = 1.0,
    sym: bool = True,
    number_bits: int = 8,
    use_fp: bool = False,
    add_scale_eps: bool = False,
    optimization_on_bound: bool = False,
    p_value: float = 1.0,
    per_channel: bool = False,
) -> Union[
    tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]
]:
  """Quantize a tensor.

  Args:
    var: The variable to be quantized.
    axis: The axis along which variable will be quantized.
    factor: The clipping factor.
    sym: Symmetric or asymmetric quantize the variable.
    number_bits: Number of bits for quantized value.
    use_fp: do fp with number of bits (i.e. fp8)
    add_scale_eps: add epsilon to scale to avoid division by zero, else it will
      replace zero scale by 1.
    optimization_on_bound: If p-mean bound optimizer is used.
    p_value: Exponent of the p-mean error metric. Default to 1.0 which is MAE.
    per_channel: use per-channel clipping optimization.

  Returns:
    Quantized tensors, along with scales and zero point.
  """
  assert number_bits == 8 or number_bits == 4
  qvar, scale, zp = operations.reduce_precision(
      jnp.asarray(var),
      contract_dims=axis,
      need_gradient=False,
      bits=number_bits,
      optimization_on_bound=optimization_on_bound,
      percentile=factor,
      use_symmetric=sym,
      use_fp=use_fp,
      add_scale_eps=add_scale_eps,
      p_value=p_value,
      per_channel=per_channel,
  )
  if sym:
    return np.array(qvar), np.array(jnp.squeeze(scale, axis=axis))  # pytype: disable=wrong-arg-types  # jnp-type
  else:
    return (
        np.array(qvar),
        # CAVEAT: the following squeezes should squeeze along the quantization
        # axis only.
        np.array(jnp.squeeze(scale)),
        np.array(jnp.squeeze(zp)),
    )


def pack_4bit(
    var: np.ndarray, pack_dim: int, packed_dtype: jnp.dtype = jnp.int32
) -> np.ndarray:
  return np.asarray(utils.pack_4bit(jnp.asarray(var), pack_dim, packed_dtype))
