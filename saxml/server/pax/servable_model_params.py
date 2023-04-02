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
"""Base class for servable model configs."""

import abc
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from absl import logging

import jax
from jax.experimental import mesh_utils
import numpy as np
from paxml import base_experiment
from paxml import checkpoints
from praxis import base_hyperparams
from praxis import py_utils
from praxis.layers.quantization import quantization_hparams
from saxml.server import servable_model_params
from saxml.server import utils

NestedMap = py_utils.NestedMap
QuantizationType = quantization_hparams.QuantizationType
QuantizationMode = quantization_hparams.QuantizationMode


def get_pax_checkpoint_type() -> checkpoints.CheckpointType:
  return checkpoints.CheckpointType.GDA


logged_jax_device = False


class ServableModelParams(
    base_experiment.BaseExperiment, servable_model_params.ServableModelParams
):
  """A base class that each model config needs to implement for serving."""

  quantization_type: QuantizationType = QuantizationType.PTQ
  quant_mode: QuantizationMode = QuantizationMode.INFERENCE

  @property
  def test_mode(self) -> bool:
    return False

  @classmethod
  def get_supported_device_mesh(
      cls,
  ) -> Tuple[utils.Status, Optional[np.ndarray]]:
    global logged_jax_device
    if not logged_jax_device:
      logging.info('jax devices: %s', jax.devices())
      logged_jax_device = True

    # If mesh_shapes is a single shape, turn it into a list of a
    # singleton.
    mesh_shapes = cls.serving_mesh_shape()
    if isinstance(mesh_shapes, (tuple, list)) and all(
        isinstance(x, int) for x in mesh_shapes
    ):
      mesh_shapes = [mesh_shapes]

    errmsg = ''
    for mesh_shape in mesh_shapes:
      try:
        # If mesh_shape is supported, create_device_mesh should succeed.
        device_mesh = mesh_utils.create_device_mesh(mesh_shape)
        return utils.ok(), device_mesh
      except Exception as e:  # pylint: disable=broad-except
        errmsg += f' {e}'
    return utils.invalid_arg(f'Unsupported mesh shape:{errmsg}'), None

  @classmethod
  def check_serving_platform(cls) -> utils.Status:
    status, _ = cls.get_supported_device_mesh()
    return status

  @classmethod
  def load_ema(cls) -> bool:
    return False

  @classmethod
  @abc.abstractmethod
  def serving_mesh_shape(cls) -> Union[List[int], List[int]]:
    """Logical shape or shapes of the device mesh used for serving."""

  # TODO(zhangqiaorjc, yuanzx): Deprecated and replace with unpadded shapes
  # from checkpoints directly.
  @abc.abstractmethod
  def input_for_model_init(self) -> NestedMap:
    """Sample inputs used to initialize the model for checkpoint restore.

    Checkpoint restore requires eval_shape(model.init)(prng_key, sample_inputs).
    This method should return an input_batch that BaseModel.__call__ expects,
    typically a NestedMap of np.arrays that the restored model was originally
    trained with. Often only shape and dtype of the sample input matters for
    model.init; the batch size, seq len dimension do not matter and can be
    small for sample inputs.
    """

  def get_quant_configs(self) -> Tuple[QuantizationType, QuantizationMode]:
    return self.quantization_type, self.quant_mode  # pytype: disable=attribute-error

  def set_quantization_type(self, quantization_type: QuantizationType) -> None:
    self.quantization_type = (
        quantization_type  # pytype: disable=attribute-error
    )

  def set_quant_mode(self, mode: QuantizationMode) -> None:
    self.quant_mode = mode  # pytype: disable=attribute-error

  @classmethod
  def get_checkpoint_type(cls) -> checkpoints.CheckpointType:
    return get_pax_checkpoint_type()

  def load(
      self,
      model_key: str,
      checkpoint_path: str,
      primary_process_id: int,
      prng_key: int,
  ) -> Any:
    """Loads and returns the ServableModel."""
    model = self.create_model(primary_process_id)
    model.load(checkpoint_path, jax.random.PRNGKey(prng_key))
    return model

  @abc.abstractmethod
  def create_model(self, primary_process_id: int) -> Any:
    """Creates the model to be loaded."""


ServableModelParamsT = Type[ServableModelParams]


class ServableMethodParams(
    base_hyperparams.BaseHyperParams, servable_model_params.ServableMethodParams
):
  """A base config class for a method.

  Attributes:
    extra_inputs: Extra inputs are a dictionary of {key: default_value} pairs.
      The input for a function is a NestedMap. The `key` in `extra_inputs` can
      be one of the key in the input. The `default_value` is the default value
      for input[key].
    extra_inputs_dtypes: A dictionary of {key: np.dtype} pairs. If the dtype is
      not defined, default type for the extra input is np.float32.
    bucket_keys: keys for branch computations such as sequence length
      bucketization, this usually represents a list of possible sizes for a
      non-batch dimension, in increasing order.
    batching_wait_secs: batching wait secs for the next item. If the serving
      latency for this model is fast (<2s), do not need to set this value.
      Usually, the suggested waiting seconds for batching could set to less than
      10% device latency for the given batch size.
    cast_bfloat16_outputs: if the output tensors from device are in bfloat16,
      convert them to float32.
  """

  batch_size: Union[int, List[int]] = 1
  max_live_batches: int = 4
  extra_inputs: Optional[Dict[str, float]] = None
  extra_inputs_dtypes: Optional[Dict[str, np.dtype]] = None
  bucket_keys: Optional[List[int]] = None
  batching_wait_secs: Optional[float] = None
  polymorphic_seq_len_exclusion: Optional[List[str]] = None
  cast_bfloat16_outputs: bool = True

  def get_batch_size(self) -> Union[int, List[int]]:
    return self.batch_size

  def get_max_live_batches(self) -> int:
    return self.max_live_batches

  def get_default_extra_inputs(self) -> Optional[Dict[str, float]]:
    return self.extra_inputs

  def get_extra_inputs_dtypes(self) -> Optional[Dict[str, np.dtype]]:
    return self.extra_inputs_dtypes

  def get_batching_wait_secs(self) -> Optional[float]:
    return self.batching_wait_secs
