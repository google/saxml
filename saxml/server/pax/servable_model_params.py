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


class ServableModelParams(base_experiment.BaseExperiment,
                          servable_model_params.ServableModelParams):
  """A base class that each model config needs to implement for serving."""

  quantization_type: QuantizationType = QuantizationType.PTQ
  quant_mode: QuantizationMode = QuantizationMode.INFERENCE

  @classmethod
  def check_serving_platform(cls) -> utils.Status:
    global logged_jax_device
    if not logged_jax_device:
      logging.info('jax devices: %s', jax.devices())
      logged_jax_device = True
    mesh_shape = cls.serving_mesh_shape()
    try:
      # If mesh_shape is supported, create_device_mesh should succeed.
      mesh_utils.create_device_mesh(mesh_shape)
    except Exception as e:  # pylint: disable=broad-except
      return utils.invalid_arg(f'Unsupported mesh shape: {e}')
    return utils.ok()

  @classmethod
  @abc.abstractmethod
  def serving_mesh_shape(cls) -> List[int]:
    """Logical shape of the device mesh used for serving."""

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
    self.quantization_type = quantization_type  # pytype: disable=attribute-error

  def set_quant_mode(self, mode: QuantizationMode) -> None:
    self.quant_mode = mode  # pytype: disable=attribute-error

  @classmethod
  def get_checkpoint_type(cls) -> checkpoints.CheckpointType:
    return get_pax_checkpoint_type()

  def load(self, model_key: str, checkpoint_path: str, primary_process_id: int,
           prng_key: int) -> Any:
    """Loads and returns the ServableModel."""
    model = self.create_model(primary_process_id)
    model.load(checkpoint_path, jax.random.PRNGKey(prng_key))
    return model

  @abc.abstractmethod
  def create_model(self, primary_process_id: int) -> Any:
    """Creates the model to be loaded."""


ServableModelParamsT = Type[ServableModelParams]


class ServableMethodParams(base_hyperparams.BaseHyperParams,
                           servable_model_params.ServableMethodParams):
  """A base config class for a method.

  Attributes:
    bucket_keys: keys for branch computations such as sequence length
      bucketization, this usually represents a list of possible sizes for a
      non-batch dimension, in increasing order.
    batching_wait_secs: batching wait secs for the next item. If the serving
      latency for this model is fast (<2s), do not need to set this value.
      Usually, the suggested waiting seconds for batching could set to less than
      10% device latency for the given batch size.
  """
  batch_size: Union[int, List[int]] = 1
  max_live_batches: int = 4
  extra_inputs: Optional[Dict[str, float]] = None
  bucket_keys: Optional[List[int]] = None
  batching_wait_secs: Optional[float] = None

  def get_batch_size(self) -> Union[int, List[int]]:
    return self.batch_size

  def get_max_live_batches(self) -> int:
    return self.max_live_batches

  def get_default_extra_inputs(self) -> Optional[Dict[str, float]]:
    return self.extra_inputs

  def get_batching_wait_secs(self) -> Optional[float]:
    return self.batching_wait_secs
