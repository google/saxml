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
"""Base classes that define sax Servable{Method,Model} using pytorch."""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from saxml.server import servable_model
from saxml.server import servable_model_params

import torch
from torch import nn
import torch.utils._pytree as pytree

DeviceTensors = Any
ExtraInput = servable_model.ExtraInput
HostTensors = Any
InputShapeInfo = servable_model.InputShapeInfo


class ServableMethod(servable_model.ServableMethod):
  """Base class for a method that calls a torch model."""

  def __init__(
      self,
      model: nn.Module,
      method_params: ServableMethodParams,
      device: str = "cuda",
  ):
    """Constructor.

    Args:
        model: a torch module.
        method_params: parameters of this method.
        device: either "cpu" or "cuda", the device to run the model on.
    """
    super().__init__(method_params)
    self._model = model
    if device not in ["cpu", "cuda"]:
      raise ValueError(f"Invalid device: {device}!")
    self._device = device
    for k, v in method_params.method_attrs.items():
      setattr(self, k, v)

  def unload(self) -> None:
    """Clears references held by this model."""
    del self._model

  def _maybe_to_device(self, x):
    if isinstance(x, torch.Tensor):
      return x.to(device=self._device)
    return x

  def _maybe_to_host(self, x):
    if isinstance(x, torch.Tensor):
      return x.cpu().numpy()
    return x

  def input_to_device(
      self, one_core_inputs: HostTensors, unpadded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Transfers input data to device."""
    return pytree.tree_map(self._maybe_to_device, one_core_inputs)

  def device_compute(
      self, input_batch: DeviceTensors, unpadded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes the device computation."""
    with torch.no_grad():
      return self._model(input_batch)

  def output_to_host(
      self, output_tensors: DeviceTensors, unpadded_batch_size: int
  ) -> HostTensors:
    """Fetches device outputs to host. Removes batch padding."""
    return pytree.tree_map(self._maybe_to_host, output_tensors)

  def remove_batch_padding(
      self, host_tensors: HostTensors, unpadded_batch_size: int
  ) -> HostTensors:
    """Removes batch padding."""
    return host_tensors

  def update_extra_inputs(
      self,
      input_batch: HostTensors,
      batch_size: int,
      extra_inputs: Optional[List[ExtraInput]] = None,
  ) -> HostTensors:
    if extra_inputs and extra_inputs[0]:
      raise ValueError(
          f"extra_inputs is not supported by this method! Got {extra_inputs}."
      )
    return input_batch

  @property
  def streamable(self) -> bool:
    return False

  def device_compute_with_dummy_data(
      self, unpadded_shape: InputShapeInfo
  ) -> DeviceTensors:
    raise NotImplementedError


@dataclasses.dataclass
class ServableMethodParams(servable_model_params.ServableMethodParams):
  """A base param class for a torch method.

  Attributes:
    method_cls: the class name to construct the ServableMethod instance.
    method_attrs: extra attributes to assign to the ServableMethod.
  """

  method_cls: Type[ServableMethod]
  method_attrs: Dict[str, Any] = dataclasses.field(default_factory=dict)

  batch_size: Union[int, List[int]] = 1
  max_live_batches: int = 4
  extra_inputs: Optional[Dict[str, float]] = None
  extra_inputs_dtypes: Optional[Dict[str, np.dtype]] = None

  def get_batch_size(self) -> Union[int, List[int]]:
    return self.batch_size

  def get_max_live_batches(self) -> int:
    return self.max_live_batches

  def get_default_extra_inputs(self) -> Optional[Dict[str, float]]:
    return self.extra_inputs

  def get_extra_inputs_dtypes(self) -> Optional[Dict[str, np.dtype]]:
    return self.extra_inputs_dtypes

  def get_batching_wait_secs(self) -> Optional[float]:
    return None


class ServableModel(servable_model.ServableModel):
  """A generic ServableModel for pytorch models."""

  def __init__(
      self,
      model: nn.Module,
      method_params: Dict[str, ServableMethodParams],
      device: str = "cuda",
  ):
    """Constructor.

    Args:
        model: a torch module.
        method_params: parameters of each method in this model.
        device: either "cpu" or "cuda", the device to run the model on.
    """
    super().__init__()
    model.eval()  # Use evaluation mode (for dropout, etc.).
    for k, v in method_params.items():
      self.add_method(k, v.method_cls(model, v, device))  # pytype: disable=not-instantiable # b/183649930

  def supports_dummy_compute_on_primary(self) -> bool:
    return False
