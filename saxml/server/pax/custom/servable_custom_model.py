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
"""Wraps a model with custom service APIs."""

import abc
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from paxml import checkpoint_pb2
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params
from saxml.server.services import custom_service

CheckpointType = checkpoint_pb2.CheckpointType
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap


FetchOutputFn = Callable[[NestedJTensor, NestedJTensor], NestedJTensor]


class PreProcessingFn(Protocol):
  """Preprocessor function: (inputs, optional_method_state) -> numpy arrays.

  Without CreateInitStateFn: (inputs,) -> numpy input arrays
  With CreateInitStateFn: (inputs, method_state) -> numpy input arrays.
  """

  def __call__(
      self, raw_inputs: List[Any], method_state: Optional[Any] = None
  ) -> NestedNpTensor:
    ...


class PostProcessingFn(Protocol):
  """Postprocessor function: (numpy arrays, optional_method_state) -> outputs.

  Without CreateInitStateFn: (numpy arrays,) -> outputs
  With CreateInitStateFn: (numpy arrays, method_states) -> outputs
  """

  def __call__(
      self, raw_inputs: NestedNpTensor, method_state: Optional[Any] = None
  ) -> List[Any]:
    ...


# Optional creator of initial method state.
CreateInitStateFn = Callable[['ServableCustomMethod'], Any]

# Custom way of calling the model function (before fetch_output). Optional.
# Signature is
#  (model, inputs, mdl_vars, prng_key, method_state) -> (outputs, updated_vars).
CallModelFn = Callable[
    [base_model.BaseModel, NestedJTensor, NestedJTensor, PRNGKey, Any],
    Tuple[NestedJTensor, NestedJTensor]]


class CustomMethodName:
  CUSTOM = 'custom'


class CustomCallHParams(servable_model_params.ServableMethodParams):
  """HParameters for LM score method.

  Attributes:
    model_fn_name: function name from the model to call.
    dummy_input_sample: dummpy input sample for the custom call.
    fetch_output_fn: A callable fetch_output_fn for the custom call.
    pre_process_fn: A callable pre_process_fn for the custom call.
    post_process_fn: A callable post_process_fn for the custom call.
    create_init_state_fn: A callable to initialize custom method state.
    call_model_fn: Optional custom way of calling the model function.
  """
  model_fn_name: str = ''
  dummy_input_sample: Any = None
  fetch_output_fn: Optional[FetchOutputFn] = None
  pre_process_fn: Optional[PreProcessingFn] = None
  post_process_fn: Optional[PostProcessingFn] = None
  create_init_state_fn: Optional[CreateInitStateFn] = None
  call_model_fn: Optional[CallModelFn] = None


class ServableCustomModelParams(
    servable_model_params.ServableModelParams, metaclass=abc.ABCMeta):
  """A base class that each Custom model config needs to implement for serving.
  """

  def methods(self) -> Dict[str, CustomCallHParams]:
    return {}

  def create_model(self, primary_process_id: int) -> 'ServableCustomModel':
    return ServableCustomModel(self, primary_process_id,
                               self.get_checkpoint_type())


class ServableCustomMethod(servable_model.ServableMethod):
  """Implements custom method."""

  def __init__(self, model: base_model.BaseModel,
               model_state: servable_model.ServableModelState,
               method_hparams: CustomCallHParams, prng_key: PRNGKey,
               model_config: Any):
    self._model_config = model_config
    self._method_hparams = method_hparams
    self._state = None
    assert method_hparams.fetch_output_fn is not None
    assert method_hparams.pre_process_fn is not None
    assert method_hparams.post_process_fn is not None
    super().__init__(
        model,
        method_hparams.model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        method_hparams.dummy_input_sample,
        exportable=False,
        load=False,
    )
    if method_hparams.create_init_state_fn is not None:
      self._state = method_hparams.create_init_state_fn(self)
    self.load()

  @classmethod
  def service_id(cls) -> str:
    return custom_service.SERVICE_ID

  def fetch_output(self, model_fn_outputs: NestedJTensor,
                   model_fn_inputs: NestedJTensor) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    return self._method_hparams.fetch_output_fn(model_fn_outputs,
                                                model_fn_inputs)

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    if self._state is not None:
      return self._method_hparams.pre_process_fn(raw_inputs, self._state)
    return self._method_hparams.pre_process_fn(raw_inputs)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    if self._state is not None:
      return self._method_hparams.post_process_fn(compute_outputs, self._state)
    return self._method_hparams.post_process_fn(compute_outputs)

  def call_model_function(self, inputs: NestedJTensor, mdl_vars: NestedJTensor,
                          prng_key: PRNGKey) -> NestedJTensor:
    if self._method_hparams.call_model_fn is not None:
      return self._method_hparams.call_model_fn(self.pax_model, inputs,
                                                mdl_vars, prng_key, self._state)
    return super().call_model_function(inputs, mdl_vars, prng_key)


class ServableCustomModel(servable_model.ServableModel):
  """Represents an implementation for the Custom service, backed by a model.

  This class is responsible for model loading, batch padding, etc.
  """

  def init_method(self, method: str, model: base_model.BaseModel,
                  model_state: servable_model.ServableModelState,
                  method_params: servable_model_params.ServableMethodParams,
                  prng_key: PRNGKey) -> servable_model.ServableMethod:
    assert isinstance(method_params, CustomCallHParams)
    return ServableCustomMethod(model, model_state, method_params, prng_key,
                                self._model_config)
