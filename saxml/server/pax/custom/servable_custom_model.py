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
from typing import Any, Callable, Dict, List

from paxml import checkpoint_pb2
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params
from saxml.server.pax.custom import custom_service

CheckpointType = checkpoint_pb2.CheckpointType
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap


FetchOutputFn = Callable[[NestedJTensor, NestedJTensor], NestedJTensor]
PreProcessingFn = Callable[[List[Any]], NestedNpTensor]
PostProcessingFn = Callable[[NestedNpTensor], List[Any]]


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
  """
  model_fn_name: str = ''
  dummy_input_sample: Any = None
  fetch_output_fn: FetchOutputFn = None
  pre_process_fn: PreProcessingFn = None
  post_process_fn: PostProcessingFn = None


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
    super().__init__(
        model,
        method_hparams.model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        method_hparams.dummy_input_sample,
        exportable=False)

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
    return self._method_hparams.pre_process_fn(raw_inputs)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    return self._method_hparams.post_process_fn(compute_outputs)


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
