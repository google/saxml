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
from typing import Any, Callable, Optional, List, Dict

import jax
from paxml import checkpoint_pb2
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params

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


@servable_model_params.create_service_id_for_model_type
class ServableCustomModelParams(
    servable_model_params.ServableModelParams, metaclass=abc.ABCMeta):
  """A base class that each Custom model config needs to implement for serving.
  """

  def custom_calls(self) -> Optional[Dict[str, CustomCallHParams]]:
    """Returns the params for the custom method."""
    return None

  def load(self, model_key: str, checkpoint_path: str, primary_process_id: int,
           prng_key: int) -> 'ServableCustomModel':
    """Loads and initializes the model."""
    model = ServableCustomModel(self, primary_process_id,
                                self.get_checkpoint_type())
    model.load(checkpoint_path, jax.random.PRNGKey(prng_key))
    return model


class ServableCustomMethod(servable_model.ServableMethod):
  """Implements custom method."""

  def __init__(self, model: base_model.BaseModel,
               model_state: servable_model.ServableModelState,
               method_hparams: CustomCallHParams, prng_key: PRNGKey,
               model_config: Any):
    self._model_config = model_config
    self._method_hprams = method_hparams
    super().__init__(
        model,
        method_hparams.model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        method_hparams.dummy_input_sample,
        exportable=False)

  def fetch_output(self, model_fn_outputs: NestedJTensor,
                   model_fn_inputs: NestedJTensor) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    return self._method_hprams.fetch_output_fn(model_fn_outputs,
                                               model_fn_inputs)

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    return self._method_hprams.pre_process_fn(raw_inputs)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    return self._method_hprams.post_process_fn(compute_outputs)


class ServableCustomModel(servable_model.ServableModel):
  """Represents an implementation for the Custom service, backed by a model.

  This class is responsible for model loading, batch padding, etc.
  """

  def __init__(self,
               model_config: ServableCustomModelParams,
               primary_process_id: int,
               ckpt_type: CheckpointType,
               test_mode: bool = False):
    self._model_config = model_config
    self._custom_call_params = model_config.custom_calls()

    super().__init__(model_config, primary_process_id,
                     self._custom_call_params.keys(), ckpt_type, test_mode)

  def init_method(self, method: str, model: base_model.BaseModel,
                  model_state: servable_model.ServableModelState,
                  prng_key: PRNGKey) -> servable_model.ServableMethod:
    if method not in self._custom_call_params:
      raise NotImplementedError(f'method {method} not implemented')
    return ServableCustomMethod(model, model_state,
                                self._custom_call_params.get(method), prng_key,
                                self._model_config)
