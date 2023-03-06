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
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple
import numpy as np

from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params
from saxml.server.services import custom_service

import tensorflow as tf

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
NestedTfTensor = pytypes.Nested[tf.Tensor]
NestedTfTensorSpec = pytypes.Nested[tf.TensorSpec]
NestedTfTrackable = pytypes.Nested[
    tf.saved_model.experimental.TrackableResource
]

NestedPolyShape = pytypes.Nested[str]
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap
InputShapeInfo = servable_model.InputShapeInfo
HostTensors = servable_model.HostTensors
ShapesAndDtypes = servable_model.ShapesAndDtypes


FetchOutputFn = Callable[[NestedJTensor, NestedJTensor], NestedJTensor]


class PreProcessingFn(Protocol):
  """Preprocessor function: (inputs, optional_method_state) -> numpy arrays.

  Without CreateInitStateFn: (inputs,) -> numpy input arrays
  With CreateInitStateFn: (inputs, method_state) -> numpy input arrays.
  """

  def __call__(
      self, raw_inputs: List[bytes], method_state: Optional[Any] = None
  ) -> NestedNpTensor:
    ...


class PostProcessingFn(Protocol):
  """Postprocessor function: (numpy arrays, optional_method_state) -> outputs.

  Without CreateInitStateFn: (numpy arrays,) -> outputs
  With CreateInitStateFn: (numpy arrays, method_states) -> outputs
  """

  def __call__(
      self, raw_inputs: NestedNpTensor, method_state: Optional[Any] = None
  ) -> List[bytes]:
    ...


# Optional creator of initial method state.
CreateInitStateFn = Callable[['ServableCustomMethod'], Any]

# Custom way of calling the model function (before fetch_output). Optional.
# Signature is
#  (model, inputs, mdl_vars, prng_key, method_state) -> (outputs, updated_vars).
CallModelFn = Callable[
    [base_model.BaseModel, NestedJTensor, NestedJTensor, PRNGKey, Any],
    Tuple[NestedJTensor, NestedJTensor],
]

TfProcessingFn = Callable[..., NestedTfTensor]
TfInputSignatureGenerator = Callable[[Optional[int]], NestedTfTensorSpec]
GetSortedInputShapesFn = Callable[
    [Sequence[int], Sequence[int]], List[InputShapeInfo]
]
HandleHostInputWithInputShapeFn = Callable[[HostTensors, Any], HostTensors]
GetPaddedInputShapeFn = Callable[[Any], Any]
GetUnpaddedInputShapeFn = Callable[[int, HostTensors], Any]
DeserializeInputShapeFn = Callable[[str], Any]
ResizeHostArrayFn = Callable[[np.ndarray, ShapesAndDtypes, Any], HostTensors]


class CustomMethodName:
  CUSTOM = 'custom'


class CustomCallHParams(servable_model_params.ServableMethodParams):
  """HParameters for a custom call method.

  Attributes:
    model_fn_name: function name from the model to call.
    dummy_input_sample: dummpy input sample for the custom call.
    fetch_output_fn: A callable fetch_output_fn for the custom call.
    pre_process_fn: A callable pre_process_fn for the custom call.
    post_process_fn: A callable post_process_fn for the custom call.
    create_init_state_fn: A callable to initialize custom method state.
    call_model_fn: Optional custom way of calling the model function.
    tf_pre_process_fn: TF pre-process function for model export.
    tf_post_process_fn: TF pre-process function for model export.
    tf_extra_trackables: TF tracable resources for model export.
    tf_input_signature: Input signature of `tf_pre_process_fn`.
    model_fn_input_polymorphic_shape: jax2tf polymorphic shape for the input
      tensors of `call_model_fn`.
    get_sorted_input_shapes_fn: Optional function to get sorted input shapes.
    handle_host_input_with_input_shape_fn: Optional function to handle host
      tensors with a given input shape.
    get_padded_input_shape_fn: Optional function to get padded input shape.
    get_unpadded_input_shape_fn: Optional function to get unpadded input shape.
    deserialize_input_shape_fn: Optional function to deserialize InputShapeInfo.
    resize_host_array_fn: Optional function to resize host array.
  """

  model_fn_name: str = ''
  dummy_input_sample: Any = None
  fetch_output_fn: Optional[FetchOutputFn] = None
  pre_process_fn: Optional[PreProcessingFn] = None
  post_process_fn: Optional[PostProcessingFn] = None
  create_init_state_fn: Optional[CreateInitStateFn] = None
  call_model_fn: Optional[CallModelFn] = None
  # Fields for SavedModel export only.
  exportable: bool = False
  tf_pre_process_fn: Optional[TfProcessingFn] = None
  tf_post_process_fn: Optional[TfProcessingFn] = None
  tf_extra_trackables: Optional[NestedTfTrackable] = None
  tf_input_signature: Optional[TfInputSignatureGenerator] = None
  model_fn_input_polymorphic_shape: Optional[NestedPolyShape] = None
  get_sorted_input_shapes_fn: Optional[GetSortedInputShapesFn] = None
  handle_host_input_with_input_shape_fn: Optional[
      HandleHostInputWithInputShapeFn
  ] = None
  get_padded_input_shape_fn: Optional[GetPaddedInputShapeFn] = None
  get_unpadded_input_shape_fn: Optional[GetUnpaddedInputShapeFn] = None
  deserialize_input_shape_fn: Optional[DeserializeInputShapeFn] = None
  resize_host_array_fn: Optional[ResizeHostArrayFn] = None


class ServableCustomModelParams(
    servable_model_params.ServableModelParams, metaclass=abc.ABCMeta
):
  """A base class that each Custom model config needs to implement for serving."""

  def methods(self) -> Dict[str, CustomCallHParams]:
    return {}

  def create_model(self, primary_process_id: int) -> 'ServableCustomModel':
    return ServableCustomModel(
        self,
        primary_process_id,
        self.get_checkpoint_type(),
        test_mode=self.test_mode,
    )


class ServableCustomMethod(servable_model.ServableMethod):
  """Implements custom method."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      method_hparams: CustomCallHParams,
      prng_key: PRNGKey,
      model_config: Any,
  ):
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
        exportable=method_hparams.exportable,
        load=False,
    )
    if method_hparams.create_init_state_fn is not None:
      self._state = method_hparams.create_init_state_fn(self)
    self.load()

  @classmethod
  def service_id(cls) -> str:
    return custom_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    return self._method_hparams.fetch_output_fn(
        model_fn_outputs, model_fn_inputs
    )

  def pre_processing(self, raw_inputs: List[bytes]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    if self._state is not None:
      return self._method_hparams.pre_process_fn(raw_inputs, self._state)
    return self._method_hparams.pre_process_fn(raw_inputs)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[bytes]:
    """Postprocesses the output numpy arrays to final host output."""
    if self._state is not None:
      return self._method_hparams.post_process_fn(compute_outputs, self._state)
    return self._method_hparams.post_process_fn(compute_outputs)

  def call_model_function(
      self, inputs: NestedJTensor, mdl_vars: NestedJTensor, prng_key: PRNGKey
  ) -> NestedJTensor:
    if self._method_hparams.call_model_fn is not None:
      return self._method_hparams.call_model_fn(
          self.pax_model, inputs, mdl_vars, prng_key, self._state
      )
    return super().call_model_function(inputs, mdl_vars, prng_key)

  def tf_pre_processing(self, *args: NestedTfTensor) -> NestedTfTensor:
    if self._state is not None:
      raise NotImplementedError(
          'Custom call with extra state is not exportable.'
      )
    if self._method_hparams.tf_pre_process_fn is None:
      raise ValueError('CustomCallHParams.tf_pre_process_fn not set.')
    return self._method_hparams.tf_pre_process_fn(*args)

  def tf_post_processing(
      self, compute_outputs: NestedTfTensor
  ) -> NestedTfTensor:
    if self._state is not None:
      raise NotImplementedError(
          'Custom call with extra state is not exportable.'
      )
    if self._method_hparams.tf_post_process_fn is None:
      raise ValueError('CustomCallHParams.tf_post_process_fn not set.')
    return self._method_hparams.tf_post_process_fn(compute_outputs)

  def input_signature(self, batch_size) -> Optional[NestedTfTensorSpec]:
    return self._method_hparams.tf_input_signature(batch_size)

  @property
  def extra_trackables(self) -> Optional[NestedTfTrackable]:
    return self._method_hparams.tf_extra_trackables

  @property
  def model_fn_input_polymorphic_shape(self) -> Optional[NestedPolyShape]:
    return self._method_hparams.model_fn_input_polymorphic_shape

  def get_sorted_input_shapes(self) -> List[InputShapeInfo]:
    get_sorted_input_shapes_fn = self._method_hparams.get_sorted_input_shapes_fn
    if get_sorted_input_shapes_fn:
      return get_sorted_input_shapes_fn(
          self._sorted_batch_sizes, self._bucket_keys
      )
    return super().get_sorted_input_shapes()

  def get_dummy_inputs(self, input_shape: InputShapeInfo) -> HostTensors:
    """Returns host tensors with dummy data at a batch size."""
    batched_input = self.pre_processing(
        [self._dummy_input_sample] * input_shape.batch_size
    )
    handle_host_input_with_input_shape_fn = (
        self._method_hparams.handle_host_input_with_input_shape_fn
    )
    if handle_host_input_with_input_shape_fn:
      return handle_host_input_with_input_shape_fn(batched_input, input_shape)
    return batched_input

  def get_padded_input_shape(
      self, unpadded_shape: InputShapeInfo
  ) -> InputShapeInfo:
    """Get padded input shape."""
    # Gets padded batch size.
    if self._method_hparams.get_padded_input_shape_fn:
      return self._method_hparams.get_padded_input_shape_fn(unpadded_shape)
    return super().get_padded_input_shape(unpadded_shape)

  def get_unpadded_shape(
      self, unpadded_batch_size, inputs: HostTensors
  ) -> InputShapeInfo:
    """Get unpadded input shape."""
    if self._method_hparams.get_unpadded_input_shape_fn:
      return self._method_hparams.get_unpadded_input_shape_fn(
          unpadded_batch_size, inputs
      )
    return super().get_unpadded_shape(unpadded_batch_size, inputs)

  def deserialize_input_shape(self, unpadded_shape_str: str) -> InputShapeInfo:
    """Deserialize input shape from a str."""
    if self._method_hparams.deserialize_input_shape_fn:
      return self._method_hparams.deserialize_input_shape_fn(unpadded_shape_str)
    return super().deserialize_input_shape(unpadded_shape_str)

  def resize_host_array(
      self,
      x: np.ndarray,
      global_input_shape_dtype: ShapesAndDtypes,
      unpadded_input_shape: InputShapeInfo,
  ) -> HostTensors:
    """Resizes x to the desired shape.

    Args:
      x: Host tensor.
      global_input_shape_dtype: Global input shape and dtype for this tensor.
      unpadded_input_shape: Unpadded input shape.

    Returns:
      host array after padding or slice of x.
    """
    if self._method_hparams.resize_host_array_fn:
      x = self._method_hparams.resize_host_array_fn(
          x, global_input_shape_dtype, unpadded_input_shape
      )
    # Let the parent class handle the batch dim.
    return super().resize_host_array(
        x, global_input_shape_dtype, unpadded_input_shape
    )


class ServableCustomModel(servable_model.ServableModel):
  """Represents an implementation for the Custom service, backed by a model.

  This class is responsible for model loading, batch padding, etc.
  """

  def init_method(
      self,
      method: str,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      method_params: servable_model_params.ServableMethodParams,
      prng_key: PRNGKey,
  ) -> servable_model.ServableMethod:
    assert isinstance(method_params, CustomCallHParams)
    return ServableCustomMethod(
        model, model_state, method_params, prng_key, self._model_config
    )
