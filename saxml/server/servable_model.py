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
"""Wraps a model with service APIs."""

import abc
import dataclasses
import json
import queue
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from saxml.server import servable_model_params

HostTensors = Any
DeviceTensors = Any
ExtraInput = servable_model_params.ExtraInputs


@dataclasses.dataclass(eq=True, frozen=True)
class InputShapeInfo:
  """Input shape information."""

  def __str__(self):
    return json.dumps(dataclasses.asdict(self))

  batch_size: int = -1


class ServableMethod(abc.ABC):
  """Base class for method implementation and its pre- and post-processing.

  Subclasses need to implement the abstract methods.
  """

  def __init__(self, method_params: servable_model_params.ServableMethodParams):
    self._sorted_batch_sizes = method_params.get_batch_size()
    if isinstance(self._sorted_batch_sizes, int):
      self._sorted_batch_sizes = [self._sorted_batch_sizes]
    assert isinstance(self._sorted_batch_sizes, list)
    self._sorted_batch_sizes = sorted(self._sorted_batch_sizes)
    self._max_live_batches = method_params.get_max_live_batches()
    self._batching_wait_secs = method_params.get_batching_wait_secs()
    self._extra_inputs = method_params.get_default_extra_inputs()
    self._extra_inputs_dtypes = method_params.get_extra_inputs_dtypes()
    # If an element is None, it marks the end of the stream.
    self._stream_output_queue: queue.SimpleQueue[Optional[HostTensors]] = (
        queue.SimpleQueue()
    )

  @classmethod
  @abc.abstractmethod
  def service_id(cls) -> str:
    """Unique ID for the model service that supports this model."""

  @property
  def sorted_batch_sizes(self) -> List[int]:
    """A list of sorted supported (ascending order) batch sizes."""
    return self._sorted_batch_sizes

  @property
  def default_extra_inputs(self) -> Optional[ExtraInput]:
    """Default extra inputs for requests that do not specify them."""
    return self._extra_inputs

  @property
  def extra_inputs_dtypes(self) -> Optional[Dict[str, np.dtype]]:
    """Extra input dtypes for extra_input."""
    return self._extra_inputs_dtypes

  @abc.abstractmethod
  def unload(self) -> None:
    """Clears references held by this method."""

  @abc.abstractmethod
  def input_to_device(
      self,
      one_core_inputs: HostTensors,
      unpadded_shape: InputShapeInfo,
      padded_shape: InputShapeInfo,
  ) -> DeviceTensors:
    """Transfers host inputs to device. Pads incomplete shapes."""

  @abc.abstractmethod
  def output_to_host(
      self, output_tensors: DeviceTensors, unpadded_batch_size: int
  ) -> HostTensors:
    """Transfers device outputs to host. Removes batch padding."""

  @abc.abstractmethod
  def remove_batch_padding(
      self, host_tensors: HostTensors, unpadded_batch_size: int
  ) -> HostTensors:
    """Removes batch padding."""

  @property
  def batch_size(self) -> int:
    return self.sorted_batch_sizes[-1] if self.sorted_batch_sizes else 1

  @property
  def max_live_batches(self) -> int:
    """Maximum number of live batches in the server for this method."""
    return self._max_live_batches

  @property
  def batching_wait_secs(self) -> Optional[float]:
    """Batching waiting secs in the server for this method."""
    return self._batching_wait_secs

  @abc.abstractmethod
  def pre_processing(self, raw_inputs: List[Any]) -> HostTensors:
    """Preprocesses an unpadded batch of data into host arrays."""

  def get_extra_inputs_from_request_inputs(self, request: Any) -> ExtraInput:
    """Gets extra_inputs from request_inputs."""
    extra_inputs: ExtraInput = {}
    if hasattr(request, 'extra_inputs') and request.extra_inputs:
      # Extract Scalars.
      for k, v in dict(request.extra_inputs.items).items():
        extra_inputs[k] = v
      # Extract Tensors (1d list of floats).
      # (Reshaping is delegated to the model.)
      for k, v in dict(request.extra_inputs.tensors).items():
        extra_inputs[k] = list(v.values)
      # Extract Strings.
      for k, v in dict(request.extra_inputs.strings).items():
        extra_inputs[k] = v
    return extra_inputs

  @abc.abstractmethod
  def update_extra_inputs(
      self,
      input_batch: HostTensors,
      batch_size: int,
      extra_inputs: Optional[List[ExtraInput]] = None,
  ) -> HostTensors:
    """Updates mutable input keys to input batch.

    Users would like to update some input keys for the input batch through
    RPC requests. This function updates the per example mutable input value in
    the input batch from extra_inputs.

    Args:
      input_batch: Nested host arrays for device computation function input. It
        could be mutated.
      batch_size: Batch size of the input_batch.
      extra_inputs: Optional list of dictionary for {input_key: scalar_value}
        for each example. The keys in different elements of list could be
        different. The element in the list could be an empty dictionary. When it
        is None, when fill extra_inputs with self.default_extra_inputs.

    Returns:
      Updated input batch.
    """

  @abc.abstractmethod
  def post_processing(self, compute_outputs: HostTensors) -> List[Any]:
    """Postprocesses the output host arrays to final host output.

    Args:
      compute_outputs: Output host tensors from ServableMethod.device_compute.

    Returns:
      A list of service-specific outputs per RPC. Each entry in the list is
      passed to ModelService.FillRPCResponse to populate the response for the
      RPC request.
    """

  def post_processing_stream(
      self,
      compute_outputs: Optional[HostTensors] = None,
      stream_state: Optional[Any] = None,
  ) -> Tuple[List[Any], Optional[Any]]:
    """Postprocesses one streaming output.

    Args:
      compute_outputs: Tensors streamed out of the device. If None, finalize
        streaming using stream_state.
      stream_state: The stream_state returned by the previous call. If missing,
        initialize streaming using compute_outputs as the first input.

    Returns:
      host_outputs: A list of service-specific outputs per RPC. Each entry in
        the list is passed to ModelService.FillRPCResponse to populate the
        response for the RPC request.
      stream_state: Host-side post-processing state to pass to the next call to
        post_processing_stream for this batch of streaming responses.
    """
    raise NotImplementedError('post_processing_stream not implemented')

  @abc.abstractmethod
  def device_compute(
      self, input_batch: DeviceTensors, padded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes the device computation."""

  @abc.abstractmethod
  def device_compute_with_dummy_data(
      self, padded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes device computation with dummy inputs."""
    # This is needed for multi-host SPMD programs to execute in sync.

  @property
  @abc.abstractmethod
  def streamable_output(self) -> bool:
    """Whether this method supports output streaming."""

  def dequeue_stream_output(self) -> Optional[HostTensors]:
    """Dequeues streamed output tensors, or None if done. Blocking if empty."""
    return self._stream_output_queue.get()

  def enqueue_stream_output(self, stream_outputs: HostTensors) -> None:
    """Enqueues streamed output tensors."""
    self._stream_output_queue.put(stream_outputs)

  def mark_stream_output_done(self) -> None:
    """Marks the streamed output as done."""
    self._stream_output_queue.put(None)

  def deserialize_input_shape(self, unpadded_shape_str: str) -> InputShapeInfo:
    """Deserialize input shape from a str."""
    unpadded_shape_dict = json.loads(unpadded_shape_str)
    return InputShapeInfo(batch_size=unpadded_shape_dict['batch_size'])

  def get_padded_input_shape(
      self, unpadded_shape: InputShapeInfo
  ) -> InputShapeInfo:
    """Get padded input shape.

    Args:
      unpadded_shape: Unpadded shape information contains batch size or sequence
        length.

    Returns:
      Padded input shape.
    Raises:
      ValueError if unpadded batch size or sequence length too large.
    """
    for bs in self.sorted_batch_sizes:
      if bs >= unpadded_shape.batch_size:
        return InputShapeInfo(bs)

    raise ValueError(
        f'Batch size larger than maximum: {unpadded_shape.batch_size} vs '
        f'{self.batch_size}'
    )

  def get_unpadded_shape(
      self, unpadded_batch_size, inputs: HostTensors
  ) -> InputShapeInfo:
    del inputs
    return InputShapeInfo(unpadded_batch_size)

  def compute(
      self,
      raw_inputs: List[Any],
      extra_inputs: Optional[List[ExtraInput]] = None,
  ) -> List[Any]:
    """Runs pre-processing, device compute, and post-processing on raw inputs.

    This is a convenience method that should only be used in non-performance-
    critical code, such as tests, or to demonstrate the typical workflow of a
    method. Performance critical code such as ModelServicesRunner calls
    pre_processing, device_compute, and post_processing directly and
    individually in a pipelined fashion.

    Args:
      raw_inputs: A list of raw inputs to be given to the pre-processor.
      extra_inputs: An optional list of per-example extra inputs.

    Returns:
      Post-processed outputs.
    """
    assert not self.streamable_output
    unpadded_batch_size = len(raw_inputs)
    if unpadded_batch_size > self.batch_size:
      raise ValueError(
          'Input to compute() input has a larger batch size'
          f' ({unpadded_batch_size}) than maximum ({self.batch_size})'
      )
    if extra_inputs is not None and len(extra_inputs) != unpadded_batch_size:
      raise ValueError(
          f'Extra inputs ({extra_inputs}) must have the same length as that of'
          f' input to compute() ({unpadded_batch_size})'
      )

    inputs = self.pre_processing(raw_inputs)
    inputs = self.update_extra_inputs(inputs, unpadded_batch_size, extra_inputs)
    unpadded_shape = self.get_unpadded_shape(unpadded_batch_size, inputs)
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    inputs = self.input_to_device(inputs, unpadded_shape, padded_shape)
    outputs = self.device_compute(inputs, padded_shape)
    outputs = self.output_to_host(outputs, unpadded_shape.batch_size)
    return self.post_processing(outputs)

  @property
  def continuous_batching(self) -> bool:
    """Returns if the model method supports continuous batching."""
    return False

  @property
  def num_cache_slots(self) -> int:
    raise NotImplementedError('num_cache_slots not implemented')

  @property
  def max_decode_steps(self) -> int:
    raise NotImplementedError('max_decode_steps not implemented')

  def prefill(
      self, inputs: DeviceTensors
  ) -> tuple[DeviceTensors, DeviceTensors, DeviceTensors]:
    """Prefills the KV cache with the input sequence.

    Args:
      inputs: An opaque object that represents a sequence (`prompt`) [B, T] to
        run prefill on.

    Returns:
      scores: Log probability [B] of sampled next tokens.
      token: Next token [B] of the prompt, sampled by model's sampler.
      cache: Prefilled KV state.
    """
    raise NotImplementedError('prefill not implemented')

  def prefill_with_dummy(
      self,
  ) -> tuple[DeviceTensors, DeviceTensors, DeviceTensors]:
    """Prefills the KV cache with a dummy sequence. Used by secondary hosts.

    Returns:
      scores: Log probability [B] of sampled next tokens.
      token: Next token [B] of the prompt, sampled by model's default sampler.
      cache: Prefilled KV cache.
    """
    raise NotImplementedError('prefill_with_dummy not implemented')

  def insert(self, prefix_state: DeviceTensors, slot: int) -> None:
    """Insert the prefix state into the specified slot of the target state.

    The target state is an internal state managed by the ServableMethod object.

    Args:
      prefix_state: the prefix kv state generated by prefill.
      slot: index of the cache slot to insert into.
    """
    raise NotImplementedError('insert_cache not implemented')

  def generate(self) -> tuple[DeviceTensors, DeviceTensors, DeviceTensors]:
    """Given previous tokens and the KV state (managed internally), generate the next batch of tokens.

    Returns:
      scores: Log probability [B] of sampled next tokens.
      new_tokens: a batch of new tokens [B] sampled by model's sampler.
      done: a batch of booleans [B] indicating whether the sampled token is EOS.
    """
    raise NotImplementedError('generate not implemented')

  def detokenize(self, tokens: HostTensors) -> List[str]:
    """Detokenize a batch of sequences into a list of strings."""
    raise NotImplementedError('detokenize not implemented')

  def input_to_device_for_continuous_batching(
      self, one_core_inputs: HostTensors, unpadded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Transfers input data to device for either prefill or generate."""
    raise NotImplementedError(
        'input_to_device_for_continuous_batching not implemented'
    )


class ServableModel(abc.ABC):
  """Base class for service implementation, backed by a model."""

  def __init__(self):
    self._methods: Dict[str, ServableMethod] = {}
    self._acls: Dict[str, str] = {}
    self._unloaded = False

  @property
  def unloaded(self) -> bool:
    return self._unloaded

  @property
  def methods(self) -> Dict[str, ServableMethod]:
    return self._methods

  def method(self, method: str) -> ServableMethod:
    """Gets a method with the given name."""
    return self._methods[method]

  def unload(self) -> None:
    """Clears references held by this model."""
    self._unloaded = True
    for method in self._methods.values():
      method.unload()
    self._methods = {}

  def save(self, checkpoint_path: Optional[str]) -> None:
    raise NotImplementedError('Save checkpoint not implemented')

  def add_method(self, key: str, method: ServableMethod) -> None:
    """Adds an initialized method."""
    self._methods[key] = method

  def set_acls(self, acls: Dict[str, str]):
    """Sets the ACLs for this model.

    Args:
      acls: A dictionary from method names (e.g., lm.score) to the name of an
        access control list (e.g., sax-log-access-acl).
    """
    self._acls = acls

  def get_acl(self, method_name: str):
    """Returns the ACL name for the method name.

    Args:
      method_name: The method name (e.g., lm.score).

    Returns:
      None if no explicit ACL name is given. Otherwise, returns
      the ACL name (e.g., sax-log-access-acl).
    """
    return self._acls.get(method_name, None)

  @abc.abstractmethod
  def supports_dummy_compute_on_primary(self) -> bool:
    """Returns if the primary host can use device_compute_with_dummy_data()."""
    # This allows optimizations that performs mult-host sync before the
    # preprocessing, and if error occurred during preprocessing, dummy data can
    # be used to allow the primary host to execute the same program which was
    # already communicated to the secondary hosts.
