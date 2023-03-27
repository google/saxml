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
ExtraInput = Dict[str, float]


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
    self._stream_queue: queue.SimpleQueue[Optional[HostTensors]] = (
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
      self, one_core_inputs: HostTensors, unpadded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Transfers input data to device. Pads incomplete batches."""

  @abc.abstractmethod
  def output_to_host(
      self, output_tensors: DeviceTensors, unpadded_batch_size: int
  ) -> HostTensors:
    """Fetches device outputs to host. Removes batch padding."""

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
    """Bathing waiting secs in the server for this method."""
    return self._batching_wait_secs

  @abc.abstractmethod
  def pre_processing(self, raw_inputs: List[Any]) -> HostTensors:
    """Preprocesses an unpadded batch of data into host arrays."""

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
    """Postprocesses the output host arrays to final host output."""

  def post_processing_stream(
      self,
      compute_outputs: Optional[HostTensors] = None,
      stream_state: Optional[Any] = None,
  ) -> Tuple[List[Tuple[List[str], List[float]]], Optional[Any]]:
    """Postprocesses one streaming output.

    Args:
      compute_outputs: Tensors streamed out of the device. If missing, finalize
        streaming using stream_state.
      stream_state: The stream_state returned by the previous call. If missing,
        initialize streaming using compute_outputs as the first input.

    Returns:
      Final host output and state to pass to the next call. The output contains
      a list of batch tuples. Each tuple contains two lists of num_samples
      elements: decoded strings and scores, respectively.
    """
    raise NotImplementedError('post_processing_stream not implemented')

  @abc.abstractmethod
  def device_compute(
      self, input_batch: DeviceTensors, unpadded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes the device computation."""

  @abc.abstractmethod
  def device_compute_with_dummy_data(
      self, unpadded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes device computation with dummy inputs."""
    # This is needed for multi-host SPMD programs to execute in sync.

  @property
  @abc.abstractmethod
  def streamable(self) -> bool:
    """Whether this method supports streaming."""

  def dequeue_stream_output(self) -> Optional[HostTensors]:
    """Dequeues streamed tensors, or None if done. Blocking if empty."""
    return self._stream_queue.get()

  def enqueue_stream_output(self, stream_outputs: HostTensors) -> None:
    """Enqueues streamed tensors."""
    self._stream_queue.put(stream_outputs)

  def mark_stream_output_done(self) -> None:
    """Marks the streaming as done."""
    self._stream_queue.put(None)

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
    """Executes pre_processing, device_compute, and post_processing."""
    assert not self.streamable
    unpadded_batch_size = len(raw_inputs)
    if unpadded_batch_size > self.batch_size:
      raise ValueError(
          'Inputs to compute() had a larger batch size ('
          f'{unpadded_batch_size}) than was '
          f'configured ({self.batch_size})'
      )
    inputs = self.pre_processing(raw_inputs)
    unpadded_shape = self.get_unpadded_shape(unpadded_batch_size, inputs)
    inputs = self.update_extra_inputs(
        inputs, unpadded_shape.batch_size, extra_inputs
    )
    inputs = self.input_to_device(inputs, unpadded_shape)
    padded_shape = self.get_padded_input_shape(unpadded_shape)
    outputs = self.device_compute(inputs, padded_shape)
    outputs = self.output_to_host(outputs, unpadded_shape.batch_size)
    return self.post_processing(outputs)


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
