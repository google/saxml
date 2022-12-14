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
"""JAX sharded implementation of servable model."""

import abc
import dataclasses
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from absl import logging
import jax
from jax import numpy as jnp
from jax.experimental import maps
from jax.experimental import pjit
from jax.interpreters import pxla
import numpy as np
from saxml.server import servable_model
from saxml.server import servable_model_params

ExtraInput = Dict[str, float]
# TODO(sax-dev): define these types or use pax's definitions.
HostTensors = Any
DeviceTensors = Any
PSpecs = Any
ShapesAndDtypes = Any
Shapes = Any
JaxTensors = Any


def remove_padding(x: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
  if list(x.shape) == shape:
    return x
  return jax.lax.slice(x, [0] * x.ndim, shape)


class StepCounter:
  """A thread-safe counter to increment the step number."""

  def __init__(self):
    self._mu: threading.Lock = threading.Lock()
    self._value: int = 0

  def next(self) -> int:
    with self._mu:
      result = self._value
      self._value += 1
      return result


@dataclasses.dataclass
class ServableModelState:
  """A data structure holding the state of a loaded model."""
  # Whether the current host is the primary in a multi-jax-client setup. It is
  # set to True for Pathways.
  is_primary_host: bool
  # pjit global mesh.
  global_mesh: maps.Mesh
  # Model variables.
  mdl_vars: DeviceTensors
  # Model variables' partition specs.
  mdl_var_pspecs: PSpecs
  # Shapes of model variables without GSPMD padding.
  mdl_var_unpadded_shapes: Shapes
  # Whether input prefetching to device is needed.
  input_prefetch: bool
  # Whether to precompile device computation during model load.
  precompile: bool


@dataclasses.dataclass
class MethodInputInfo:
  """Holds metadata and placeholder data for a method at a batch size."""
  # Partition specs for the inputs of the device function.
  input_pspecs: PSpecs
  # Global shape and dtype for the inputs of the device function.
  global_input_shape_dtypes: ShapesAndDtypes
  # Dummy input tensors used for secondary hosts.
  dummy_inputs: Optional[DeviceTensors] = None
  # Dummy input device buffers (on the local devices)
  dummy_inputs_per_device_buffers: Optional[Any] = None
  # The method function to run on the device.
  device_fn: Optional[Callable[..., DeviceTensors]] = None


class ServableMethod(servable_model.ServableMethod):
  """Base class for method implementation and its pre- and post-processing.

  This class initializes the method based on a jax function jax_func(). It also
  provides device-only computation with dummy data for secondary hosts in a
  multi-Jax-client setup.

  Subclasses need to override methods:
    - jax_func()
    - pre_processing()
    - post_processing()
    - add_extra_inputs()
  """

  def __init__(self, method_params: servable_model_params.ServableMethodParams,
               model_state: ServableModelState, prng_key: jnp.ndarray,
               dummy_input_sample: Any) -> None:
    super().__init__(method_params)
    self._model_state = model_state
    self._per_bs_infos: Dict[int, MethodInputInfo] = {}
    self._dummy_input_sample = dummy_input_sample
    self._prng_key = prng_key
    self._step = StepCounter()
    self._local_devices = list(model_state.global_mesh.local_devices)

  def load(self) -> None:
    for batch_size in self._sorted_batch_sizes:
      logging.info('Initializing for batch size %s', batch_size)
      self._register_for_batch_size(batch_size)

  def get_dummy_inputs(self, batch_size: int) -> HostTensors:
    """Returns host tensors with dummy data at a batch size."""
    return self.pre_processing([self._dummy_input_sample] * batch_size)

  def _register_for_batch_size(self, batch_size: int) -> None:
    batched_host_dummy = self.get_dummy_inputs(batch_size)
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy, batch_size,
        [self.default_extra_inputs] * batch_size)

    def _assert_type(x):
      assert isinstance(x, np.ndarray), (
          f'Output of pre_processing contained an invalid type: {type(x)}')
      return x

    dummy_step = np.array(0, dtype=np.int32)
    host_dummy = (dummy_step, batched_host_dummy,
                  self.get_nonbatch_inputs(batched_host_dummy))
    host_dummy = jax.tree_util.tree_map(_assert_type, host_dummy)

    def _get_pspec(x):
      # Add a `cores` dimension.
      return pjit.PartitionSpec(self.model_state.global_mesh.axis_names,
                                *((None,) * len(x.shape)))

    input_pspecs = jax.tree_util.tree_map(_get_pspec, host_dummy)

    num_cores = len(self.model_state.global_mesh.devices.flat)
    global_input_shape_dtypes = jax.tree_util.tree_map(
        lambda x: ((num_cores,) + x.shape, x.dtype), host_dummy)

    self._per_bs_infos[batch_size] = MethodInputInfo(
        input_pspecs=input_pspecs,
        global_input_shape_dtypes=global_input_shape_dtypes,
    )
    info = self._per_bs_infos[batch_size]
    info.dummy_inputs_per_device_buffers = self._input_to_device_buffers(
        batched_host_dummy, batch_size, is_dummy=True)
    info.dummy_inputs = self._device_buffers_to_jax_arrays(
        info.dummy_inputs_per_device_buffers, batch_size)
    # Initialize the device function.
    info.device_fn = self._pjit_device_fn(input_pspecs)

    # Compute with dummy to trigger compilation.
    if self.model_state.precompile:
      init_dummy_outputs = self.device_compute(info.dummy_inputs, batch_size)

    if self.model_state.is_primary_host:
      # Transfer dummy to host to block until dummy computation is done.
      if self.model_state.precompile:
        outs = self.output_to_host(init_dummy_outputs, self.batch_size)
        # Warm up post processor.
        self.post_processing(outs)

  @property
  def model_state(self) -> ServableModelState:
    return self._model_state

  def get_nonbatch_inputs(self, one_core_inputs: HostTensors) -> HostTensors:
    """Returns some optional tensors without batch dim.

    E.g., this can be a step counter, or something that depends on the batched
    input.

    Args:
      one_core_inputs: inputs with a batch dimension.

    Returns:
      Additional inputs without a batch dimension.
    """
    return ()

  def _input_to_device_buffers(self, one_core_inputs: HostTensors,
                               unpadded_batch_size: int,
                               is_dummy: bool) -> DeviceTensors:
    info = self._per_bs_infos[self.get_padded_batch_size(unpadded_batch_size)]

    def _check_and_pad_to_host_array(x, global_input_shape_dtype):
      global_shape, global_dtype = global_input_shape_dtype
      assert x.dtype == global_dtype, (x.dtype, global_dtype)
      assert x.shape[1:] == global_shape[2:], (x.shape, global_shape)
      b = x.shape[0]
      assert unpadded_batch_size == b
      full_b = global_shape[1]
      if b != full_b:
        assert b < full_b
        x = np.concatenate([x, np.repeat(x[:1], full_b - b, 0)], axis=0)
      return x

    step = np.array(self._step.next(), dtype=np.int32)
    host_inputs = jax.tree_util.tree_map(
        _check_and_pad_to_host_array,
        one_core_inputs,
        # Only the batched inputs.
        info.global_input_shape_dtypes[1])
    host_inputs = (step, host_inputs, self.get_nonbatch_inputs(host_inputs))

    def _pad_for_devices(x):
      # Keep x on only one device, and use zeros on other devices.
      return np.pad(
          np.expand_dims(x, (0, 1)),
          [[0, len(self._local_devices) - 1]] + [[0, 0]] * (x.ndim + 1))

    if not self.model_state.input_prefetch:

      if is_dummy:
        return jax.tree_util.tree_map(
            lambda x: np.zeros((len(self._local_devices),) + x.shape, x.dtype),
            host_inputs)
      return jax.tree_util.tree_map(_pad_for_devices, host_inputs)

    if is_dummy:

      def _to_buffers(x):
        if self.model_state.is_primary_host:
          return pxla.device_put(
              _pad_for_devices(x), self._local_devices, replicate=False)
        else:
          x = np.zeros((1,) + x.shape, x.dtype)
          return pxla.device_put(x, self._local_devices, replicate=True)

      return jax.tree_util.tree_map(_to_buffers, host_inputs)

    assert info.dummy_inputs_per_device_buffers is not None

    def _update_buffers(x, existing_buffers):
      x = np.expand_dims(x, axis=0)
      # Dummy buffers already created before. We only need to update the first
      # device.
      return pxla.device_put(
          x, [self._local_devices[0]], replicate=True) + existing_buffers[1:]

    return jax.tree_util.tree_map(_update_buffers, host_inputs,
                                  info.dummy_inputs_per_device_buffers)

  def _device_buffers_to_jax_arrays(self, buffers: Any,
                                    batch_size: int) -> DeviceTensors:
    if not self.model_state.input_prefetch:
      return buffers
    info = self._per_bs_infos[batch_size]

    def _to_jax_array(pspec, bufs, shape_dtype):
      shape, _ = shape_dtype
      return jax.make_array_from_single_device_arrays(
          shape, jax.sharding.NamedSharding(self.model_state.global_mesh,
                                            pspec), bufs)

    return jax.tree_util.tree_map(_to_jax_array, info.input_pspecs, buffers,
                                  info.global_input_shape_dtypes)

  def input_to_device(self, one_core_inputs: HostTensors,
                      unpadded_batch_size: int) -> DeviceTensors:
    """Transfers input data to device. Pads incomplete batches."""
    buffers = self._input_to_device_buffers(
        one_core_inputs, unpadded_batch_size, is_dummy=False)
    return self._device_buffers_to_jax_arrays(
        buffers, self.get_padded_batch_size(unpadded_batch_size))

  def output_to_host(self, output_tensors: DeviceTensors,
                     unpadded_batch_size: int) -> HostTensors:
    """Fetches device outputs to host. Removes batch padding."""
    return jax.tree_util.tree_map(
        lambda x: np.array(x.addressable_data(0))[:unpadded_batch_size],
        output_tensors)

  @abc.abstractmethod
  def add_extra_inputs(
      self, input_batch: HostTensors,
      extra_input_tensors: Dict[str, np.ndarray]) -> HostTensors:
    """Adds extra inputs to input_batch (maybe inplace) and returns it."""

  def update_extra_inputs(
      self,
      input_batch: HostTensors,
      batch_size: int,
      extra_inputs: Optional[List[ExtraInput]] = None) -> HostTensors:
    """Updates mutable input keys to input batch.

    Users would like to update some input keys for the input batch through
    PRC requests. This function updates the per example mutable input value in
    the input batch from extra_inputs.

    Args:
      input_batch: Nested numpy arrays for device computation function input. It
        could be mutated.
      batch_size: Batch size of the input_batch.
      extra_inputs: Optional list of dictionary for {input_key: scalar_value}
        for each example. The keys in different elements of list could be
        different. The element in the list could be an empty dictionary. When it
        is None, when fill extra_inputs with self.default_extra_inputs.

    Returns:
      Updated input batch.
    """
    if self.default_extra_inputs is None:
      return input_batch

    if extra_inputs is None:
      extra_inputs = [self.default_extra_inputs] * batch_size

    # Add extra signatures to the input_batch.
    extra_input_tensors = {}
    for input_key, default_value in self.default_extra_inputs.items():
      input_value = np.empty((batch_size,), dtype=np.float32)
      for i in range(batch_size):
        input_value[i] = extra_inputs[i].get(input_key, default_value)
      extra_input_tensors[input_key] = input_value
    return self.add_extra_inputs(input_batch, extra_input_tensors)

  def device_compute(self, input_batch: DeviceTensors,
                     unpadded_batch_size: int) -> DeviceTensors:
    """Executes the device computation."""
    padded_batch_size = self.get_padded_batch_size(unpadded_batch_size)
    with self.model_state.global_mesh:
      output_batch = self._per_bs_infos[padded_batch_size].device_fn(
          self.model_state.mdl_vars, input_batch)
      return output_batch

  def compute_with_dummy_data(self, unpadded_batch_size: int) -> DeviceTensors:
    """Executes device computation with dummy inputs."""
    padded_batch_size = self.get_padded_batch_size(unpadded_batch_size)
    return self.device_compute(
        self._per_bs_infos[padded_batch_size].dummy_inputs, padded_batch_size)

  @abc.abstractmethod
  def jax_func(self, mdl_vars: JaxTensors, prng_key: jnp.ndarray,
               batched_inputs: JaxTensors,
               non_batched_inputs: JaxTensors) -> JaxTensors:
    """Invokes the JAX function that implements the device computation."""

  def _pjit_device_fn(
      self, input_pspecs: PSpecs
  ) -> Callable[[DeviceTensors, DeviceTensors], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""

    def _wrapped_fn(mdl_vars, inputs):
      # Remove padding on the vars.
      mdl_vars = jax.tree_util.tree_map(
          remove_padding, mdl_vars, self.model_state.mdl_var_unpadded_shapes)
      mdl_vars = jax.tree_util.tree_map(pjit.with_sharding_constraint, mdl_vars,
                                        self.model_state.mdl_var_pspecs)

      # Only one core has real data, others have zeros. Summing on the leading
      # leading `cores` dimension can make data replicated.
      def _replicate(x):
        return pjit.with_sharding_constraint(
            jnp.sum(x, axis=0, promote_integers=False), None)

      inputs = jax.tree_util.tree_map(_replicate, inputs)
      step, batched_inputs, non_batched_inputs = inputs
      prng_key = jax.random.fold_in(self._prng_key, step)
      outputs = self.jax_func(mdl_vars, prng_key, batched_inputs,
                              non_batched_inputs)
      # Make sure outputs are replicated.
      return jax.tree_map(lambda x: pjit.with_sharding_constraint(x, None),
                          outputs)

    # pjit-ed function.
    return pjit.pjit(
        _wrapped_fn,
        in_axis_resources=(self.model_state.mdl_var_pspecs, input_pspecs),
        out_axis_resources=None)

  def unload(self) -> None:
    """Clears references held by this method."""
    del self._model_state
    del self._per_bs_infos
    del self._dummy_input_sample
    del self._prng_key


class ServableModel(servable_model.ServableModel):

  def supports_dummy_compute_on_primary(self) -> bool:
    return True
