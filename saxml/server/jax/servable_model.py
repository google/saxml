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
import functools
import os
import pprint
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence

from absl import logging
from flax import struct as flax_struct
import jax
from jax import experimental as jax_exp
from jax import numpy as jnp
from jax.experimental import pjit
import numpy as np
from paxml import host_callback as paxml_hcb
from praxis import pytypes
from saxml.server import servable_model
from saxml.server import servable_model_params

# string values can be used in host callbacks and are converted to integers
# when passing around through a global repository on host.
ExtraInput = servable_model_params.ExtraInputs
# TODO(sax-dev): define these types or use pax's definitions.
HostTensors = Any
DeviceTensors = Any
PSpecs = Any
ShapesAndDtypes = Any
Shapes = Any
JaxTensors = Any
InputShapeInfo = servable_model.InputShapeInfo


def remove_padding(x: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
  if list(x.shape) == list(shape):
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
class ServableModelState(flax_struct.PyTreeNode):
  """A data structure holding the state of a loaded model."""

  # Model variables.
  mdl_vars: DeviceTensors
  # Shapes of model variables without GSPMD padding.
  mdl_var_unpadded_shapes: Shapes

  # Whether the current host is the primary in a multi-jax-client setup. It is
  # set to True for Pathways.
  is_primary_host: bool = flax_struct.field(pytree_node=False)
  # Process ID of the primary host.
  primary_process_id: int = flax_struct.field(pytree_node=False)
  # Whether input prefetching to device is needed.
  input_prefetch: bool = flax_struct.field(pytree_node=False)
  # Whether to precompile device computation during model load.
  precompile: bool = flax_struct.field(pytree_node=False)
  # Step for the model variables.
  step: int = flax_struct.field(pytree_node=False)
  # pjit global mesh.
  global_mesh: jax.sharding.Mesh = flax_struct.field(pytree_node=False)
  # Model variables' partition specs.
  mdl_var_pspecs: PSpecs = flax_struct.field(pytree_node=False)


@dataclasses.dataclass
class MethodInputInfo:
  """Holds metadata and placeholder data for a method at a batch size."""

  # Partition specs for the inputs of the device function.
  input_pspecs: PSpecs
  # Global shape and dtype for the inputs of the device function.
  global_inputs_shape_dtype: ShapesAndDtypes
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

  Note on streaming: the final outputs of jax_func() must depend on previous
  host calls inside jax_func() to guarantee correct ordering.
  """

  def __init__(
      self,
      method_params: servable_model_params.ServableMethodParams,
      model_state: ServableModelState,
      prng_key: jnp.ndarray,
      dummy_input_sample: Any,
      enable_auto_sharding: bool = False,
      compiler_options: jax.stages.CompilerOptions | None = None,
  ) -> None:
    # TODO(yuanzx): Need update the docstring.
    # The `compiler options` will be passed to JAX `lowered.compile` during AOT
    # compiliation.
    super().__init__(method_params)
    self._model_state = model_state
    self._per_bs_infos: Dict[InputShapeInfo, MethodInputInfo] = {}
    self._dummy_input_sample = dummy_input_sample
    self._prng_key = prng_key
    self._step = StepCounter()
    self._local_devices = list(model_state.global_mesh.local_devices)
    self._enable_auto_sharding = enable_auto_sharding
    self._compiler_options = compiler_options
    logging.info(
        'Primary host: %d, Current: %d',
        model_state.primary_process_id,
        jax.process_index(),
    )

    devices = model_state.global_mesh.devices.flatten()
    for i, d in enumerate(devices):
      if d.process_index == model_state.primary_process_id:
        logging.info('Setting callback device index %d: %s', i, d)
        self._callback_device = d
        self._callback_device_index = i
        break
    else:
      self._callback_device = devices[0]
      self._callback_device_index = 0

  # TODO(b/322190730): Clean up `callback_device_index`.
  @property
  def callback_device_index(self) -> int:
    return self._callback_device_index

  @property
  def callback_device(self) -> jax.Device:
    return self._callback_device

  def get_sorted_input_shapes(self) -> List[InputShapeInfo]:
    result = []
    for batch_size in self._sorted_batch_sizes:
      result.append(InputShapeInfo(batch_size))
    return result

  def load(self) -> None:
    for input_shape in self.get_sorted_input_shapes():
      logging.info('Initializing for input_shape %s', input_shape)
      self._register_for_input_shape(input_shape)
    if self.batching_wait_secs:
      logging.info('Batching wait time: %fs', self.batching_wait_secs)

  def get_dummy_inputs(self, input_shape: InputShapeInfo) -> HostTensors:
    """Returns host tensors with dummy data at a batch size."""
    return self.pre_processing(
        [self._dummy_input_sample] * input_shape.batch_size
    )

  def _jax_aot_compile(
      self,
      step_fn: Any,
      train_state: ServableModelState,
      inputs_shape_dtype: pytypes.NestedShapedArray,
      compiler_options: jax.stages.CompilerOptions | None,
  ) -> Any:
    """Compiles step_fn ahead of time to extract the shardings.

    Args:
      step_fn: The step_fn function which will be compiled ahead of time.
      train_state: Train state which contains abstract values for ahead of time
        compilation.
      inputs_shape_dtype: Inputs with shape/dtype attributes to be used for
        shape inference.
      compiler_options: The compiler_option passed to `compile` API.

    Returns:
      * A compiled step_fn function
      * The input shardings returned by the auto spmd partitioner.
    """

    def _create_aval(x):
      # canonicalize_dtype is necessary to avoid errors like
      # data types are different when compiling and when being called.
      dtype = jax.dtypes.canonicalize_dtype(x.dtype)
      return jax.ShapeDtypeStruct(x.shape, dtype)

    compiled = step_fn.lower(
        jax.tree_map(_create_aval, train_state.mdl_vars),
        inputs_shape_dtype,
    ).compile(compiler_options=compiler_options)
    return compiled

  def _register_for_input_shape(self, input_shape: InputShapeInfo) -> None:
    batched_host_dummy = self.get_dummy_inputs(input_shape)
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy,
        input_shape.batch_size,
        [self.default_extra_inputs] * input_shape.batch_size,
    )

    def _assert_type(x):
      assert isinstance(x, np.ndarray) or isinstance(
          x, jnp.ndarray
      ), f'Output of pre_processing contained an invalid type: {type(x)}'
      return x

    dummy_step = np.array(0, dtype=np.int32)
    dummy_prng_key = jax.random.PRNGKey(0)
    host_dummy = (
        dummy_step,
        dummy_prng_key,
        batched_host_dummy,
        self.get_nonbatch_inputs(batched_host_dummy),
    )
    host_dummy = jax.tree_util.tree_map(_assert_type, host_dummy)

    def _get_pspec(x):
      # Add a `cores` dimension.
      return jax.sharding.PartitionSpec(
          self.model_state.global_mesh.axis_names, *(None,) * (len(x.shape))
      )

    input_pspecs = jax.tree_util.tree_map(_get_pspec, host_dummy)
    num_cores = len(self.model_state.global_mesh.devices.flat)

    global_inputs_shape_dtype = jax.tree_util.tree_map(
        lambda x: ((num_cores,) + x.shape, x.dtype), host_dummy
    )
    global_inputs_shaped_arrays = jax.tree_util.tree_map(
        lambda x: jax.core.ShapedArray((num_cores,) + x.shape, x.dtype),
        host_dummy,
    )

    info = MethodInputInfo(
        input_pspecs=input_pspecs,
        global_inputs_shape_dtype=global_inputs_shape_dtype,
    )

    info.dummy_inputs_per_device_buffers = self._input_to_device_buffers(
        batched_host_dummy, input_shape, info, is_dummy=True
    )
    info.dummy_inputs = self._device_buffers_to_jax_arrays(
        info.dummy_inputs_per_device_buffers, info
    )

    # Initialize the device function.
    device_fn = self._pjit_device_fn(
        input_pspecs, input_shape.batch_size, info.dummy_inputs
    )

    logging.info(
        'SAX servable_model JAX AOT compiler_options:\n%s',
        pprint.pformat(self._compiler_options),
    )
    if xla_flags := os.getenv('XLA_FLAGS'):
      logging.info(
          'Environment XLA_FLAGS that may be superseded by'
          ' compiler_options:\n%s',
          '\n'.join(xla_flags.split()),
      )

    if self._enable_auto_sharding or self._compiler_options:
      device_fn = self._jax_aot_compile(
          step_fn=device_fn,
          train_state=self.model_state,
          inputs_shape_dtype=global_inputs_shaped_arrays,
          compiler_options=self._compiler_options,
      )

      if self._enable_auto_sharding:
        input_shardings = device_fn.input_shardings[0]
        self._prng_key, _ = jax.random.split(self._prng_key)
        input_pspecs = jax.tree_util.tree_map(
            lambda x: x.spec, input_shardings[1]
        )
        mdl_var_pspecs = input_shardings[0]

        # Reshard model vars based on shardings inferred by the
        # auto-sharding pass. In PAX, we create model vars and
        # initialize them based on the shardings inferred. In SAX, we
        # need to re-shard them as they are created before we run
        # auto-sharding (say when the model is loaded from a
        # checkpoint).
        resharded_mdl_vars = jax.device_put(
            self.model_state.mdl_vars, mdl_var_pspecs
        )
        # Update the model state (self._model_state) with the resharded
        # model vars and their shardings. Note that self.model_state is
        # a getter for the underlying self._model_state field (see
        # below).
        self._model_state = ServableModelState(
            resharded_mdl_vars,
            self.model_state.mdl_var_unpadded_shapes,
            self.model_state.is_primary_host,
            self.model_state.primary_process_id,
            self.model_state.input_prefetch,
            self.model_state.precompile,
            self.model_state.step,
            self.model_state.global_mesh,
            mdl_var_pspecs,
        )

    info.device_fn = device_fn

    if self.model_state.precompile:
      # Compute with dummy to trigger compilation.
      with self.model_state.global_mesh:
        init_dummy_outputs = info.device_fn(
            self.model_state.mdl_vars, info.dummy_inputs
        )

      # Only warm up post processing on primary host.
      if self.model_state.is_primary_host:
        if self.streamable_output:
          # Retrieve streamed outputs until streaming is done.
          stream_state = None
          while True:
            stream_outs = self.dequeue_stream_output()
            _, stream_state = self.post_processing_stream(
                stream_outs, stream_state
            )
            if stream_outs is None:
              break
        else:
          # Transfer of output to host blocks until dummy computation is done.
          outs = self.output_to_host(init_dummy_outputs, self.batch_size)
          # Warm up post processor.
          self.post_processing(outs)

    self._per_bs_infos[input_shape] = info

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

  def resize_host_array(
      self,
      x: np.ndarray,
      global_input_shape_dtype: ShapesAndDtypes,
      unpadded_input_shape: InputShapeInfo,
  ):
    """Checks the shape of x and resizes to the desired shape.

    Args:
      x: Host tensor.
      global_input_shape_dtype: Global input shape and dtype for this tensor.
      unpadded_input_shape: Unpadded input shape.

    Returns:
      host array after padding or slice of x.
    """
    global_shape, global_dtype = global_input_shape_dtype
    assert x.dtype == global_dtype, (x.dtype, global_dtype)
    assert x.shape[1:] == global_shape[2:], (x.shape, global_shape)
    b = x.shape[0]
    assert unpadded_input_shape.batch_size == b, (
        unpadded_input_shape.batch_size,
        b,
    )
    full_b = global_shape[1]
    if b != full_b:
      assert b < full_b
      x = np.concatenate([x, np.repeat(x[:1], full_b - b, 0)], axis=0)
    return x

  def _input_to_device_buffers(
      self,
      one_core_inputs: HostTensors,
      unpadded_input_shape: InputShapeInfo,
      info: MethodInputInfo,
      is_dummy: bool,
  ) -> DeviceTensors:
    step = np.array(self._step.next(), dtype=np.int32)
    host_inputs = jax.tree_util.tree_map(
        functools.partial(
            self.resize_host_array, unpadded_input_shape=unpadded_input_shape
        ),
        one_core_inputs,
        # Only the batched inputs.
        info.global_inputs_shape_dtype[2],
    )
    host_inputs = (
        step,
        self._prng_key,
        host_inputs,
        self.get_nonbatch_inputs(host_inputs),
    )

    def _pad_for_devices(x):
      # Keep x on only one device, and use zeros on other devices.
      return np.pad(
          np.expand_dims(x, (0, 1)),
          [[0, len(self._local_devices) - 1]] + [[0, 0]] * (x.ndim + 1),
      )

    if not self.model_state.input_prefetch:
      if is_dummy:
        pad_fn = lambda x: np.zeros(
            (len(self._local_devices),) + x.shape, x.dtype
        )
      else:
        pad_fn = _pad_for_devices
      return jax.tree_util.tree_map(pad_fn, host_inputs)
    else:
      if is_dummy:
        assert info.dummy_inputs_per_device_buffers is None

        def _to_buffers(x):
          if self.model_state.is_primary_host:
            return [
                jax.device_put(x, d)
                for x, d in zip(_pad_for_devices(x), self._local_devices)
            ]
          else:
            x = np.zeros((1,) + x.shape, x.dtype)
            return [jax.device_put(x, d) for d in self._local_devices]

        return jax.tree_util.tree_map(_to_buffers, host_inputs)
      else:
        assert info.dummy_inputs_per_device_buffers is not None

        def _update_buffers(x, buffers):
          x = np.expand_dims(x, axis=0)
          # Dummy buffers already created before. We only need to update the
          # first device.
          return [jax.device_put(x, self._local_devices[0])] + buffers[1:]

        return jax.tree_util.tree_map(
            _update_buffers, host_inputs, info.dummy_inputs_per_device_buffers
        )

  def _device_buffers_to_jax_arrays(
      self, buffers: Any, info: MethodInputInfo
  ) -> DeviceTensors:
    if not self.model_state.input_prefetch:
      return buffers

    def _to_jax_array(pspec, bufs, shape_dtype):
      shape, _ = shape_dtype
      return jax.make_array_from_single_device_arrays(
          shape,
          jax.sharding.NamedSharding(self.model_state.global_mesh, pspec),
          bufs,
      )

    return jax.tree_util.tree_map(
        _to_jax_array,
        info.input_pspecs,
        buffers,
        info.global_inputs_shape_dtype,
    )

  def input_to_device(
      self,
      one_core_inputs: HostTensors,
      unpadded_shape: InputShapeInfo,
      padded_shape: InputShapeInfo,
  ) -> DeviceTensors:
    """Transfers host inputs to device. Pads incomplete shapes."""
    info = self._per_bs_infos[padded_shape]
    buffers = self._input_to_device_buffers(
        one_core_inputs, unpadded_shape, info, is_dummy=False
    )
    return self._device_buffers_to_jax_arrays(buffers, info)

  def output_to_host(
      self, output_tensors: DeviceTensors, unpadded_batch_size: int
  ) -> HostTensors:
    """Transfers device outputs to host. Removes batch padding."""

    if self.streamable_output:
      # Wait for all host callbacks to finish.
      jax.effects_barrier()

    return jax.tree_util.tree_map(
        lambda x: np.array(x.addressable_data(0))[:unpadded_batch_size],
        output_tensors,
    )

  def remove_batch_padding(
      self, host_tensors: HostTensors, unpadded_batch_size: int
  ) -> HostTensors:
    return jax.tree_util.tree_map(
        lambda x: x[:unpadded_batch_size], host_tensors
    )

  @abc.abstractmethod
  def add_extra_inputs(
      self, input_batch: HostTensors, extra_input_tensors: Dict[str, np.ndarray]
  ) -> HostTensors:
    """Adds extra inputs to input_batch (maybe inplace) and returns it."""

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

    extra_input_dtypes = (
        self.extra_inputs_dtypes if self.extra_inputs_dtypes else {}
    )

    # Add extra signatures to the input_batch.
    extra_input_tensors = {}
    for input_key, default_value in self.default_extra_inputs.items():
      input_value = []
      for i in range(batch_size):
        value = extra_inputs[i].get(input_key, default_value)
        if isinstance(value, str):
          # Map string to int.
          # Downstream usage will look up the string with the repository.
          value = paxml_hcb.repository(input_key).add(value)
          extra_input_dtypes[input_key] = np.int32
        input_value.append(value)

      # Some extra inputs such as per_example_max_decode_steps are ints
      extra_input_tensors[input_key] = np.array(
          input_value, dtype=extra_input_dtypes.get(input_key, np.float32)
      )
    return self.add_extra_inputs(input_batch, extra_input_tensors)

  def device_compute(
      self, input_batch: DeviceTensors, padded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes the device computation."""
    with self.model_state.global_mesh:
      output_batch = self._per_bs_infos[padded_shape].device_fn(
          self.model_state.mdl_vars, input_batch
      )
      return output_batch

  def device_compute_with_dummy_data(
      self, padded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes device computation with dummy inputs."""
    return self.device_compute(
        self._per_bs_infos[padded_shape].dummy_inputs, padded_shape
    )

  @abc.abstractmethod
  def jax_func(
      self,
      mdl_vars: JaxTensors,
      prng_key: jnp.ndarray,
      batched_inputs: JaxTensors,
      non_batched_inputs: JaxTensors,
  ) -> JaxTensors:
    """Invokes the JAX function that implements the device computation."""

  def _pjit_device_fn(
      self, input_pspecs: PSpecs, batch_size: int, dummy_inputs: DeviceTensors
  ) -> Callable[[DeviceTensors, DeviceTensors], DeviceTensors]:
    """Returns a pjit-ed model function with input handling."""
    del dummy_inputs

    def _wrapped_fn(mdl_vars, inputs):
      # Remove padding on the vars.
      mdl_vars = jax.tree_util.tree_map(
          remove_padding, mdl_vars, self.model_state.mdl_var_unpadded_shapes
      )
      mdl_vars = jax.tree_util.tree_map(
          jax.lax.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )

      # Only one core has real data, others have zeros. Summing on the
      # leading `cores` dimension can make data replicated.
      def _replicate(x):
        return jax.lax.with_sharding_constraint(
            jnp.sum(x, axis=0, promote_integers=False), None
        )

      inputs = jax.tree_util.tree_map(_replicate, inputs)
      step, prng_key, batched_inputs, non_batched_inputs = inputs
      prng_key = jax.random.fold_in(prng_key, step)
      outputs = self.jax_func(
          mdl_vars, prng_key, batched_inputs, non_batched_inputs
      )
      if self.streamable_output:
        # This is guaranteed to be called after all previous host callbacks as
        # all host callbacks are marked as ordered.
        jax_exp.io_callback(
            self.mark_stream_output_done,
            None,
            ordered=True,
            sharding=jax.sharding.SingleDeviceSharding(self.callback_device),
        )
        # Unused final outputs. Return something to make output_to_host
        # blocking.
        return jnp.zeros((batch_size,), dtype=jnp.int32)

      return outputs

    # pjit-ed function.
    if self._enable_auto_sharding:
      return pjit.pjit(
          _wrapped_fn,
          in_shardings=(pjit.AUTO(self.model_state.global_mesh), input_pspecs),
          out_shardings=None,
      )
    else:
      return pjit.pjit(
          _wrapped_fn,
          in_shardings=(self.model_state.mdl_var_pspecs, input_pspecs),
          out_shardings=None,
      )

  def unload(self) -> None:
    """Clears references held by this method."""
    del self._model_state
    del self._per_bs_infos
    del self._dummy_input_sample
    del self._prng_key


class ServableModel(servable_model.ServableModel):

  def supports_dummy_compute_on_primary(self) -> bool:
    return True
