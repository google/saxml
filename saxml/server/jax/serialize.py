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
"""Utilities for serializing a model. (This is experimental.)"""

import dataclasses
from typing import Any, Callable, Sequence

import jax
from jax.experimental import pjit
from jax.lib import xla_client as xc


@dataclasses.dataclass
class SerializedPjitFunction:
  """Represents a single serializable function with fixed shapes."""

  # Serializable IR for the device computation.
  ir: Any
  # Abstract arrays for the flattend inputs.
  flat_global_in_avals: Sequence[jax.abstract_arrays.ShapedArray]
  # Abstract arrays for the flattend outputs.
  flat_global_out_avals: Sequence[jax.abstract_arrays.ShapedArray]
  # Whether the compilation uses a tuple to hold all args.
  tuple_args: bool
  # Shardings for the flattened inputs.
  flat_in_op_shardings: Sequence[xc.OpSharding]
  # Output pytree structure.
  out_tree: Any

  def to_proto(self):
    raise NotImplementedError('not implemented')

  def from_proto(self):
    raise NotImplementedError('not implemented')


def serialize_pjittable_function(
    fun: Callable[..., Any],
    global_inputs_shape_dtype: Sequence[Any],
    input_pspecs: Sequence[Any],
    mesh: jax.sharding.Mesh,
) -> SerializedPjitFunction:
  """Converts a pjittable function to a SerializedPjitFunction.

  Args:
    fun: The python JAX function to be pjitted. Only positional args are
      supported, but each arg can be an arbitrary pytree of arrays.
    global_inputs_shape_dtype: Sequence of input shape_dtypes for the positional
      args. It should have the same pytree structure as the inputs to fun.
    input_pspecs: Sequence of pspecs for the positional args. It should have the
      same pytree structure as the inputs to fun.
    mesh: Global device mesh for pjit.

  Returns:
    The converted SerializedPjitFunction.
  """
  # pjit-ed function. We always replicate the output.
  pjitted = pjit.pjit(fun, in_shardings=input_pspecs, out_shardings=None)
  with mesh:
    lowered = pjitted.lower(*global_inputs_shape_dtype)
    mesh_comp = lowered._lowering  # pylint: disable=protected-access
    assert isinstance(mesh_comp, jax.pxla.MeshComputation)
    global_in_avals = mesh_comp.compile_args['global_in_avals']
    global_out_avals = mesh_comp.compile_args['global_out_avals']
    tuple_args = mesh_comp.compile_args['tuple_args']
    flat_in_shardings = []
    for s, x in zip(mesh_comp.compile_args['in_shardings'], global_in_avals):
      # pylint: disable=protected-access
      flat_in_shardings.append(s._to_xla_op_sharding(x.ndim))
      # pylint: enable=protected-access
    ir = lowered.compiler_ir()
  return SerializedPjitFunction(
      ir=ir,
      flat_global_in_avals=global_in_avals,
      flat_global_out_avals=global_out_avals,
      tuple_args=tuple_args,
      flat_in_op_shardings=flat_in_shardings,
      out_tree=lowered.out_tree,
  )


def deserialize_pjitted_function(
    serialized: SerializedPjitFunction, mesh: jax.sharding.Mesh
) -> Callable[..., Any]:
  """Converts a SerializedPjitFunction to a callable compiled function.

  Args:
    serialized: The SerializedPjitFunction.
    mesh: Global device mesh for pjit.

  Returns:
    The compiled, sharded function that can be called from python.
  """
  with mesh:
    devices = list(mesh.devices.flat)
    backend = devices[0].client
    in_shardings = tuple(
        jax.sharding.GSPMDSharding(devices, s)
        for s in serialized.flat_in_op_shardings
    )
    rep_sharding = jax.sharding.GSPMDSharding.get_replicated(devices)
    num_ins = len(serialized.flat_in_op_shardings)
    num_outs = len(serialized.flat_global_out_avals)

    mc = jax.pxla.MeshComputation(
        'step_fn',
        serialized.ir,
        False,
        (False,) * num_ins,
        in_shardings=in_shardings,
        out_shardings=(rep_sharding,) * num_outs,
        global_in_avals=serialized.flat_global_in_avals,
        global_out_avals=serialized.flat_global_out_avals,
        spmd_lowering=True,
        tuple_args=serialized.tuple_args,
        auto_spmd_lowering=False,
        unordered_effects=[],
        ordered_effects=[],
        host_callbacks=[],
        keepalive=[],
        kept_var_idx=set(range(num_ins)),
        mesh=None,
        backend=backend,
        device_assignment=devices,
        committed=True,
    )

    compiled = mc.compile()

    def dev_fun(*args):
      flat_ins, _ = jax.tree_util.tree_flatten(args)
      flat_outs = compiled.unsafe_call(*flat_ins)
      return jax.tree_util.tree_unflatten(serialized.out_tree, flat_outs)

    return dev_fun
