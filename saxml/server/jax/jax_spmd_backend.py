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
"""JAX SPMD backend."""

import functools
from typing import Callable

import jax
from jax.experimental import mesh_utils
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
from saxml.server import utils
from saxml.server.spmd_backend import SPMDBackend

JTensor = jnp.ndarray
NpTensor = np.ndarray
_MESSAGE_BUF_LEN = 1024


def _encode_str_to_tensor(message: str) -> NpTensor:
  """Encodes a message to a fix-sized uint8 buffer."""
  # Suppot only messages without any `0` byte.
  data = bytearray(message, 'ascii')
  assert 0 not in data
  assert len(data) <= _MESSAGE_BUF_LEN
  tensor = np.array(data, dtype=np.uint8)
  tensor = np.pad(tensor, [0, _MESSAGE_BUF_LEN - len(data)])
  return tensor


def _decode_tensor_to_str(tensor: JTensor) -> str:
  data = bytearray(tensor)
  length = data.find(b'\x00')
  if length >= 0:
    data = data[:length]
  return data.decode()


@functools.partial(pjit.pjit, out_axis_resources=None)
def _all_reduce(x: jax.Array) -> jax.Array:
  """Computes a sum of the values of x across all devices."""
  return jnp.sum(x, axis=0, promote_integers=False)


class JaxSPMDBackend(SPMDBackend):
  """JAX SPMD backend."""

  def __init__(self):
    mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    self._mesh = jax.sharding.Mesh(mesh, ('all',))
    zero = np.zeros(
        (
            1,
            _MESSAGE_BUF_LEN,
        ),
        dtype=np.uint8,
    )
    self._local_devices = list(self._mesh.local_devices)
    self._zero_bufs = [jax.device_put(zero, d) for d in self._local_devices]
    self._sharding = jax.sharding.NamedSharding(
        self._mesh, jax.sharding.PartitionSpec('all', None)
    )
    self._global_shape = (len(self._mesh.devices.flat), _MESSAGE_BUF_LEN)
    self._zero_jax_array = jax.make_array_from_single_device_arrays(
        self._global_shape, self._sharding, self._zero_bufs
    )
    self._process_idx = jax.process_index()

    self._process_count = jax.process_count()
    # Mock TPU requires single host to avoid syncs
    if utils.is_mock_tpu_backend():
      self._process_count = 1

    @functools.lru_cache()
    def _cached_str_to_jax_array(message: str) -> jax.Array:
      data = np.expand_dims(_encode_str_to_tensor(message), 0)
      buf = jax.device_put(data, self._local_devices[0])
      return jax.make_array_from_single_device_arrays(
          self._global_shape, self._sharding, [buf] + self._zero_bufs[1:]
      )

    self._str_to_jax_array = _cached_str_to_jax_array

  def spmd_host_index(self) -> int:
    return self._process_idx

  def spmd_host_count(self) -> int:
    return self._process_count

  def send_via_device(self, message: str) -> None:
    """Sends raw string via device communication. Does not block."""
    jax_array = self._str_to_jax_array(message)
    # We don't block on the result.
    with self._mesh:
      _all_reduce(jax_array)

  def receive_via_device(self) -> str:
    with self._mesh:
      result = _all_reduce(self._zero_jax_array)
    return _decode_tensor_to_str(np.array(result.addressable_data(0)))  # pytype: disable=wrong-arg-types  # jax-ndarray

  def receive_via_device_async(
      self, thread_pool: utils.ThreadPool, done: Callable[[str], None]
  ) -> None:
    with self._mesh:
      result = _all_reduce(self._zero_jax_array)

    def _done():
      data = _decode_tensor_to_str(np.array(result.addressable_data(0)))  # pytype: disable=wrong-arg-types  # jax-ndarray
      done(data)

    thread_pool.run(_done)
