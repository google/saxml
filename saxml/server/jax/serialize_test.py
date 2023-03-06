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
"""Tests for serialize."""
import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from saxml.server.jax import serialize


class SerializeTest(absltest.TestCase):

  def get_mesh(self, shape, axes):
    devices = np.array(jax.devices()).reshape(shape)
    return jax.sharding.Mesh(devices, axis_names=axes)

  def test_simple_pjit(self):
    mesh = self.get_mesh((8,), ['x'])

    def fn(x, y):
      return x + y

    in_x = jnp.zeros((8,), jnp.float32) + 3
    in_y = jnp.zeros((8,), jnp.float32) + 4
    shape_dtype = jax.ShapeDtypeStruct((8,), jnp.float32)
    pspec = jax.sharding.PartitionSpec('x')

    s = serialize.serialize_pjittable_function(
        fn, (shape_dtype, shape_dtype), (pspec, pspec), mesh
    )
    des = serialize.deserialize_pjitted_function(s, mesh)
    result = des(in_x, in_y)
    self.assertTrue(np.allclose(result, fn(in_x, in_y)))

  def test_pjit_nested_inputs(self):
    mesh = self.get_mesh((8,), ['x'])

    def fn(x, kw_ins):
      return x * kw_ins['y'] + kw_ins['z']

    in_x = jnp.zeros((8,), jnp.float32) + 3
    in_y = jnp.zeros((8,), jnp.float32) + 4
    in_z = jnp.zeros((8,), jnp.float32) + 8
    shape_dtype = jax.ShapeDtypeStruct((8,), jnp.float32)
    pspec = jax.sharding.PartitionSpec('x')

    s = serialize.serialize_pjittable_function(
        fn,
        (shape_dtype, {'y': shape_dtype, 'z': shape_dtype}),
        (pspec, {'y': pspec, 'z': pspec}),
        mesh,
    )
    des = serialize.deserialize_pjitted_function(s, mesh)
    result = des(in_x, {'y': in_y, 'z': in_z})
    self.assertTrue(np.allclose(result, fn(in_x, {'y': in_y, 'z': in_z})))


if __name__ == '__main__':
  os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
  absltest.main()
