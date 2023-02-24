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
"""Test for tf_sess_wraper."""

from absl.testing import absltest
import numpy as np
from saxml.server.tf import np_tf_sess_wrapper
import tensorflow as tf


class TFSessWrapperTest(absltest.TestCase):

  def test_simple(self):
    def fn(x, y):
      return tf.add(x, y) * x

    wrapper = np_tf_sess_wrapper.wrap_tf_session(fn)

    in_x = np.zeros((8, 3)) + 3
    in_y = np.zeros((8, 3)) + 4
    ref_out = fn(in_x, in_y)
    wrapper_out = wrapper(in_x, in_y)
    self.assertTrue(np.allclose(ref_out, wrapper_out))

    # Change batch dim.
    in_x = np.zeros((9, 3)) + 8
    in_y = np.zeros((9, 3)) + 7
    ref_out = fn(in_x, in_y)
    wrapper_out = wrapper(in_x, in_y)
    self.assertTrue(np.allclose(ref_out, wrapper_out))

  def test_nested(self):
    def fn(x, y):
      return {'c': tf.add(x['a'], y) * x['b'], 'd': x['a'] * x['b']}

    wrapper = np_tf_sess_wrapper.wrap_tf_session(fn)

    in_x = {'a': np.zeros((8, 3)) + 3, 'b': np.zeros((8, 3)) + 5}
    in_y = np.zeros((8, 3)) + 4
    ref_out = fn(in_x, in_y)
    wrapper_out = wrapper(in_x, in_y)
    self.assertTrue(np.allclose(ref_out['c'], wrapper_out['c']))
    self.assertTrue(np.allclose(ref_out['d'], wrapper_out['d']))

    # Change batch dim.
    in_x = {'a': np.zeros((3, 3)) + 3, 'b': np.zeros((3, 3)) + 5}
    in_y = np.zeros((3, 3)) + 4
    ref_out = fn(in_x, in_y)
    wrapper_out = wrapper(in_x, in_y)
    self.assertTrue(np.allclose(ref_out['c'], wrapper_out['c']))
    self.assertTrue(np.allclose(ref_out['d'], wrapper_out['d']))

  def test_class_member(self):
    class _TestClass:

      @np_tf_sess_wrapper.wrap_tf_session_class_member
      def fn(self, x, y):
        return tf.add(x, y) * x

    in_x = np.zeros((8, 3)) + 3
    in_y = np.zeros((8, 3)) + 4
    ref_out = (in_x + in_y) * in_x
    wrapper_out = _TestClass().fn(in_x, in_y)
    self.assertTrue(np.allclose(ref_out, wrapper_out))


if __name__ == '__main__':
  tf.test.main()
