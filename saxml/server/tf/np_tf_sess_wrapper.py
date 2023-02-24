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
"""Wraps a TF function with a session to reduce launch overhead."""

import functools
import threading
from typing import Any, Callable

from absl import logging
from saxml.server.tf import tf_session_runner
import tensorflow.compat.v1 as tf

# A threadsafe counter to make session names unique.
_NEXT_COUNTER = 0
_NEXT_COUNTER_LOCK = threading.Lock()

# nested -> sequence, tree.
tree_flatten = lambda x: (tf.nest.flatten(x), x)
# tree, sequence -> nested.
tree_unflatten = tf.nest.pack_sequence_as
# function, tree -> nested.
tree_map = tf.nest.map_structure


class _NumpyTFSessionWrapper:
  """Wrapper of a TF function using tf.Session with numpy.array inputs/outputs.

  Arguments can have nested structures, but the leaf nodes should all be
  numpy.array or None.

  It does not allow kwargs because it needs to maintain a fixed flattening
  order.
  """

  def __init__(
      self,
      fun: Callable[..., Any],
      fix_non_batch_dims: bool,
      is_class_member: bool,
  ) -> None:
    """Creates the wrapper.

    Args:
      fun: The function to be wrapped.
      fix_non_batch_dims:  If True, the wrapped TF computation expects every
        dimension except the 0-th dimesnion of every argument has a known size.
        Otherwise, the wrapped computation assumes the sizes of all dimensions
        are unknown.
      is_class_member: Whether the wrapper is used as an inline decorator on a
        member function of a class.
    """
    global _NEXT_COUNTER
    self._fun = fun
    with _NEXT_COUNTER_LOCK:
      name = f'{fun.__name__}__{_NEXT_COUNTER}'
      _NEXT_COUNTER += 1
    self._runner = tf_session_runner.TFSessionRunner(name)
    self._initialized = False
    self._init_lock = threading.Lock()
    self._fix_non_batch_dims = fix_non_batch_dims
    self._is_class_member = is_class_member

  def _initialize(self, *args) -> None:
    """Initializes on the first call."""
    if self._is_class_member:
      maybe_self_or_cls = (args[0],)
      args = args[1:]
    else:
      maybe_self_or_cls = ()
    args = tree_map(lambda x: None if x is None else x, args)

    def _create_placeholder(x):
      if x is None:
        return None
      if self._fix_non_batch_dims:
        # Shape without known batch dims.
        return tf.placeholder(x.dtype, shape=[None] + list(x.shape[1:]))
      else:
        return tf.placeholder(x.dtype, shape=[None] * x.ndim)

    flat_args, args_tree = tree_flatten(args)

    g = tf.Graph()
    with g.as_default():
      placeholders = [_create_placeholder(x) for x in flat_args]
      self._feed_names = [p.name for p in placeholders if p is not None]
      graph_args = tree_unflatten(args_tree, placeholders)
      logging.info('tf_sess_wrapper tracing %s', self._fun.__name__)
      outs = self._fun(*maybe_self_or_cls, *graph_args)
      self._fetch_names = []

      feed_set = set(self._feed_names)

      def _fill_out_name(x):
        if x is None:
          return None
        # Prevent directly fetching the input.
        if x.name in feed_set:
          x = tf.identity(x)
        idx = len(self._fetch_names)
        self._fetch_names.append(x.name)
        return idx

      self._out_to_fetch_idx = tree_map(_fill_out_name, outs)
    self._runner.initialize(g.as_graph_def())
    self._initialized = True

  def __call__(self, *args) -> Any:
    with self._init_lock:
      if not self._initialized:
        self._initialize(*args)
    if self._is_class_member:
      args = args[1:]
    flat_args = [a for a in tree_flatten(args)[0] if a is not None]
    assert len(flat_args) == len(self._feed_names), 'Unexpected arg count'
    outs = self._runner.run(self._feed_names, self._fetch_names, flat_args)
    return tree_map(
        lambda i: (None if i is None else outs[i]), self._out_to_fetch_idx
    )

  def __get__(self, obj: Any, objtype: Any) -> Any:
    """Supports instance methods."""
    return functools.partial(self.__call__, obj)


def wrap_tf_session(
    fun: Callable[..., Any], fix_non_batch_dims=True
) -> _NumpyTFSessionWrapper:
  """Wraps a TF function with a session to reduce launch overhead.

  Args:
    fun: The function that creates TF ops to compute tensors.
    fix_non_batch_dims: If True, the wrapped TF computation expects every
      dimension except the 0-th dimesnion of every argument has a known size.
      Otherwise, the wrapped computation assumes the sizes of all dimensions are
      unknown.

  Returns:
    A wrapped callable that initializes the graph only once.
  """
  return _NumpyTFSessionWrapper(
      fun, fix_non_batch_dims=fix_non_batch_dims, is_class_member=False
  )


def wrap_tf_session_class_member(
    fun: Callable[..., Any], fix_non_batch_dims=True
) -> _NumpyTFSessionWrapper:
  """Same as wrap_tf_session but used on a class member function."""
  return _NumpyTFSessionWrapper(
      fun, fix_non_batch_dims=fix_non_batch_dims, is_class_member=True
  )
