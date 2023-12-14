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
"""Utility funcions to read checkpoint from TensorStore."""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from jax.experimental.array_serialization import serialization
from lingvo import compat as tf
import numpy as np
import p_runner
import tensorstore as ts

PRunner = p_runner.PRunner


def _var_prefix(var_name: str) -> int:
  """Gets variable prefix from the variable name.

  Args:
    var_name: variable name from TensorStore checkpoint.

  Returns:
    If the variable has prefix pattern, returns the prefix dim size,
  otherwise, returns 1.
  """
  pattern = r"#(\d+)#"
  match = re.search(pattern, var_name)
  if match:
    substring = match.group(1)
    return int(substring)
  return 1


def _base_name(var_name) -> Optional[str]:
  """Gets base_name from the variable name.

  Args:
    var_name: variable name from TensorStore checkpoint.

  Returns:
    Base name of the variable in the checkpoint tree.
  """
  if var_name.startswith("opt"):
    return var_name.split(".", 3)[-1]

  if var_name.startswith("mdl"):
    return var_name.split(".", 1)[-1]
  return None


def _get_slice_def(slices, index, chunk_shape) -> List[slice]:
  """Gets slice definition for TensorStore dataset."""
  slice_def = []
  slice_index = index
  for axis, per_dim_slice in enumerate(slices):
    if per_dim_slice == 1:
      slice_def.append(slice(None))
      continue
    if axis == (len(slices) - 1):
      denom = 1
    else:
      denom = int(np.prod(slices[axis + 1 :]))
    start = slice_index // denom
    slice_index -= start * denom
    start *= chunk_shape[axis]
    end = start + chunk_shape[axis]
    slice_def.append(slice(start, end, 1))
  return slice_def


def _prime_factors(n: int) -> list[int]:
  """Returns the prime factors of the given integer."""
  i = 2
  factors = []
  while i * i <= n:
    if n % i:
      i += 1
    else:
      n //= i
      factors.append(i)
  if n > 1:
    factors.append(n)
  return factors


def _get_almost_largest_chunk_shape_under(
    *,
    var_shape: np.ndarray,
    min_chunk_shape: np.ndarray,
    max_num_bytes: int,
) -> list[int]:
  """Returns the largest chunk shape under the given size limit."""
  total_num_chunks = int(max_num_bytes / (np.prod(min_chunk_shape) * 4))
  chunk_scales = []
  # We try to expand the chunk as much as possible on higher axes.
  # Note this algorithm doesn't guarantee to find the largest chunk shape, but
  # more readable than the optimal algorithm.
  for num_slices in reversed(var_shape // min_chunk_shape):
    chunk_scale = 1
    for factor in _prime_factors(num_slices):
      if total_num_chunks >= factor:
        total_num_chunks //= factor
        chunk_scale *= factor
    chunk_scales.append(chunk_scale)
  chunk_scales = np.array(list(reversed(chunk_scales)))
  return list(min_chunk_shape * chunk_scales)


def _get_tensorstore_spec(path: str) -> dict[str, object]:
  """Create a Spec that may be used to open/recreate TensorStore.

  Args:
    path: Name of the tensorstore path.

  Returns:
    TensorStore spec.
  """
  if path.startswith("gs://"):
    return serialization.get_tensorstore_spec(path)
  else:
    return {"driver": "zarr", "kvstore": {"driver": "file", "path": path}}


class TensorStoreReader(object):
  """Class to read tensors from TensorStore.

  TensorStore documentation: https://google.github.io/tensorstore/

  Typical usage example:

  reader = TensorStoreReader(CKPT_PATH)
  var_value = reader.read_variable(var_name)
  """

  def __init__(self, checkpoint_dir: str):
    self._ckpt_dir = checkpoint_dir
    self._vars = self._get_variables()
    self._stacked_vars = self._get_stacked_vars()

  def _get_variables(self) -> List[str]:
    """Private function to get variable list from the checkpoint."""
    return tf.io.gfile.listdir(self._ckpt_dir)

  def get_variables(self) -> List[str]:
    """Get variable list from the checkpoint."""
    return self._vars

  def _get_stacked_vars(self) -> Dict[str, int]:
    """Gets variables those are stacked in the checkpoint.

    Returns:
      A dictionary of {variable_base_name: num_stacks}.
    """
    dict_result = {}
    for var_name in self._vars:
      num_stacks = _var_prefix(var_name)
      if num_stacks > 1:
        base_name = _base_name(var_name)
        if base_name is not None:
          dict_result[base_name] = num_stacks
    return dict_result

  def get_num_stacks(self, var_name: str) -> int:
    """Gets number of stacks for the variable.

    Args:
      var_name: variable name from TensorStore checkpoint.

    Returns:
      If the variable is stacked, return number of stacks size. Otherwise,
    return 1.
    """
    if "no_prefix" in var_name:
      return 1

    base_name = _base_name(var_name)
    if base_name and base_name in self._stacked_vars:
      return self._stacked_vars[base_name]
    return 1

  def _get_variable_dataset(self, var_name: str) -> ts.TensorStore:
    assert var_name in self._vars, (
        f"Unexpected variable {var_name} from checkpoint, available vars are"
        f" {self._vars}."
    )
    path = os.path.join(self._ckpt_dir, var_name)
    return ts.open(_get_tensorstore_spec(path)).result()

  def read_chunk_variable(
      self, var_name: str, chunk_shape: List[int], index: int
  ) -> np.ndarray:
    """Reads a chunk from a variable."""
    dataset = self._get_variable_dataset(var_name)
    dest_shape = dataset.shape
    assert len(dest_shape) == len(chunk_shape)
    chunk_shape = np.array(chunk_shape)

    slices = [dest_shape[i] // chunk_shape[i] for i in range(len(chunk_shape))]
    tf.logging.info("slices: %s", slices)

    if index >= int(np.prod(slices)):
      index = int(np.prod(slices)) - 1
    slice_def = _get_slice_def(slices, index, chunk_shape)
    return dataset[tuple(slice_def)].read().result()

  def read_variable(
      self, var_name: str, stack_id: int = 0, num_stacks: int = 1
  ) -> np.ndarray:
    """Reads variable value from the checkpoint.

    Args:
      var_name: variable name from TensorStore checkpoint.
      stack_id: Stack id in the stacked variable.
      num_stacks: Number of stacks for the stacked variable.

    Returns:
      If the variable is stacked, returns a slice of the variable using
    stack_id. Otherwise returns the whole tensor of the variable.
    """
    if num_stacks > 1:
      return self._get_variable_dataset(var_name)[stack_id, ...].read().result()
    return self._get_variable_dataset(var_name).read().result()

  def get_variable_to_shape_map(self) -> Dict[str, Tuple[int]]:
    """Returns variable name to shape map from the checkpoint."""

    def worker(var_names):
      res = dict()
      for var_name in var_names:
        res[var_name] = self.get_variable_shape(var_name)
      return res

    result = PRunner(worker, 30).run(self._vars)
    return result

  def get_variable_to_chunks_map(self) -> Dict[str, Tuple[int]]:
    """Returns variable name to shape map from the checkpoint."""

    def worker(var_names):
      res = dict()
      for var_name in var_names:
        res[var_name] = self.get_variable_chunks(var_name)
      return res

    result = PRunner(worker, 30).run(self._vars)
    return result

  def get_variable_to_dtype_map(self) -> Dict[str, np.dtype]:
    """Returns variable name to dtype map from the checkpoint."""

    def worker(var_names):
      res = dict()
      for var_name in var_names:
        res[var_name] = self.get_variable_dtype(var_name)
      return res

    result = PRunner(worker, 30).run(self._vars)
    return result

  def get_variable_dtype(self, var_name: str) -> Tuple[int]:
    """Returns variable dtype from the checkpoint."""
    return self._get_variable_dataset(var_name).dtype

  def get_variable_shape(self, var_name: str) -> Tuple[int]:
    """Returns variable shape from the checkpoint."""
    return self._get_variable_dataset(var_name).shape

  def get_variable_chunks(self, var_name: str) -> Tuple[int]:
    """Returns variable chunks from the checkpoint."""
    assert var_name in self._vars, (
        f"Unexpected variable {var_name} from checkpoint, available vars are"
        f" {self._vars}."
    )
    path = os.path.join(self._ckpt_dir, f"{var_name}/.zarray")
    with tf.io.gfile.GFile(path, "r") as json_file:
      spec = json.load(json_file)
      return spec["chunks"]


class TensorStoreWriter:
  """Class to write tensors to TensorStore.

  TensorStore documentation: https://google.github.io/tensorstore/

  Typical usage example:

  writer = TensorStoreWriter(CKPT_PATH)
  writer.write_variable(key, value)
  """

  def __init__(self, checkpoint_dir: str):
    """Constructor for TensorStoreWriter.

    Args:
      checkpoint_dir: Checkpoint directory path.
    """
    self._ckpt_dir = checkpoint_dir

  def write_chunk_variable(
      self, var_name: str, value: np.ndarray, full_shape: List[int], index: int
  ) -> None:
    """Writes variable to a TensorStore chunk."""
    chunk_shape = value.shape
    dest_shape = np.array(full_shape)
    assert len(dest_shape) == len(chunk_shape)

    slices = [dest_shape[i] // chunk_shape[i] for i in range(len(chunk_shape))]
    tf.logging.info("slices: %s", slices)

    if index >= int(np.prod(slices)):
      return

    path = os.path.join(self._ckpt_dir, var_name)
    tf.logging.info("var_name: %s", var_name)
    tf.logging.info("dtype: %s", value.dtype)
    if str(value.dtype) == "bfloat16":
      # zarr drivier doesn't support '<V2' type
      dtype = "bfloat16"
    else:
      dtype = np.dtype(value.dtype).str

    tensorstore_spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "file",
            "path": path,
        },
    }
    tensorstore_spec["metadata"] = {
        "compressor": {"id": "gzip"},
        "shape": dest_shape,
        "chunks": chunk_shape,
        "dtype": dtype,
    }
    dataset = ts.open(
        ts.Spec(tensorstore_spec),
        create=True,
        open=True,
        context=ts.Context({"file_io_concurrency": {"limit": 128}}),
    ).result()
    slice_def = _get_slice_def(slices, index, chunk_shape)
    dataset[tuple(slice_def)].write(value).result()

  def write_variable(
      self,
      var_name: str,
      value: np.ndarray,
      stack_id: int = 0,
      num_stacks: int = 1,
      expert_id: int = -1,
      num_experts: int = 1,
  ) -> None:
    """Writes variable to the TensorStore.

    Args:
      var_name: variable name.
      value: value in numpy array format.
      stack_id: stack id of the source checkpoint var in destination checkpoint.
      num_stacks: number of stacks of the checkpont.
      expert_id: expert id in MoE variable.
      num_experts: number of experts in MoE variable.
    """
    path = os.path.join(self._ckpt_dir, var_name)
    tf.logging.info("var_name: %s", var_name)
    tf.logging.info("stack_id: %d", stack_id)
    tf.logging.info("dtype: %s", value.dtype)
    if str(value.dtype) == "bfloat16":
      # zarr drivier doesn't support '<V2' type
      dtype = "bfloat16"
    else:
      dtype = np.dtype(value.dtype).str

    tensorstore_spec = _get_tensorstore_spec(path)
    dest_shape = value.shape
    chunk_shape = value.shape
    if num_stacks > 1:
      if expert_id == -1:
        dest_shape = (num_stacks,) + value.shape
        chunk_shape = (1,) + value.shape
      else:
        dest_shape = (
            num_stacks,
            num_experts,
        ) + value.shape
        chunk_shape = (
            1,
            1,
        ) + value.shape
    tensorstore_spec["metadata"] = {
        "compressor": {"id": "gzip"},
        "shape": dest_shape,
        "chunks": chunk_shape,
        "dtype": dtype,
    }
    dataset = ts.open(
        ts.Spec(tensorstore_spec),
        create=True,
        open=True,
        context=ts.Context({"file_io_concurrency": {"limit": 128}}),
    ).result()
    if num_stacks > 1:
      if expert_id != -1:
        dataset[stack_id, expert_id, ...].write(value).result()
      else:
        dataset[stack_id, ...].write(value).result()
    else:
      dataset.write(value).result()


class LowRamTensorStoreReader(TensorStoreReader):
  """A TensorStoreReader that consumes less RAM."""

  def __init__(
      self, *args, max_num_bytes_per_read: Optional[int] = None, **kwargs
  ):
    super().__init__(*args, **kwargs)
    self._max_num_bytes_per_read = max_num_bytes_per_read

  def read_variable(
      self, var_name: str, stack_id: int = 0, num_stacks: int = 1
  ) -> np.ndarray:
    if num_stacks == 1:
      return self._get_variable_dataset(var_name).read().result()

    chunk_shape = list(self.get_variable_chunks(var_name))
    if chunk_shape[0] == 1:
      return self._get_variable_dataset(var_name)[stack_id, ...].read().result()

    dataset = self._get_variable_dataset(var_name)
    var_shape = dataset.shape
    if self._max_num_bytes_per_read is not None:
      chunk_shape = _get_almost_largest_chunk_shape_under(
          var_shape=np.array(var_shape),
          min_chunk_shape=np.array(chunk_shape),
          max_num_bytes=self._max_num_bytes_per_read,
      )
    slices = [var_shape[i] // chunk_shape[i] for i in range(len(chunk_shape))]

    if slices[0] != 1:
      raise NotImplementedError(
          f"Unsupported chunk shape {chunk_shape} and "
          f"variable shape {var_shape}"
      )

    num_slices = int(np.prod(slices))
    chunks = np.empty(slices, object)
    for i in range(num_slices):
      chunk = self.read_chunk_variable(var_name, chunk_shape, i)
      indices = []
      for slice_size in reversed(slices):
        indices.append(i % slice_size)
        i //= slice_size
      # Use np.copy to avoid holding the reference to the entire chunk.
      chunks[tuple(reversed(indices))] = np.copy(chunk[stack_id : stack_id + 1])
    return np.block(chunks.tolist())[0]
