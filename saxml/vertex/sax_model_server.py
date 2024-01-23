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
"""Business logic for running a SAX server as a separate process.

Typical usage:
  server = SAXModelServer()
  server.run(cmd_list)
  exit_code = server.wait()
"""

# Necessary to annotate subprocess.Popen[...] for older builds.
from __future__ import annotations

import dataclasses
import subprocess
import threading
from typing import AnyStr, Callable, Optional, List, Sequence

from absl import logging
from saxml.vertex import constants
from tornado import process as tornado_process

_DEFAULT_FLAGS = (
    "--nodeterministic_rng",
    "--host_ordinal=0",
    "--jax_enable_compilation_cache=false",
    "--alsologtostderr",
)

_DEFAULT_GPU_FLAGS = []

_DEFAULT_TPU_FLAGS = []

_SUPPORTED_GPU_PLATFORM_CHIPS = frozenset(
    ["a100", "p100", "v100", "l4", "t4", "h100"])
_SUPPORTED_TPU_PLATFORM_CHIPS = frozenset(
    ["tpuv2", "tpuv3", "tpuv4", "tpuv4i", "tpuv5e"]
)


@dataclasses.dataclass
class SAXRunOpts:
  """Options for running SAX model server.

  Attributes:
    admin_port: SAX admin server port. Defaults to 10000.
    grpc_port: SAX model server RPC port. Default to 14002.
    stubby_port: SAX model server stubby port. Default to 14004.
    jax_profiler_port: JAX profiler port.
    sax_cell: SAX cell of the admin server. Default to /sax/ulm.
    sax_model_serving_path: Path to SAX model serving binary.
    platform_chip: chip name.
    platform_topology: topology description.
    sax_extra_args: additional flags for SAX model server.
  """
  admin_port: Optional[int] = 10000
  grpc_port: Optional[int] = 14002
  stubby_port: Optional[int] = 14004
  jax_profiler_port: Optional[int] = None
  sax_cell: str = "/sax/ulm"
  sax_model_serving_path: Optional[str] = None
  platform_chip: Optional[str] = "a100"
  platform_topology: Optional[str] = "1"
  sax_extra_args: Optional[List[str]] = None

  def is_platform_chip_gpu(self) -> bool:
    return self.platform_chip in _SUPPORTED_GPU_PLATFORM_CHIPS

  def is_platform_chip_tpu(self) -> bool:
    return self.platform_chip in _SUPPORTED_TPU_PLATFORM_CHIPS

  def get_cmd_list(self) -> Sequence[str]:
    """Builds a command to start SAX Model Server binary.

    Returns:
      Shell command to execute.
    """
    if self.platform_chip == "cpu":
      return self._get_cpu_cmd_list()
    elif self.is_platform_chip_gpu():
      return self._get_gpu_cmd_list()
    elif self.is_platform_chip_tpu():
      return self._get_tpu_cmd_list()
    else:
      raise ValueError("Platform is not supported: " + self.platform_chip)

  def _get_common_cmd_args(self) -> Sequence[str]:
    """Builds a command to start SAX Model Server binary.

    Returns:
      Shell command to execute.
    """
    cmd_args = []
    cmd_args += _DEFAULT_FLAGS
    cmd_args += [
        f"--platform_chip={self.platform_chip}",
        f"--platform_topology={self.platform_topology}",
        f"--port={self.grpc_port}",
        f"--admin_port={self.admin_port}",
        f"--sax_cell={self.sax_cell}",
    ]

    if self.jax_profiler_port is not None:
      cmd_args.append(
          f"--jax_profiler_port={self.jax_profiler_port}",
      )

    if self.sax_extra_args:
      cmd_args += self.sax_extra_args
    return cmd_args

  def _get_cpu_cmd_list(self) -> Sequence[str]:
    """Builds a command to start CPU SAX Model Server binary.

    Returns:
      Shell command to execute.
    """
    cmd_list = [
        self.sax_model_serving_path if self.sax_model_serving_path else
        constants.DEFAULT_SAX_SERVING_PATH]

    cmd_list += self._get_common_cmd_args()
    return cmd_list

  def _get_gpu_cmd_list(self) -> Sequence[str]:
    """Builds a command to start GPU SAX Model Server binary.

    Returns:
      Shell command to execute.
    """
    cmd_list = [
        self.sax_model_serving_path if self.sax_model_serving_path else
        constants.DEFAULT_SAX_SERVING_PATH]

    cmd_list += _DEFAULT_GPU_FLAGS
    cmd_list += self._get_common_cmd_args()
    return cmd_list

  def _get_tpu_cmd_list(self) -> Sequence[str]:
    """Builds a command to start TPU SAX Model Server binary.

    Returns:
      Shell command to execute.
    """
    cmd_list = [
        self.sax_model_serving_path if self.sax_model_serving_path else
        constants.DEFAULT_SAX_SERVING_PATH]

    cmd_list += _DEFAULT_TPU_FLAGS
    cmd_list += self._get_common_cmd_args()
    return cmd_list


class SAXModelServer:
  """SAX serving server process."""

  def __init__(self, shutdown_callback: Callable[[int], None]):
    self._lock = threading.Lock()
    self._popen = None
    self._shutdown_callback = shutdown_callback

  def run(self, run_opts: SAXRunOpts) -> subprocess.Popen[AnyStr]:
    logging.info("Launching SAX Serving.\n")
    logging.info("args: %s", run_opts.get_cmd_list())
    self._lock = threading.Lock()
    with self._lock:
      sax_process = tornado_process.Subprocess(run_opts.get_cmd_list())
      self._popen = sax_process.proc
      logging.info("SAX Model Server pid=%d", self._popen.pid)
      sax_process.set_exit_callback(self._shutdown_callback)
    return self._popen

  def wait(self) -> int:
    """Wait for server to terminate.

    Returns:
      Process exit code.

    Raises:
      ValueError if process is not initialized.
    """
    popen = None
    with self._lock:
      popen = self._popen
    if popen is None:
      raise ValueError("Popen is not initialized.")
    exit_code = popen.wait()
    logging.info("SAX terminated with exit code: %d.", exit_code)
    return exit_code
