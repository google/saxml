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
"""Interface for the server with the backend that supports SPMD programs."""

import abc
from typing import Callable

from saxml.server import utils


class SPMDBackend(abc.ABC):
  """Interface for the server to interact with the SPMD ML framework backend."""

  @abc.abstractmethod
  def spmd_host_index(self) -> int:
    """Returns the index of the current host for SPMD programs."""

  @abc.abstractmethod
  def spmd_host_count(self) -> int:
    """Returns the number of hosts participating in the SPMD program."""

  @abc.abstractmethod
  def send_via_device(self, message: str) -> None:
    """Sends data to other hosts via the reliable device network."""

  @abc.abstractmethod
  def receive_via_device(self) -> str:
    """Receives data from a sending host via the reliable device network."""

  @abc.abstractmethod
  def receive_via_device_async(
      self, thread_pool: utils.ThreadPool, done: Callable[[str], None]
  ) -> None:
    """Receives via device communication, and calls done in a thread pool."""


class SingleHostBackend(SPMDBackend):
  """A trivial SPMDBackend that supports only 1 host."""

  def spmd_host_index(self) -> int:
    return 0

  def spmd_host_count(self) -> int:
    return 1

  def send_via_device(self, message: str) -> None:
    pass

  def receive_via_device(self) -> str:
    raise NotImplementedError('Multihost support not implemented')

  def receive_via_device_async(
      self, thread_pool: utils.ThreadPool, done: Callable[[str], None]
  ) -> None:
    raise NotImplementedError('Multihost support not implemented')
