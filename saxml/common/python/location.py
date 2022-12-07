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

"""Admin location API."""

from saxml.common.python import pybind_location


def Join(sax_cell: str, ip_port: str, specs: bytes) -> None:
  """Join is called by model servers to join the admin server in a SAX cell.

  A background address watcher starts running indefinitely on successful calls.
  This address watcher will attempt to join initially after a small delay
  and then periodically, as well as whenever the admin server address changes
  if the platform supports address watching.

  Args:
    sax_cell: The Sax cell to join, e.g. /sax/test.
    ip_port: The IP:port of the joining model server.
    specs: Serialized ModelServer proto.

  Raises:
    RuntimeError: The caller failed to join the admin server.
  """
  result: str = pybind_location.Join(sax_cell, ip_port, specs)
  if result:
    raise RuntimeError(result)
