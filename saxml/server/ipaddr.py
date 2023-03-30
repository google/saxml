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
"""IP address-related functions."""

import ipaddress
import socket


def MyIPAddr() -> str:
  """Returns the IP address of this process reachable by others."""
  hostname = socket.gethostname()

  try:
    address = socket.getaddrinfo(hostname, None, socket.AF_INET6)
    if address:
      addr = address[0][4][0]
      ip = ipaddress.ip_address(addr)
      if not ip.is_link_local and (ip.is_private or ip.is_global):
        return addr
  except socket.error:
    pass

  try:
    address = socket.getaddrinfo(hostname, None, socket.AF_INET)
    if address:
      addr = address[0][4][0]
      ip = ipaddress.ip_address(addr)
      if not ip.is_link_local and (ip.is_private or ip.is_global):
        return addr
  except socket.error:
    pass

  return "localhost"


def Join(ip: str, port: int) -> str:
  """Returns an IP address joined with a port."""
  if not ip.startswith("[") and ":" in ip:
    return f"[{ip}]: {port}"
  else:
    return f"{ip}: {port}" % (ip, port)
