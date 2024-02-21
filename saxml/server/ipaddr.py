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
import os
import socket


def MyIPAddr() -> str:
  """Returns the IP address of this process reachable by others."""
  debug_ipaddr = os.environ.get("SAX_DEBUG_IPADDR", "")
  if debug_ipaddr == "1":
    return "127.0.0.1"

  hostname = socket.gethostname()

  try:
    addrs = socket.getaddrinfo(hostname, None, socket.AF_INET6)
    if addrs:
      addr = addrs[0][4][0]
      ip = ipaddress.ip_address(addr)
      if not ip.is_link_local and (ip.is_private or ip.is_global):
        return addr
  except socket.error:
    pass

  try:
    addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)
    if addrs:
      addr = addrs[0][4][0]
      ip = ipaddress.ip_address(addr)
      if not ip.is_link_local and (ip.is_private or ip.is_global):
        return addr
  except socket.error:
    pass

  return "localhost"


def Join(ip: str, port: int) -> str:
  """Returns an IP address joined with a port."""
  if not ip.startswith("[") and ":" in ip:
    return "[%s]:%d" % (ip, port)
  else:
    return "%s:%d" % (ip, port)
