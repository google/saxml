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
"""Access control list support."""

from typing import Optional


def Check(aclname: Optional[str], principal: Optional[str]) -> bool:
  """Returns true iff username is allowed by the acl.

  Args:
    aclname: The access control list. The ACL is currently just a dot-separated
      list of user names.
    principal: The user name.

  Returns:
    True iff username is allowed by the acl.
  """
  if aclname is None:
    return True
  if principal is None:
    return True
  return principal in aclname.split(',')
