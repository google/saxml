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

"""Validate RPC request."""

from typing import Dict, Optional

from saxml.server import utils

from google.protobuf import message


def ValidateRequestForExtraInputs(
    req: Optional[message.Message] = None,
    extra_inputs: Optional[Dict[str, float]] = None) -> utils.Status:
  """Validate RPC request's extra_input field.

  If req has `extra_input` field, it is required that the `items` map in the
  `extra_input` field's keys is a subset of `extra_input`'s keys.


  Args:
    req: Optional RPC request message. If it is None, will return OK status.
    extra_inputs: Optional dictionary. If it is None, but `req` has
      `extra_input` field defined, will return NOT_FOUND status.

  Returns:
    If the keys in `req.extra_input.items` are a subset of `extra_input`'s keys,
    will return OK status, otherwise, will return NOT_FOUND status.
  """
  if req is None:
    return utils.ok()

  req_extra_inputs = (
      req.extra_inputs.items
      if hasattr(req, 'extra_inputs') and req.extra_inputs else None)

  if req_extra_inputs is None:
    return utils.ok()

  for input_key in req_extra_inputs:
    if extra_inputs is None or input_key not in extra_inputs:
      return utils.invalid_arg(
          f"key {input_key} in RPC request's extra_inputs field is not in"
          'ServableModel.extra_inputs. extra_inputs in ServableModel are'
          f'{extra_inputs}'
      )

  return utils.ok()
