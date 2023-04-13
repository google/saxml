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

from typing import Dict, List, Optional, Union

from saxml.server import utils

from google.protobuf import message


def ValidateRequestForExtraInputs(
    req: Optional[message.Message] = None,
    extra_inputs: Optional[Dict[str, Union[float, List[float]]]] = None,
) -> utils.Status:
  """Validate RPC request's extra_input field.

  If req has `extra_input` field, it is required that the `items` map in the
  `extra_input` field's keys is a subset of `extra_input`'s keys.


  Args:
    req: Optional RPC request message. If it is None, will return OK status.
    extra_inputs: Optional dictionary. If it is None, but `req` has
      `extra_input` field defined, will return NOT_FOUND status.

  Returns:
    If
    (1) the keys in `req.extra_input.items` are a subset of `extra_input`'s keys
    (2) the keys in `req.extra_input.tensors` are a subset of `extra_input`'s
      keys
    (3) the keys in `req.extra_input.tensors` are not a subset of
      `req.extra_input.items`'s keys
    (4) the values in `req.extra_input.tensors` are the same size a the values
      of `req.extra_input.tensors`'s keys
    will return OK status, otherwise, will return NOT_FOUND status.
  """
  if req is None:
    return utils.ok()

  req_extra_inputs = (
      dict(req.extra_inputs.items)
      if hasattr(req, 'extra_inputs') and req.extra_inputs
      else None
  )

  if req_extra_inputs is None:
    return utils.ok()

  for input_key in req_extra_inputs:
    if extra_inputs is None or input_key not in extra_inputs:
      return utils.invalid_arg(
          f'The extra_inputs key `{input_key}` in the RPC request is not'
          ' defined in ServableModel.extra_inputs. Current'
          f' ServableModel.extra_inputs is {extra_inputs}.'
      )

  for key, tensor in dict(req.extra_inputs.tensors).items():
    if extra_inputs is None or key not in extra_inputs:
      return utils.invalid_arg(
          f'The extra_inputs key `{key}` in the RPC request is not'
          ' defined in ServableModel.extra_inputs. Current'
          f' ServableModel.extra_inputs is {extra_inputs}.'
      )
    if key in req_extra_inputs:
      return utils.invalid_arg(
          'It is invalid for the same key to appear in both items and tensors.'
      )
    if not isinstance(extra_inputs[key], list):
      return utils.invalid_arg(
          f'Extra inputs `{key}` is a list but the default value is not.'
      )
    if len(list(tensor.values)) != len(extra_inputs[key]):
      return utils.invalid_arg(
          f'Extra inputs `{key}` does not have the same size as the default.'
      )

  return utils.ok()
