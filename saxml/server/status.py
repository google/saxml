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
"""Status helpers."""

import dataclasses

import grpc


@dataclasses.dataclass
class Status:
  code: grpc.StatusCode
  details: str = ''

  def ok(self) -> bool:
    return self.code == grpc.StatusCode.OK

  def status_code(self) -> grpc.StatusCode:
    return self.code


def ok() -> Status:
  return Status(grpc.StatusCode.OK)


def cancelled(errmsg: str = '') -> Status:
  return Status(grpc.StatusCode.CANCELLED, errmsg)


def invalid_arg(errmsg: str) -> Status:
  return Status(grpc.StatusCode.INVALID_ARGUMENT, errmsg)


def internal_error(errmsg: str) -> Status:
  return Status(grpc.StatusCode.INTERNAL, errmsg)


def not_found(errmsg: str) -> Status:
  return Status(grpc.StatusCode.NOT_FOUND, errmsg)


def permission_denied(errmsg: str) -> Status:
  return Status(grpc.StatusCode.PERMISSION_DENIED, errmsg)


def resource_exhausted(errmsg: str) -> Status:
  return Status(grpc.StatusCode.RESOURCE_EXHAUSTED, errmsg)


def unimplemented(errmsg: str) -> Status:
  return Status(grpc.StatusCode.UNIMPLEMENTED, errmsg)


def already_exists(errmsg: str) -> Status:
  return Status(grpc.StatusCode.ALREADY_EXISTS, errmsg)


def unavailable(errmsg: str) -> Status:
  return Status(grpc.StatusCode.UNAVAILABLE, errmsg)
