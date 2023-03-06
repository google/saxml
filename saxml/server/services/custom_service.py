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
"""RPC service Custom inference."""

from typing import Any

from saxml.protobuf import custom_pb2
from saxml.protobuf import custom_pb2_grpc
from saxml.server import model_service_base

SERVICE_ID = 'custom'


class CustomService(model_service_base.ModelService):
  """CustomService implementation."""

  def ParseMethodRPCRequest(
      self, method_name: str, request: custom_pb2.CustomRequest
  ) -> bytes:
    return request.request

  def FillRPCResponse(self, method_name: str, method_outputs: bytes,
                      response: custom_pb2.CustomResponse) -> None:
    response.response = method_outputs
    return


@model_service_base.register_service(SERVICE_ID)
class CustomServiceGRPC(
    model_service_base.ModelServiceGRPC,
    CustomService,
    custom_pb2_grpc.CustomServiceServicer,
):
  """CustomService gRPC service."""

  def ServiceName(self) -> str:
    return custom_pb2.DESCRIPTOR.services_by_name['CustomService'].full_name

  def AddToServer(self, server: Any) -> None:
    custom_pb2_grpc.add_CustomServiceServicer_to_server(self, server)

  async def Custom(self, request, context):
    resp = custom_pb2.CustomResponse()
    await self.EnqueueRequest(
        request.method_name, request.model_key, context, request, resp
    )
    return resp
