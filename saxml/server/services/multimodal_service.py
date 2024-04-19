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
"""RPC service multimodal model inference."""

from typing import Any

from saxml.protobuf import multimodal_pb2
from saxml.protobuf import multimodal_pb2_grpc
from saxml.server import model_service_base

SERVICE_ID = 'mm'


class MultimodalMethodName:
  GENERATE = 'mm.generate'
  SCORE = 'mm.score'
  GENERATE_STREAM = 'mm.generate_stream'
  EMBED = 'mm.embed'


class MultimodalService(model_service_base.ModelService):
  """Shared MultimodalService implementation for different RPC backends."""

  def ParseMethodRPCRequest(self, method_name: str, request: Any) -> Any:
    if method_name == MultimodalMethodName.GENERATE:
      return request.request
    if method_name == MultimodalMethodName.SCORE:
      return request.request
    if method_name == MultimodalMethodName.EMBED:
      return request.request
    raise NotImplementedError(f'Method {method_name} unimplemented.')

  def FillRPCResponse(
      self, method_name: str, method_outputs: Any, response: Any
  ) -> None:
    if method_name == MultimodalMethodName.GENERATE:
      response.response.CopyFrom(method_outputs)
      return
    if method_name == MultimodalMethodName.SCORE:
      response.response.CopyFrom(method_outputs)
      return
    if method_name == MultimodalMethodName.EMBED:
      response.response.CopyFrom(method_outputs)
      return
    raise NotImplementedError(f'Method {method_name} unimplemented.')


@model_service_base.register_service(SERVICE_ID)
class MultimodalServiceGRPC(
    model_service_base.ModelServiceGRPC,
    MultimodalService,
    multimodal_pb2_grpc.MultimodalServiceServicer,
):
  """MultimodalService gRPC service."""

  def ServiceName(self) -> str:
    return multimodal_pb2.DESCRIPTOR.services_by_name[
        'MultimodalService'
    ].full_name

  def AddToServer(self, server: Any) -> None:
    multimodal_pb2_grpc.add_MultimodalServiceServicer_to_server(self, server)

  async def Generate(self, request, context):
    resp = multimodal_pb2.GenerateRpcResponse()
    await self.EnqueueRequest(
        MultimodalMethodName.GENERATE, request.model_key, context, request, resp
    )
    return resp

  async def Score(self, request, context):
    resp = multimodal_pb2.ScoreRpcResponse()
    await self.EnqueueRequest(
        MultimodalMethodName.SCORE, request.model_key, context, request, resp
    )
    return resp

  async def Embed(self, request, context):
    resp = multimodal_pb2.EmbedRpcResponse()
    await self.EnqueueRequest(
        MultimodalMethodName.EMBED, request.model_key, context, request, resp
    )
    return resp
