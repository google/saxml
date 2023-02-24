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
"""RPC service LM inference."""

from collections.abc import Iterable
from typing import Any

import numpy as np
from saxml.protobuf import lm_pb2
from saxml.protobuf import lm_pb2_grpc
from saxml.server import model_service_base

SERVICE_ID = 'lm'


class LMMethodName:
  SCORE = 'lm.score'
  GENERATE = 'lm.generate'
  GENERATE_STREAM = 'lm.generate_stream'
  EMBED = 'lm.embed'
  GRADIENT = 'lm.gradient'


class LmService(model_service_base.ModelService):
  """LmService implementation."""

  def ParseMethodRPCRequest(self, method_name: str, request: Any) -> Any:
    if method_name == LMMethodName.SCORE:
      return (request.prefix or '', request.suffix or [''])
    if method_name == LMMethodName.GENERATE:
      return request.text
    if method_name == LMMethodName.GENERATE_STREAM:
      return request.text
    if method_name == LMMethodName.EMBED:
      return request.text
    raise NotImplementedError(f'Method {method_name} unimplemented.')

  def FillRPCResponse(
      self, method_name: str, method_outputs: Any, response: Any
  ) -> None:
    if method_name == LMMethodName.SCORE:
      if isinstance(method_outputs, Iterable):
        response.logp.extend(method_outputs)
      else:
        response.logp.append(method_outputs)
      return
    if method_name == LMMethodName.GENERATE:
      texts, scores = method_outputs
      for text, score in zip(texts, scores):
        response.texts.append(lm_pb2.DecodedText(text=text, score=score))
      return
    if method_name == LMMethodName.GENERATE_STREAM:
      texts, scores = method_outputs
      for text, score in zip(texts, scores):
        # Let GenerateStream below add the correct value of prefix_len.
        response.items.append(lm_pb2.GenerateStreamItem(text=text, score=score))
      return
    if method_name == LMMethodName.EMBED:
      embeddings = method_outputs
      embeddings = embeddings.reshape(-1)
      if embeddings.dtype in [np.float32, np.double]:
        response.embedding.extend(list(embeddings))
      else:
        raise NotImplementedError(
            'EMBED does not support returned '
            f'embeddings of type {embeddings.dtype}.'
        )
      return

    raise NotImplementedError(f'Method {method_name} unimplemented.')


@model_service_base.register_service(SERVICE_ID)
class LmServiceGRPC(
    model_service_base.ModelServiceGRPC,
    LmService,
    lm_pb2_grpc.LMServiceServicer,
):
  """LmService gRPC service."""

  def ServiceName(self) -> str:
    return lm_pb2.DESCRIPTOR.services_by_name['LMService'].full_name

  def AddToServer(self, server: Any) -> None:
    lm_pb2_grpc.add_LMServiceServicer_to_server(self, server)

  async def Score(self, request, context):
    resp = lm_pb2.ScoreResponse()
    await self.EnqueueRequest(
        LMMethodName.SCORE, request.model_key, context, request, resp
    )
    return resp

  async def Generate(self, request, context):
    resp = lm_pb2.GenerateResponse()
    await self.EnqueueRequest(
        LMMethodName.GENERATE, request.model_key, context, request, resp
    )
    return resp

  async def GenerateStream(self, request, context):
    curr_lengths = []
    empty_resp = lm_pb2.GenerateStreamResponse()
    q = self.EnqueueStreamRequest(
        LMMethodName.GENERATE_STREAM,
        request.model_key,
        context,
        request,
        empty_resp,
    )
    while True:
      msg = await q.get()
      if msg is None:
        break

      # In this implementation, we never erase previously generated text and
      # only append new text. Therefore, track the lengths of currently
      # accumulated results and return them in `prefix_len`.
      if len(curr_lengths) < len(msg.items):
        curr_lengths += [0] * (len(msg.items) - len(curr_lengths))
      for i, item in enumerate(msg.items):
        item.prefix_len = curr_lengths[i]
        curr_lengths[i] += len(item.text)

      yield msg

  async def Embed(self, request, context):
    resp = lm_pb2.EmbedResponse()
    await self.EnqueueRequest(
        LMMethodName.EMBED, request.model_key, context, request, resp
    )
    return resp
