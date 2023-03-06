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
"""RPC service vision model inference."""

from typing import Any

import numpy as np
from saxml.protobuf import vision_pb2
from saxml.protobuf import vision_pb2_grpc
from saxml.server import model_service_base

SERVICE_ID = 'vm'


class VisionMethodName:
  CLASSIFY = 'vm.classify'
  TEXT_TO_IMAGE = 'vm.generate'
  EMBED = 'vm.embed'
  DETECT = 'vm.detect'
  IMAGE_TO_TEXT = 'vm.image_to_text'
  VIDEO_TO_TEXT = 'vm.video_to_text'


class VisionService(model_service_base.ModelService):
  """Shared VisionService implementation for differen RPC backends."""

  def ParseMethodRPCRequest(self, method_name: str, request: Any) -> Any:
    if method_name == VisionMethodName.CLASSIFY:
      return {'image_bytes': np.array(request.image_bytes, dtype=object)}
    if method_name == VisionMethodName.TEXT_TO_IMAGE:
      return request.text
    if method_name == VisionMethodName.EMBED:
      return {'image_bytes': np.array(request.image_bytes, dtype=object)}
    if method_name == VisionMethodName.DETECT:
      return {
          'image_bytes': np.array(request.image_bytes, dtype=object),
          'text': np.array(request.text),
      }
    if method_name == VisionMethodName.IMAGE_TO_TEXT:
      return {
          'image_bytes': np.array(request.image_bytes, dtype=object),
          'text': np.array(request.text),
      }
    if method_name == VisionMethodName.VIDEO_TO_TEXT:
      return {
          'image_frames': list(request.image_frames),
          'text': np.array(request.text),
      }
    raise NotImplementedError(f'Method {method_name} unimplemented.')

  def FillRPCResponse(
      self, method_name: str, method_outputs: Any, response: Any
  ) -> None:
    if method_name == VisionMethodName.CLASSIFY:
      # Convert tuple of labels / scores to output format.
      texts, scores = method_outputs
      for text, score in zip(texts, scores):
        response.texts.append(vision_pb2.DecodedText(text=text, score=score))
      return
    if method_name == VisionMethodName.TEXT_TO_IMAGE:
      images, scores = method_outputs
      for image, score in zip(images, scores):
        response.images.append(
            vision_pb2.ImageGenerations(image=image, score=score)
        )
      return
    if method_name == VisionMethodName.EMBED:
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
    if method_name == VisionMethodName.DETECT:
      boxes, scores, texts = method_outputs
      for box, score, text in zip(boxes, scores, texts):
        response.bounding_boxes.append(
            vision_pb2.BoundingBox(
                cx=box[0], cy=box[1], w=box[2], h=box[3], text=text, score=score
            )
        )
      return
    if method_name == VisionMethodName.IMAGE_TO_TEXT:
      texts, scores = method_outputs
      for text, score in zip(texts, scores):
        response.texts.append(vision_pb2.DecodedText(text=text, score=score))
      return
    if method_name == VisionMethodName.VIDEO_TO_TEXT:
      texts, scores = method_outputs
      for text, score in zip(texts, scores):
        response.texts.append(vision_pb2.DecodedText(text=text, score=score))
      return
    raise NotImplementedError(f'Method {method_name} unimplemented.')


@model_service_base.register_service(SERVICE_ID)
class VisionServiceGRPC(
    model_service_base.ModelServiceGRPC,
    VisionService,
    vision_pb2_grpc.VisionServiceServicer,
):
  """gRPC VisionService."""

  def ServiceName(self) -> str:
    return vision_pb2.DESCRIPTOR.services_by_name['VisionService'].full_name

  def AddToServer(self, server: Any) -> None:
    vision_pb2_grpc.add_VisionServiceServicer_to_server(self, server)

  async def Classify(self, request, context):
    resp = vision_pb2.ClassifyResponse()
    await self.EnqueueRequest(
        VisionMethodName.CLASSIFY, request.model_key, context, request, resp
    )
    return resp

  async def TextToImage(self, request, context):
    resp = vision_pb2.TextToImageResponse()
    await self.EnqueueRequest(
        VisionMethodName.TEXT_TO_IMAGE,
        request.model_key,
        context,
        request,
        resp,
    )
    return resp

  async def Embed(self, request, context):
    resp = vision_pb2.EmbedResponse()
    await self.EnqueueRequest(
        VisionMethodName.EMBED, request.model_key, context, request, resp
    )
    return resp

  async def Detect(self, request, context):
    resp = vision_pb2.DetectResponse()
    await self.EnqueueRequest(
        VisionMethodName.DETECT, request.model_key, context, request, resp
    )
    return resp

  async def ImageToText(self, request, context):
    resp = vision_pb2.ImageToTextResponse()
    await self.EnqueueRequest(
        VisionMethodName.IMAGE_TO_TEXT,
        request.model_key,
        context,
        request,
        resp,
    )
    return resp

  async def VideoToText(self, request, context):
    resp = vision_pb2.VideoToTextResponse()
    await self.EnqueueRequest(
        VisionMethodName.VIDEO_TO_TEXT,
        request.model_key,
        context,
        request,
        resp,
    )
    return resp
