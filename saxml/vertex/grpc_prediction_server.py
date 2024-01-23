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
"""gRPC prediction server for SAX."""

import concurrent
import logging
import multiprocessing
import queue
import threading
from typing import Iterator

import grpc
from saxml.client.python import sax
from saxml.protobuf import lm_pb2

from saxml.vertex import constants
from saxml.vertex import prediction_service_pb2
from saxml.vertex import prediction_service_pb2_grpc


class GrpcPredictionService(
    prediction_service_pb2_grpc.PredictionServiceServicer
):
  """gRPC server for SAX streaming prediction."""

  def __init__(
      self,
      server,
      model_key: str,
      user_request_timeout: int = constants.DEFAULT_PREDICTION_TIMEOUT_SECONDS,
  ):
    self._server = server
    logging.info("Starting %s gRPC server.", self.__class__.__name__)
    self._model_key = model_key
    model = sax.Model(self._model_key)
    self._lm = model.LM()
    self._user_request_timeout = user_request_timeout

  def PredictStreamed(
      self,
      request: prediction_service_pb2.PredictRequest,
      context: grpc.ServicerContext,
  ) -> Iterator[lm_pb2.GenerateStreamResponse]:
    """Handles PredictStreamed RPC.

    Arguments:
      request: lm request representing the text and extra inputs
      context: grpc.ServicerContext representing the call context.

    Yields:
      lm_pb2.GenerateStreamResponse objects according to incoming requests.
    """

    options = self._create_model_options(request)

    try:
      yield from self._generate_stream(request.text, options)
      context.set_code(grpc.StatusCode.OK)

    except Exception as e:  # pylint: disable=broad-except
      logging.exception("Caught exception: %s", e)
      context.set_code(grpc.StatusCode.INTERNAL)

  def _create_model_options(
      self, req: prediction_service_pb2.PredictRequest
  ) -> sax.ModelOptions:
    """Creates Sax model option from request."""
    option = sax.ModelOptions()
    option.SetTimeout(self._user_request_timeout)

    if hasattr(req, "extra_inputs") and req.extra_inputs:
      for key, value in req.extra_inputs.items.items():
        option.SetExtraInput(key, value)

      for key, value in req.extra_inputs.tensors.items():
        option.SetExtraInputTensor(key, value)

    return option

  def _generate_stream(
      self, text: str, option: sax.ModelOptions
  ) -> Iterator[lm_pb2.GenerateStreamResponse]:
    """Wrapper for the sax.GenerateStream call."""

    # Converting the callback into an Iterator using a thread.
    q = queue.Queue(maxsize=1)
    is_last = object()

    def callback(last: bool, response: list[tuple[str, int, list[float]]]):
      nonlocal q
      if last:
        q.put(is_last)
        return
      items = []
      for text, prefix_len, scores in response:
        items.append(
            lm_pb2.GenerateStreamItem(
                text=text, prefix_len=prefix_len, score=scores[0]
            )
        )
      q.put(items)

    def task():
      self._lm.GenerateStream(text, callback, option)

    t = threading.Thread(target=task)
    t.start()

    while True:
      # we set a timeout in case of failure
      chunk = q.get(timeout=self._user_request_timeout)
      if chunk is is_last:
        break
      yield lm_pb2.GenerateStreamResponse(items=chunk)

    t.join()


def run(grpc_port: int, model_key: str, prediction_timeout_seconds: int):
  """Run the gRPC prediction server."""
  server = grpc.server(
      thread_pool=concurrent.futures.ThreadPoolExecutor(
          max_workers=multiprocessing.cpu_count()
      ),
      options=[
          ("grpc.max_send_message_length", -1),
          ("grpc.max_receive_message_length", -1),
      ],
  )
  prediction_service_pb2_grpc.add_PredictionServiceServicer_to_server(
      GrpcPredictionService(
          server, model_key, prediction_timeout_seconds
      ),
      server,
  )

  creds = grpc.insecure_server_credentials()
  server.add_secure_port(f"0.0.0.0:{grpc_port}", creds)
  server.start()

  logging.info("gRPC server listening on port %d.", grpc_port)
  server.wait_for_termination()
  logging.info("gRPC server stopped")
