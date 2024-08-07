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
"""Http prediction server."""

import asyncio
import concurrent
import json
import multiprocessing
import os

from absl import flags
from absl import logging
import grpc
from saxml.client.python import sax
from saxml.protobuf import admin_pb2
from saxml.protobuf import admin_pb2_grpc
from saxml.vertex import constants
from saxml.vertex import translate
import tornado.web

# Has to be true for multihost deployments due to:
# https://github.com/kubernetes-sigs/lws/issues/85
_BYPASS_HEALTH_CHECK = flags.DEFINE_bool(
    name="bypass_health_check",
    required=False,
    default=False,
    help=(
        "If true, return 200 for health check handler. This flag should only "
        "used for debugging purposes and not in production use."
    )
)

_HEALTH_CHECK_SUCCESS_REPORTED = False


class PredictHandler(tornado.web.RequestHandler):
  """HTTP handler for prediction requests."""

  # Informs pytype of type info otherwise it will throw attribute error.
  model_key: str = ...
  prediction_timeout_seconds: int = ...
  executor: concurrent.futures.ThreadPoolExecutor = ...

  def initialize(
      self,
      model_key: str = "",
      prediction_timeout_seconds: int = (
          constants.DEFAULT_PREDICTION_TIMEOUT_SECONDS)):
    self.model_key = model_key
    self.prediction_timeout_seconds = prediction_timeout_seconds
    self.executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=multiprocessing.cpu_count())

  async def _send_request(self, lm, lm_request_text, option):
    return await asyncio.get_running_loop().run_in_executor(
        self.executor, lm.Generate, lm_request_text, option
    )

  async def post(self):
    logging.debug("got prediction request")
    try:
      req = json.loads(self.request.body.decode())
      logging.debug("prediction request body: %s", req)

      lm_requests = translate.user_request_to_lm_generate_request(
          req, self.prediction_timeout_seconds
      )

      model = sax.Model(self.model_key)
      lm = model.LM()

      response = {"predictions": []}

      tasks = []
      for lm_request_text, option in lm_requests:
        tasks.append(
            asyncio.create_task(self._send_request(lm, lm_request_text, option))
        )
      lm_responses = await asyncio.gather(*tasks)

      response["predictions"].extend(lm_responses)

      self.write(json.dumps(response))
      self.set_status(200)
      logging.debug("prediction success: %s", json.dumps(response))

    except (ValueError, json.JSONDecodeError) as e:
      logging.debug("Bad request. Error is: %s", str(e))
      self.write(json.dumps({"error": str(e)}))
      self.set_status(400)

    except Exception as e:  # pylint: disable=broad-except
      logging.debug("Predict Failed. Error is: %s", str(e))
      self.write(json.dumps({"error": str(e)}))
      self.set_status(500)


class HealthHandler(tornado.web.RequestHandler):
  """Health handler for SAX."""

  # Informs pytype of type info otherwise it will throw attribute error.
  admin_server_address: str = ...
  model_key: str = ...

  def initialize(self, admin_server_address: str, model_key: str = ""):
    self.model_key = model_key
    self.admin_server_address = admin_server_address

  def get(self):
    global _HEALTH_CHECK_SUCCESS_REPORTED
    logging.debug("got health request")
    # If model loading time takes more than Vertex Prediction timeout,
    # bypass health check handler to return health before model loaded.
    if _BYPASS_HEALTH_CHECK.value:
      success_response = {
          "model_status": {
              "status": "AVAILABLE",
          }
      }
      self.write(json.dumps(success_response))
      self.set_status(200)
      logging.debug("health request _BYPASS_HEALTH_CHECK success")
      return

    try:
      channel_creds = grpc.local_channel_credentials()
      channel = grpc.secure_channel(
          self.admin_server_address, channel_creds
      )

      stub = admin_pb2_grpc.AdminStub(channel)
      wait_for_ready_request = admin_pb2.WaitForReadyRequest(
          model_id=self.model_key, num_replicas=1
      )

      stub.WaitForReady(
          wait_for_ready_request,
          timeout=constants.DEFAULT_HEALTH_CHECK_TIMEOUT_SECONDS,
      )
      success_response = {"model_status": {"status": "AVAILABLE"}}
      self.write(json.dumps(success_response))
      self.set_status(200)
      if not _HEALTH_CHECK_SUCCESS_REPORTED:
        logging.info("health request success")  # log only once
        _HEALTH_CHECK_SUCCESS_REPORTED = True
      else:
        logging.debug("health request success")
      return

    except Exception as e:  # pylint: disable=broad-except
      logging.debug("Health Check Failed. Error is: %s", str(e))
      self.write(json.dumps({"error": str(e)}))
      self.set_status(500)


def _make_app(
    admin_server_address: str,
    model_key: str,
    prediction_timeout_seconds: int) -> tornado.web.Application:
  """Makes the tornado web application.

  Args:
    admin_server_address: SAX admin address.
    model_key: model key.
    prediction_timeout_seconds: prediction timeout in seconds.

  Returns:
    Tornado web application.

  """
  # pylint: disable=g-line-too-long
  # https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements
  predict_handler = (
      os.getenv("PREDICT_ROUTE")
      or os.getenv("AIP_PREDICT_ROUTE")
      or "/predict")
  health_handler = (
      os.getenv("HEALTH_ROUTE")
      or os.getenv("AIP_HEALTH_ROUTE")
      or "/health")

  logging.info("Predict handler: %s", predict_handler)
  logging.info("Health handler: %s", health_handler)
  return tornado.web.Application([
      (
          os.getenv("PREDICT_ROUTE")
          or os.getenv("AIP_PREDICT_ROUTE")
          or "/predict",
          PredictHandler,
          dict(model_key=model_key,
               prediction_timeout_seconds=prediction_timeout_seconds),
      ),
      (
          os.getenv("HEALTH_ROUTE")
          or os.getenv("AIP_HEALTH_ROUTE")
          or "/health",
          HealthHandler,
          dict(admin_server_address=admin_server_address, model_key=model_key),
      ),
  ])


async def _run_async(
    http_port: int,
    admin_server_address: str,
    model_key: str,
    prediction_timeout_seconds: int,
):
  """Run the tornado http prediction server using asyncio."""

  logging.info("Using Tornado version: %s", str(tornado.version_info))
  webserver_app = _make_app(
      admin_server_address=admin_server_address,
      model_key=model_key,
      prediction_timeout_seconds=prediction_timeout_seconds
  )
  logging.info("tornado server listening on port %d.", http_port)
  webserver_app.listen(http_port)

  logging.info("Tornado loop started")
  await asyncio.Event().wait()
  logging.info("Tornado server stopped.")


def run(
    http_port: int,
    admin_server_address: str,
    model_key: str,
    prediction_timeout_seconds: int,
):
  """Run the tornado http prediction server blocking wrapper."""
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(_run_async(
      http_port,
      admin_server_address,
      model_key,
      prediction_timeout_seconds
  ))
  loop.close()
