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
"""Launcher for SAX Model Server, HTTP and GRPC servers."""

import asyncio
import os
import subprocess
import sys
import threading
import time
from typing import Sequence

from absl import flags
from absl import logging
from saxml.client.python import sax
from saxml.protobuf import admin_pb2
from saxml.vertex import constants
from saxml.vertex import grpc_prediction_server
from saxml.vertex import http_prediction_server
from saxml.vertex import sax_model_server
import tornado.ioloop
import tornado.web


_PORT = flags.DEFINE_integer(
    name="port",
    default=None,
    help=(
        "The port on which to run the HTTP Prediction server. When running on"
        " Vertex Prediction, should be overridden by AIP_HTTP_PORT. HTTP"
        " service won't be started unless port is specified."
    ),
    required=False,
)

_GRPC_PREDICTION_PORT = flags.DEFINE_integer(
    name="grpc_prediction_port",
    default=None,
    help=(
        "The port on which to run the gRPC Prediction server. When running on"
        " Vertex Prediction, should be overridden by AIP_GRPC_PORT. gRPC"
        " service won't be started unless grpc_prediction_port is specified."
    ),
    required=False,
)

_ADMIN_PORT = flags.DEFINE_integer(
    name="admin_port",
    default=10000,
    help="port for the built-in SAX admin server.",
    required=False,
)

_GRPC_PORT = flags.DEFINE_integer(
    name="grpc_port",
    default=14002,
    help="Port for the RPC service.",
    required=False)

_JAX_PROFILER_PORT = flags.DEFINE_integer(
    name="jax_profiler_port",
    default=None,
    help="Port for the JAX profiler service.",
    required=False)

_MODEL_KEY = flags.DEFINE_string(
    name="model_key",
    default="/sax/ulm/model",
    help=(
        "Model key of form /sax/<cell>/<model> to identifying the model to"
        " load."
    ),
    required=False,
)

_MODEL_PATH = flags.DEFINE_string(
    name="model_path",
    default=None,
    help=(
        "Path of the model in Sax's model registry linked in the server binary."
    ),
    required=False,
)

_CKPT_GCS_PREFIX = flags.DEFINE_string(
    name="ckpt_gcs_prefix",
    default="",
    help=(
        "GCS bucket that stores checkpoint."
        "AIP_STORAGE_URI environment variable takes a presedence."
    ),
    required=False,
)

_CKPT_PATH_SUFFIX = flags.DEFINE_string(
    name="ckpt_path_suffix",
    default="",
    help=(
        "Relative path of the checkpoint in the form of `/checkpoint_<steps>` "
        "in GCS. Prefix bucket should be defined by artifact uri. "
        "See requirement in "
        "https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#artifacts"
    ),
    required=False,
)

_SAX_CELL = flags.DEFINE_string(
    name="sax_cell",
    default="/sax/ulm",
    help="SAX cell of the admin server. If set, heartbeat is enabled.",
    required=False,
)

_SAX_ROOT = flags.DEFINE_string(
    name="sax_root",
    default="/tmp/sax-test-root/",
    help=(
        "SAX root of the admin server. SAX_ROOT env variable takes precedence "
        "over sax_root flag as we configure SAX admin server using env variable"
        " in cloud environment."
    ),
    required=False,
)

_SAX_FS_ROOT = flags.DEFINE_string(
    name="sax_fs_root",
    default="/tmp/sax-fs-root/",
    help="SAX fs root of the admin server",
    required=False,
)

_PLATFORM_CHIP = flags.DEFINE_string(
    name="platform_chip",
    default=None,
    help="Optional chip name.",
    required=False,
)

_PLATFORM_TOPOLOGY = flags.DEFINE_string(
    name="platform_topology",
    default=None,
    help="Optional topology description.",
    required=False,
)

_SAX_SERVING_BINARY_PATH = flags.DEFINE_string(
    name="sax_serving_binary_path",
    required=False,
    default=None,
    help="Path to SAX Serving binary",
)

_SAX_ADMIN_CONFIG_BINARY_PATH = flags.DEFINE_string(
    name="sax_admin_config_binary_path",
    required=False,
    default=constants.DEFAULT_ADMIN_CONFIG_BINARY_PATH,
    help="Path to SAX Admin Config binary",
)

_SAX_MODEL_SERVER_EXTRA_ARGS = flags.DEFINE_multi_string(
    "sax_extra_flag",
    default=[],
    help=(
        "Optional additional flags to pass to SAX Model Server."
        "Each flag is passed as a separate argument:"
        "--sax_extra_flag='--vmodule=*profiler*=2'"
        "--sax_extra_flag='--undefok=profiler'"
    ),
)

_PREDICTION_TIMEOUT_SECONDS = flags.DEFINE_integer(
    name="prediction_timeout_seconds",
    required=False,
    default=constants.DEFAULT_PREDICTION_TIMEOUT_SECONDS,
    help="Query timeout constant for SAX model server.",
)

_PUBLISH_RETRIES = 20
_PUBLISH_RETRY_DELAY_SECONDS = 30
_TPU_WORKER_HOSTNAMES_ENV_VAR = "TPU_WORKER_HOSTNAMES"
_TPU_WORKER_ID = "TPU_WORKER_ID"


def _get_worker_id() -> int:
  """Returns worker id."""
  return int(os.getenv(_TPU_WORKER_ID, "0"))


def _is_primary() -> bool:
  """Returns true if container is running on primary node."""
  return _get_worker_id() == 0


def _get_admin_server_address() -> str:
  """Returns admin server address."""
  tpu_worker_hostnames = os.getenv(_TPU_WORKER_HOSTNAMES_ENV_VAR, "")
  if _is_primary() or not tpu_worker_hostnames:
    return f"localhost:{_ADMIN_PORT.value}"
  else:
    tpu_worker_0 = tpu_worker_hostnames.split(",")[0]
    return f"{tpu_worker_0}:{_ADMIN_PORT.value}"


def get_admin_config_cmd_list() -> Sequence[str]:
  """return cmd for admin config subprocess."""
  sax_cell = _SAX_CELL.value
  # SAX_ROOT env var is required
  sax_root = os.getenv("SAX_ROOT", _SAX_ROOT.value)
  sax_fs_root = _SAX_FS_ROOT.value
  cmd_list = [
      f"{_SAX_ADMIN_CONFIG_BINARY_PATH.value}",
      f"--sax_cell={sax_cell}",
      f"--sax_root={sax_root}",
      f"--fs_root={sax_fs_root}",
      "--alsologtostderr",
  ]
  return cmd_list


def publish_model(model_key: str, model_path: str):
  """Publish SAX model with retry."""

  ckpt_path = os.path.join(
      os.getenv("AIP_STORAGE_URI", _CKPT_GCS_PREFIX.value),
      _CKPT_PATH_SUFFIX.value)

  if not ckpt_path.strip():
    ckpt_path = None

  logging.info(
      "Model %s is being published with checkpoint %s", model_key, ckpt_path
  )
  for retry in range(0, _PUBLISH_RETRIES):
    try:
      sax.Publish(model_key, model_path, ckpt_path, 1)
      return True
    except Exception as err:  # pylint: disable=broad-except
      logging.warning("Error publishing model %s on retry %d", err, retry)
      logging.warning("Error %s type %s", str(err), type(retry))
      time.sleep(_PUBLISH_RETRY_DELAY_SECONDS)
  logging.error("Failed to publish model %s after %d retries",
                model_key, _PUBLISH_RETRY_DELAY_SECONDS)
  return False


def _shutdown(return_code: int) -> None:
  logging.info("subprocess exit with return code: %d", -return_code)
  # Stop tornado ioloop once SAX model server stoped.
  loop = tornado.ioloop.IOLoop.current()
  if loop:
    loop.stop()


def configure_admin_server():
  """Run admin_config binary to config SAX admin server."""
  if _is_primary():
    logging.info("Configuring SAX admin server on primary node")
    admin_config_cmd_list = get_admin_config_cmd_list()
    admin_config_process = subprocess.Popen(admin_config_cmd_list)
    logging.info(
        "Configuring SAX admin server pid=%d",
        admin_config_process.pid)
    exit_code = admin_config_process.wait()
    if exit_code != 0:
      logging.info(
          "Configuring SAX admin server failed with exit code: %d.", exit_code
      )
      os._exit(exit_code)  # pylint: disable=protected-access
    else:
      logging.info("Successfully configured SAX admin server.")
  else:
    tpu_worker_hostnames = os.getenv(_TPU_WORKER_HOSTNAMES_ENV_VAR, "")
    if not tpu_worker_hostnames:
      logging.warning("%s is empty.", _TPU_WORKER_HOSTNAMES_ENV_VAR)
      return
    sax_root = os.getenv("SAX_ROOT", _SAX_ROOT.value)
    sax_cell_path = sax_root.rstrip("/") + "/" + _SAX_CELL.value.lstrip("/")
    location = admin_pb2.Location(location=_get_admin_server_address())
    os.makedirs(sax_cell_path, exist_ok=True)
    location_file_path = os.path.join(sax_cell_path, "location.proto")
    logging.info("Manually updating %s to point to primary node %s",
                 location_file_path, location.location)
    with open(location_file_path, "wb") as w:
      w.write(location.SerializeToString())


def launch_sax_model_server():
  """Start SAX model server."""
  run_opts = sax_model_server.SAXRunOpts(
      worker_id=_get_worker_id(),
      admin_port=_ADMIN_PORT.value,
      grpc_port=_GRPC_PORT.value,
      sax_cell=_SAX_CELL.value,
      sax_model_serving_path=_SAX_SERVING_BINARY_PATH.value,
      platform_chip=_PLATFORM_CHIP.value,
      platform_topology=_PLATFORM_TOPOLOGY.value,
      jax_profiler_port=_JAX_PROFILER_PORT.value,
      sax_extra_args=_SAX_MODEL_SERVER_EXTRA_ARGS.value,
  )

  sax_server = sax_model_server.SAXModelServer(_shutdown)
  sax_server.run(run_opts)
  logging.info("SAX server started.")
  return sax_server


def _get_prediction_service_ports():
  """Resolve and validate port value.

  Returns:
    http_port for http prediction service.
    grpc_port for grpc prediction service.
  """

  # AIP_HTTP_PORT specifies the http port used by Vertex Prediction.
  # (https://cloud.google.com/vertex-ai/docs/predictions/custom-container-requirements#aip-variables)
  http_port = (
      _PORT.value
      if not os.environ.get("AIP_HTTP_PORT", "")
      else int(os.environ.get("AIP_HTTP_PORT"))
  )

  # AIP_GRPC_PORT specify the grpc port used by Vertex Prediction.
  # go/vertex-grpc-unary
  grpc_port = (
      _GRPC_PREDICTION_PORT.value
      if not os.environ.get("AIP_GRPC_PORT", "")
      else int(os.environ.get("AIP_GRPC_PORT"))
  )

  if not http_port and not grpc_port:
    raise ValueError(
        "At least one of the http/grpc port should be enabled. "
        "If on Vertex Prediction, specify through AIP environment variables."
    )

  return http_port, grpc_port


async def main():
  logging.set_verbosity(logging.INFO)

  try:
    http_port, grpc_port = _get_prediction_service_ports()
    http_prediction_thread = None
    grpc_prediction_thread = None
    if http_port:
      # Need to start HTTP server on all nodes before SAX.
      logging.info("Starting HTTP server at port: %d", http_port)
      # HTTP server has to be started before SAX server due to:
      # https://github.com/kubernetes-sigs/lws/issues/85
      http_prediction_thread = threading.Thread(
          target=http_prediction_server.run,
          args=(
              http_port,
              _get_admin_server_address(),
              _MODEL_KEY.value,
              _PREDICTION_TIMEOUT_SECONDS.value,
          ))
      http_prediction_thread.start()
      logging.info("done initializing HTTP server")

    if _is_primary() and grpc_port:
      grpc_prediction_thread = threading.Thread(
          target=grpc_prediction_server.run,
          args=(
              grpc_port,
              _MODEL_KEY.value,
              _PREDICTION_TIMEOUT_SECONDS.value
          ))
      grpc_prediction_thread.start()
      logging.info("done initializing gRPC server")

    configure_admin_server()
    sax_server = launch_sax_model_server()

    if _is_primary():
      # Only publish model using SAX client on primary node.
      if not publish_model(_MODEL_KEY.value, _MODEL_PATH.value):
        logging.error("Failed to publish model %s from %s",
                      _MODEL_KEY.value, _MODEL_PATH.value)
        os._exit(-1)  # pylint: disable=protected-access

    sax_server.wait()
    if http_prediction_thread:
      http_prediction_thread.join()
    if grpc_prediction_thread:
      grpc_prediction_thread.join()
  except Exception as e:  # pylint: disable=broad-except
    logging.exception("Caught exception: %s", e)
    # Need to use os._exit(), instead of sys.exit(), to really exit the
    # launcher and shutdown the whole container because at least one of the
    # launched processes run as non-daemon.
    os._exit(-1)  # pylint: disable=protected-access

if __name__ == "__main__":
  flags.FLAGS(sys.argv, known_only=True)
  asyncio.run(main())
