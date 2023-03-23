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
"""The main module of model services."""

import re
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging
import grpc
import jax
import jax.experimental.maps  # needed by experimental_xmap_spmd_lowering* below
from saxml.protobuf import modelet_pb2
from saxml.protobuf import modelet_pb2_grpc
from saxml.server import model_service_base
from saxml.server import servable_model_registry
from saxml.server import spmd_backend
import tensorflow as tf

_SAX_CELL = flags.DEFINE_string(
    'sax_cell',
    None,
    'Optional SAX cell of the admin server. If set, heartbeat is enabled.',
)
_MODEL_FILTER_REGEX = flags.DEFINE_string(
    'model_filter_regex',
    None,
    'A regex to filter (full match) models in the registry by their names.',
)
_ADMIN_PORT = flags.DEFINE_integer(
    'admin_port', None, 'Optional port for the built-in admin server.'
)

_PORT = flags.DEFINE_integer(
    'port', None, 'Port for the RPC service.', required=True
)
_PLATFORM_CHIP = flags.DEFINE_string(
    'platform_chip', None, 'Optional chip name.'
)
_PLATFORM_TOPOLOGY = flags.DEFINE_string(
    'platform_topology', None, 'Optional topology description.'
)
_JAX_PROFILER_PORT = flags.DEFINE_integer(
    'jax_profiler_port',
    None,
    (
        'If set, the jax.profiler port to use. Only needed for profiling in'
        ' open source.'
    ),
)

# Internal tuning knobs. Consult sax-dev@ before tweaking these.
_MODELS = flags.DEFINE_list(
    'models', [], 'Optional model paths to load at startup time.'
)
_MODEL_KEYS = flags.DEFINE_list(
    'model_keys', [], 'Optional keys to identify loaded models at startup time.'
)
_CHECKPOINTS = flags.DEFINE_list(
    'checkpoints', [], 'Optional model checkpoints to load at startup time.'
)
_DETERMINISTIC_RNG = flags.DEFINE_bool(
    'deterministic_rng',
    False,
    'Whether to use a fixed RNG seed for all models.',
)
_HOST_ORDINAL = flags.DEFINE_integer(
    'host_ordinal',
    None,
    (
        'Ordinal of the current host in a multi-host setup. Host 0 is the'
        ' worker server that handles requests, and others will run the'
        ' secondary worker loop.'
    ),
)


@flags.multi_flags_validator(
    ['models', 'model_keys', 'checkpoints'],
    message='models, model_keys, and checkpoints must have the same length',
)
def _check_model_checkpoint_flags(flags_dict):
  return len(flags_dict['models']) == len(flags_dict['checkpoints']) and (
      len(flags_dict['models']) == len(flags_dict['model_keys'])
  )


def _load_static_model(
    port,
    model: str,
    model_key: str,
    checkpoint: str,
    channel_creds: Optional[grpc.ChannelCredentials],
) -> None:
  """Loads statically specified model to a started service."""
  logging.info(
      'Loading key %s, model %s, checkpoint %s.', model_key, model, checkpoint
  )
  if channel_creds is None:
    channel = grpc.insecure_channel(f'localhost:{port}')
  else:
    channel = grpc.secure_channel(f'localhost:{port}', channel_creds)
  with channel:
    grpc.channel_ready_future(channel).result(timeout=10)
    stub = modelet_pb2_grpc.ModeletStub(channel)
    req = modelet_pb2.LoadRequest(
        model_key=model_key, model_path=model, checkpoint_path=checkpoint
    )
    try:
      stub.Load(req)
    except grpc.RpcError as e:
      logging.exception('Exception during loading: %s', e)


def setup_jax(
    globally_use_hardware_rng: bool,
    jax_backend_target: Optional[str],
    jax_xla_backend: Optional[str],
    jax_enable_checks: bool,
) -> None:
  """Setups JAX and logs information about this job."""
  # Hide any GPUs from TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.set_visible_devices([], 'GPU')
  if globally_use_hardware_rng:
    jax.config.update('jax_default_prng_impl', 'rbg')

  # Log tracing and compilation time.
  jax.config.update('jax_log_compiles', True)
  # We use xmap only with SPMD.
  jax.config.update('experimental_xmap_spmd_lowering', True)
  # Use the manual partitioning lowering of xmap to avoid vectorization.
  jax.config.update('experimental_xmap_spmd_lowering_manual', True)

  if jax_enable_checks:
    jax.config.update('jax_enable_checks', True)
    logging.info('jax_enable_checks has been enabled.')

  if jax_backend_target:
    logging.info('Using JAX backend target %s', jax_backend_target)
    jax_xla_backend = 'None' if jax_xla_backend is None else jax_xla_backend
    logging.info('Using JAX XLA backend %s', jax_xla_backend)

  # LINT.IfChange
  # Initialize distributed jax in OSS
  # LINT.ThenChange(//depot/google3/third_party/py/paxml/export/copy.bara.sky)

  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX devices: %r', jax.devices())
  logging.info('jax.device_count(): %d', jax.device_count())
  logging.info('jax.local_device_count(): %d', jax.local_device_count())
  logging.info('jax.process_count(): %d', jax.process_count())


def set_up():
  """Sets up the server."""
  setup_jax(
      globally_use_hardware_rng=True,
      jax_backend_target=flags.FLAGS.jax_backend_target,
      jax_xla_backend=flags.FLAGS.jax_xla_backend,
      jax_enable_checks=flags.FLAGS.jax_enable_checks,
  )


def run(channel_creds: Optional[grpc.ChannelCredentials]) -> None:
  """Runs the server until it is stopped."""
  jax.monitoring.record_event('/jax/sax/model_service/run')
  if _MODEL_FILTER_REGEX.value is not None:
    logging.info('Setting model filter to %s', _MODEL_FILTER_REGEX.value)
    servable_model_registry.MODEL_FILTER_REGEX = re.compile(
        _MODEL_FILTER_REGEX.value
    )
  set_up()
  if _HOST_ORDINAL.value is None:
    is_primary = jax.process_index() == 0
  else:
    is_primary = _HOST_ORDINAL.value == 0
  seed = 1234 if _DETERMINISTIC_RNG.value else None

  if jax.process_count() > 1:
    from saxml.server.jax import jax_spmd_backend  # pylint: disable=g-import-not-at-top

    spmd_bknd = jax_spmd_backend.JaxSPMDBackend()
  else:
    spmd_bknd = spmd_backend.SingleHostBackend()

  runner = model_service_base.ModelServicesRunner(
      is_primary_process=is_primary,
      port=_PORT.value,
      deterministic_prng_seed=seed,
      sax_cell=_SAX_CELL.value,
      admin_port=_ADMIN_PORT.value,
      platform_chip=_PLATFORM_CHIP.value,
      platform_topology=_PLATFORM_TOPOLOGY.value,
      spmd_backend=spmd_bknd,
  )
  # Start jax.profiler for TensorBoard and profiling in open source.
  if _JAX_PROFILER_PORT.value:
    jax.profiler.start_server(_JAX_PROFILER_PORT.value)
  try:
    logging.info('Starting runner %d.', jax.process_index())
    runner.start()
    if is_primary:
      for model, key, ckpt in zip(
          _MODELS.value, _MODEL_KEYS.value, _CHECKPOINTS.value
      ):
        _load_static_model(_PORT.value, model, key, ckpt, channel_creds)
    runner.wait()
  finally:
    runner.stop()


def main(argv: Sequence[str]) -> None:
  del argv
  # TODO(sax-dev): Add secure channel for OSS.
  run(None)


if __name__ == '__main__':
  jax.config.config_with_absl()
  app.run(main)
