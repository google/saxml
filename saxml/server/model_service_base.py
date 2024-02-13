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
"""RPC service model inference."""

import abc
import asyncio
import collections
import copy
import dataclasses
import enum
import functools
import json
import queue
import threading
import time
import traceback
import typing
from typing import Any, Awaitable, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Tuple, Type
import uuid

from absl import logging
import grpc
from grpc_reflection.v1alpha import reflection
import jaxtyping as jt
import numpy as np
from saxml.common.python import location
from saxml.protobuf import admin_pb2
from saxml.protobuf import common_pb2
from saxml.protobuf import modelet_pb2
from saxml.protobuf import modelet_pb2_grpc
from saxml.server import acl
from saxml.server import ipaddr
from saxml.server import multi_host_sync
from saxml.server import proto_util
from saxml.server import servable_model
from saxml.server import servable_model_params
from saxml.server import servable_model_registry
from saxml.server import spmd_backend
from saxml.server import utils
from saxml.server import validate

from google.protobuf import message


DeviceTensors = servable_model.DeviceTensors
InputShapeInfo = servable_model.InputShapeInfo
Closure = Callable[[], None]
StatusCallback = utils.StatusCallback

Int = jt.Int
Float = jt.Float
Array = jt.Array


# Global variable for the service registry: mapping {key: list_of_services}.
# The value is a list because we allow both gRPC and Stubby services registered.
_SERVICE_REGISTRY = {}

# All SAX names must be prefixed with this.
_SAX_PREFIX = '/sax/'

# Keep warm for every 2 minutes
_KEEP_WARM_SECONDS = 120

# pylint: disable=invalid-name


def _maybe_all_cancelled(
    rpc_tasks: Sequence[utils.RpcQueueTask],
    log_msg: str,
    pool: Optional[utils.ThreadPool] = None,
) -> bool:
  """If all rpc_tasks are cancelled already, calls done() and returns True."""

  def _cancelled(rpc):
    if rpc is None:
      return False
    return rpc.should_cancel()

  def _notify():
    for t in rpc_tasks:
      t.done(utils.cancelled())

  if all(_cancelled(t.rpc) for t in rpc_tasks):
    logging.info('RPCs in batch cancelled. %s', log_msg)
    if pool is None:
      _notify()
    else:
      pool.run(_notify)
    return True
  return False


@enum.unique
class MethodName(str, enum.Enum):
  """Sax method types."""

  # Sax internal methods
  LOAD = enum.auto()
  UNLOAD = enum.auto()
  EXPORT = enum.auto()
  TERMINATE = enum.auto()
  KEEP_DEVICES_WARM = enum.auto()
  SAVE = enum.auto()

  # Model method
  MODEL = enum.auto()

  # Method with continuous batching.
  BATCHED_LM_GENERATE = enum.auto()

  # For stats and secondary host initialization
  PREFILL_INSERT = enum.auto()
  GENERATE = enum.auto()  # Secondary hosts only

  def __str__(self):
    return self.name.lower()


@dataclasses.dataclass(frozen=True)
class MethodKey:
  """Method key.

  Methods here can be from different models, and their keys are represented as a
  tuple of (method_name, model_method, service_id, model_key) in this
  module.
  """

  name: MethodName
  model_method: Optional[str] = None
  service_id: Optional[str] = None
  model_key: Optional[str] = None

  def method_name(self) -> str:
    if self.name == MethodName.MODEL:
      return self.model_method
    return str(self.name)


class Method:
  """Data structure used by PerMethodBatcher for each method."""

  model: Optional[servable_model.ServableModel] = None

  # Batch size the batcher should attempt to create.
  batch_size: int

  # Maximum number of live batches at any given point. This is currently an
  # approximation because it is enforced by checking the number of requests
  # which may not be fully batched.
  max_live_batches: int

  # The input rpc queue.
  queue: utils.RpcQueue

  # Admissioner to control the maximum (self.limit()) number of requests alive
  # at any time for this method, and handle synchronization during shutdown.
  admissioner: utils.Admissioner

  # statistic tracker.
  ok_stats: utils.RequestStats
  err_stats: utils.RequestStats

  # Sizes of recent batches.
  recent_batch_sizes: Deque[int]

  def limit(self) -> int:
    return max(self.batch_size * self.max_live_batches, 1)

  def __init__(
      self,
      model: Optional[servable_model.ServableModel],
      batch_size: int,
      max_live_batches: int,
      batching_wait_secs: Optional[float] = None,
  ):
    self.model = model
    self.batch_size = batch_size
    self.max_live_batches = max_live_batches
    self.queue = utils.RpcQueue(batching_wait_secs=batching_wait_secs)
    self.admissioner = utils.Admissioner(limit=self.limit())
    # pytype: disable=wrong-arg-types  # numpy-scalars
    self.ok_stats = utils.RequestStats(timespan_sec=60.0)
    self.err_stats = utils.RequestStats(timespan_sec=60.0)
    # pytype: enable=wrong-arg-types  # numpy-scalars
    self.recent_batch_sizes = collections.deque()

  def record_batch_size(self, size: int):
    self.recent_batch_sizes.append(size)
    if len(self.recent_batch_sizes) > 10:
      self.recent_batch_sizes.popleft()


@dataclasses.dataclass
class Batch:
  """A batch with multiple incoming requests."""

  method: MethodKey
  rpc_tasks: Sequence[utils.RpcQueueTask]
  done: Closure  # Must be called when the batch is done processing.

  input_tensors: Optional[DeviceTensors] = None
  ready: bool = False
  cv: threading.Condition = threading.Condition()
  # Whether to skip the asynchronous host-based multi-host sync, and use the
  # synchronous device-based sync only. This is used when there is no unfinished
  # batch in queue to overlap with async host sync.
  skip_host_sync: bool = False

  # Exactly one of these two fields should be filled in with a not-None value
  # before the batch is used. In addition, padded_shape is None if and only if
  # input_tensors is None.
  unpadded_shape: InputShapeInfo | None = None
  padded_shape: InputShapeInfo | None = None

  def size(self):
    return len(self.rpc_tasks)

  def mark_as_ready(self):
    with self.cv:
      self.ready = True
      self.cv.notify_all()

  def wait_for_ready(self):
    with self.cv:
      while not self.ready:
        self.cv.wait()

  def finish(self):
    assert self.done is not None
    self.done()
    self.done = None

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, tb):
    self.finish()


@dataclasses.dataclass
class PrefillRequest:
  """Prefill request.

  A transient object created after an lm.generate RPC request's inputs are
  pre-processed and destroyed when an empty cache slot takes its content.
  """

  preprocessed_inputs: DeviceTensors | None
  rpc_task: utils.RpcQueueTask
  enqueue_time: float


@dataclasses.dataclass
class ContinuousBatchingState:
  """Continuous batching state."""

  method: servable_model.ServableMethod
  model_key: str
  model_method: str
  service_id: str
  num_cache_slots: int
  max_decode_steps: int

  # Prefill requests are enqueued here after getting successfully pre-processed
  # on the primary host. When cache slots become available, dequeue requests
  # from this queue in the generation loop for further prefill and generation.
  prefill_queue: queue.SimpleQueue[PrefillRequest]
  # RPC tasks for requests being actively processed. One entry per cache slot.
  # Entries are filled in when a request leaves prefill_queue and freed when
  # the lm.generate RPC is fulfilled.
  rpc_tasks: list[utils.RpcQueueTask | None]

  # Decoded tokens, left aligned.
  decoded_tokens: Int[Array, 'B T']
  # Sum of log probabilities across prefilled and decoded tokens.
  scores: Float[Array, 'B']
  # Number of steps decoded.
  steps: Int[Array, 'B']
  # Whether each cache slot is being used (1) or unused (0).
  slots_in_use: Int[Array, 'B']

  # Stats.
  prefill_wait_time: utils.RequestStats
  insert_wait_time: utils.RequestStats
  generate_step_time: utils.RequestStats
  recent_waiting_prefills: Deque[int]
  recent_slots_in_use: Deque[int]

  def __init__(
      self,
      method: servable_model.ServableMethod,
      model_key: str,
      model_method: str,
      service_id: str,
      num_cache_slots: int,
      max_decode_step: int,
  ):
    self.method = method
    self.model_key = model_key
    self.model_method = model_method
    self.service_id = service_id
    self.num_cache_slots = num_cache_slots
    self.max_decode_steps = max_decode_step

    self.prefill_queue: queue.SimpleQueue[PrefillRequest] = queue.SimpleQueue()
    self.rpc_tasks = [None] * num_cache_slots

    self.decoded_tokens = np.zeros(
        (num_cache_slots, max_decode_step), dtype=np.int32
    )
    self.scores = np.zeros((num_cache_slots,))
    self.steps = np.zeros((num_cache_slots,), dtype=np.int32)
    self.slots_in_use = np.zeros((num_cache_slots,), dtype=np.int32)

    self.prefill_wait_time = utils.RequestStats(timespan_sec=np.float64(60.0))
    self.insert_wait_time = utils.RequestStats(timespan_sec=np.float64(60.0))
    self.generate_step_time = utils.RequestStats(timespan_sec=np.float64(60.0))
    self.recent_waiting_prefills = collections.deque()
    self.recent_slots_in_use = collections.deque()

  def update_stats(self, prefill_wait_time: float, insert_wait_time: float):
    """Update RPC stats."""
    self.prefill_wait_time.add(prefill_wait_time)
    self.insert_wait_time.add(insert_wait_time)
    self.recent_waiting_prefills.append(self.prefill_queue.qsize())
    if len(self.recent_waiting_prefills) > 10:
      self.recent_waiting_prefills.popleft()
    self.recent_slots_in_use.append(np.count_nonzero(self.slots_in_use))
    if len(self.recent_slots_in_use) > 10:
      self.recent_slots_in_use.popleft()


class PerMethodBatcher:
  """Runs per-method batching, and result batches are pushed to a queue."""

  def __init__(self):
    self._per_method_queues: Dict[MethodKey, Method] = {}
    self._batch_queue: queue.SimpleQueue[Batch] = queue.SimpleQueue()
    self._global_live_batches_lock: threading.Lock = threading.Lock()
    self._global_live_batches: int = 0

  # TODO(zhifengc, yuanzx): Some methods (say, image models'
  # methods) can involve heavy host processing. Maybe we should
  # allow multiple threads for those cases.
  def register_method(
      self,
      model: Optional[servable_model.ServableModel],
      key: MethodKey,
      batch_size: int,
      max_live_batches: int,
      preprocess_fn: Optional[
          Callable[
              [Sequence[utils.RpcQueueTask]],
              Tuple[DeviceTensors, InputShapeInfo],
          ]
      ] = None,
      batching_wait_secs: Optional[float] = None,
  ) -> None:
    """Registers a method that should be batched.

    Args:
      model: The servable model.
      key: A key identifying a method.
      batch_size: batch size for combining RPC tasks.
      max_live_batches: Maximum number of live batches.
      preprocess_fn: An optional preprocessing method that turns a sequence of
        RpcQueueTasks into device tensors to be consumed by device computation.
      batching_wait_secs: An optional batching waiting seconds in float.
    """
    method = Method(
        model=model,
        batch_size=batch_size,
        max_live_batches=max_live_batches,
        batching_wait_secs=batching_wait_secs,
    )
    self._per_method_queues[key] = method
    # If the model supports running dummy data on the primary, we can enqueue
    # to batch before preprocessing to allow early multi-host sync; if
    # preprocessing fails, we can let the primary to run the device function
    # using dummy data together with enqueued programs on secondary hosts.
    can_presync = (
        model is not None and model.supports_dummy_compute_on_primary()
    )

    # Start the batching loop.
    def _batching():
      # Keeps at most 2 active batches in the rest of pipeline.
      batch_sem = threading.Semaphore(value=2)

      def _finish_batch():
        batch_sem.release()
        with self._global_live_batches_lock:
          self._global_live_batches -= 1

      while True:
        batch_sem.acquire()
        rpc_tasks = method.queue.take_batch(batch_size)
        if method.admissioner.is_shutdown():
          # This must be an empty task generated after shutdown to unblock this
          # thread.
          assert len(rpc_tasks) == 1
          assert rpc_tasks[0].rpc is None
          batch_sem.release()
          break
        if _maybe_all_cancelled(rpc_tasks, f'batcher, {key}'):
          batch_sem.release()
          continue

        batch = Batch(
            key,
            rpc_tasks,
            _finish_batch,
        )
        # If there is no other unprocessed batch, we enqueue to batch before
        # preprocessing finishes.
        with self._global_live_batches_lock:
          # We don't need the slower but async host sync if there are no live
          # batches.
          batch.skip_host_sync = self._global_live_batches == 0
          presync = can_presync and batch.skip_host_sync
          self._global_live_batches += 1
          if presync:
            # For presync, we haven't pre-processed the input yet. So we have
            # to send the unpadded batch size to let secondary hosts pad their
            # dummy inputs to the next supported batch size.
            assert batch.input_tensors is None
            batch.unpadded_shape = InputShapeInfo(batch_size=len(rpc_tasks))
            assert batch.padded_shape is None
            self._batch_queue.put(batch)
        if preprocess_fn is None:
          batch.mark_as_ready()
        else:
          # Initialize input_tensors to None to indicate failed pre-processing.
          input_tensors = None
          unpadded_shape = InputShapeInfo(batch_size=len(rpc_tasks))
          padded_shape = None
          try:
            # Change these three values all at once or not at all.
            input_tensors, padded_shape = preprocess_fn(rpc_tasks)
            unpadded_shape = None
          except Exception as e:  # pylint: disable=broad-except
            # Catch arbitrary exception and propagate the error to the client
            # without crashing the server.
            error_msg = f'Preprocessing error: {e}\n{traceback.format_exc()}'
            logging.error(error_msg)
            for rpc_task in rpc_tasks:
              rpc_task.done(utils.internal_error(error_msg))
            if not presync:
              _finish_batch()
            continue
          finally:
            # Regardless of whether pre-processing succeeded, assign these three
            # values to batch while observing its invariance.
            batch.input_tensors = input_tensors
            batch.unpadded_shape = unpadded_shape
            batch.padded_shape = padded_shape
            batch.mark_as_ready()
        if not presync:
          self._batch_queue.put(batch)
        utils.traceprint_all(
            batch.rpc_tasks,
            f'Enqueued and preprocessed batch (batch size {batch.size()})',
        )
        logging.info(
            'Enqueued and preprocessed batch (batch size %s) for %s.',
            batch.size(),
            key,
        )

    t = threading.Thread(
        target=_batching, daemon=True, name=f'batching_{str(key)}'
    )
    t.start()

  def unregister_method(self, key: MethodKey):
    method = self._per_method_queues[key]
    method.admissioner.shutdown()
    # An empty task to unblock the batcher thread.
    method.queue.send(None, None, None, None)
    del self._per_method_queues[key]

  def has_method(self, key: MethodKey) -> bool:
    return key in self._per_method_queues

  def get_method_stats(
      self,
  ) -> List[
      Tuple[
          MethodKey,
          utils.RequestStats.Stats,
          utils.RequestStats.Stats,
          List[int],
      ]
  ]:
    """Returns the latest stats for every method key."""
    ret = []
    for mkey, method in self._per_method_queues.items():
      # TODO(zhifengc): Consider making 100 samples tunable.
      ret.append((
          mkey,
          method.ok_stats.get(100),
          method.err_stats.get(1),
          list(method.recent_batch_sizes),
      ))
    return ret

  def add_item(
      self,
      key: MethodKey,
      rpc: Optional[utils.RPCContext] = None,
      req: Optional[message.Message] = None,
      resp: Optional[message.Message] = None,
      optional_done: Optional[StatusCallback] = None,
  ):
    """Adds an item to the method's queue."""
    start_ts = time.time()
    method = self._per_method_queues.get(key)
    trace_callback = utils.get_current_trace_printer()
    trace_callback(f'Add item {key}')

    def done(status, *args, **kwargs):
      """Helper to run the done callback if it's not None."""
      if optional_done:
        optional_done(status, *args, **kwargs)
      if method is not None:
        if status.ok():
          method.ok_stats.add(time.time() - start_ts)
        else:
          method.err_stats.add(time.time() - start_ts)

    if method is None:
      return done(utils.not_found(f'method {key} is unloaded'))

    # Check ACLs.
    model: servable_model.ServableModel = method.model

    if model is not None and key.model_method is not None:
      model_method_name: str = key.model_method

      aclname: Optional[str] = model.get_acl(model_method_name)
      username: Optional[str] = rpc.username() if rpc is not None else None
      acl_ok = acl.Check(aclname, username)
      trace_callback(f'Username {username} ACL {aclname} check result {acl_ok}')
      if not acl_ok:
        return done(
            utils.permission_denied(
                f'Model {key.model_key} method {model_method_name} acl'
                f' {aclname} disallows {username}'
            )
        )
      # Check if extra_input keys are in servable_method.default_extra_inputs.
      servable_method: servable_model.ServableMethod = model.method(
          model_method_name
      )
      validate_status = validate.ValidateRequestForExtraInputs(
          req, servable_method.default_extra_inputs
      )
      if not validate_status.ok():
        return done(validate_status)

    success, active = method.admissioner.acquire(blocking=False)
    if not active:
      return done(utils.not_found(f'method {key} is unloaded'))

    if not success:
      return done(
          utils.resource_exhausted(f'Too many requests: {key} {method.limit()}')
      )

    def _done(status: utils.Status, *args, **kwargs):
      done(status, *args, **kwargs)
      method.admissioner.release()

    method.queue.send(rpc, req, resp, _done, trace_callback)

  def get_batch(self) -> Batch:
    """Dequeues an available batch."""
    qlen = self._batch_queue.qsize()  # Approximately.
    batch = self._batch_queue.get()
    self._per_method_queues[batch.method].record_batch_size(batch.size())
    utils.traceprint_all(
        batch.rpc_tasks,
        f'Dequeued from batch_queue: (qlen {qlen} bs {batch.size()})',
    )
    return batch


class LoadedModelManager:
  """A data structure that holds all loaded models."""

  def __init__(self, primary_process_id: int):
    # Indexed by key.
    # LOADED items have matching items in _models.
    # FAILED items have matching items in _errors.
    self._status = {}
    # Indexed by key.
    self._models = {}
    # Indexed by key.
    self._model_metadata = {}
    # Indexed by key.
    self._errors = {}
    self._primary_process_id = primary_process_id

  def load(
      self,
      key: str,
      model_path: str,
      ckpt_path: str,
      acls: Dict[str, str],
      overrides: Dict[str, str],
      prng_key: int,
      register_methods_callback: Optional[
          Callable[[servable_model.ServableModel], None]
      ] = None,
  ) -> servable_model.ServableModel:
    """Loads and initializes a model.

    Args:
      key: A string identifier for the model.
      model_path: Path of the model in the registry.
      ckpt_path: Path of the checkpoint.
      acls: ACL names for this model's methods.
      overrides: Overrides that can be used for dynamic reconfiguration of
        params. Values must be serialized JSONs.
      prng_key: PRNG key for this model.
      register_methods_callback: Optional callback to initialize model methods.

    Returns:
      The loaded model object.
    """
    if key in self._models:
      raise ValueError(f'Model {key} is already loaded, cannot load.')

    logging.info('Loading model for key: %s', key)
    self._status[key] = common_pb2.ModelStatus.LOADING
    try:
      model_class = servable_model_registry.get(model_path)
      if model_class is None:
        raise ValueError(f'Could not find servable model `{model_path}`.')
      if not issubclass(model_class, servable_model_params.ServableModelParams):
        raise ValueError(f'{model_path} is not a ServableModelParams')
      # pytype: disable=not-instantiable
      params = model_class()
      parsed_overrides = {
          key: json.loads(value) for key, value in overrides.items()
      }
      logging.info('model_service_base overrides= %s', parsed_overrides)
      params.apply_model_overrides(parsed_overrides)
      loaded = params.load(key, ckpt_path, self._primary_process_id, prng_key)
      # pytype: enable=not-instantiable
      loaded.set_acls(acls)
    except Exception as e:  # pylint: disable=broad-except
      self._status[key] = common_pb2.ModelStatus.FAILED
      # Stash the error message here and return it in a more detailed GetStatus
      # response when requested.
      self._errors[key] = str(e)
      raise

    if register_methods_callback is not None:
      register_methods_callback(loaded)

    self._status[key] = common_pb2.ModelStatus.LOADED
    self._model_metadata[key] = dict(
        checkpoint_path=ckpt_path,
        model_path=model_path,
        acls=acls,
        overrides=copy.deepcopy(overrides),
    )
    self._models[key] = loaded
    logging.info('Successfully loaded model for key: %s', key)
    return loaded

  def update(self, key: str, acls: Dict[str, str]) -> None:
    """Updates a model's metadata. E.g., ACLs."""
    model = self.maybe_get_model(key)
    if model:
      model.set_acls(acls)
      meta = self._model_metadata.get(key, None)
      if meta:
        meta['acls'] = acls

  def unload(self, key: str) -> None:
    """Unloads a model."""
    if not self.contains(key):
      raise ValueError(f'Model {key} is not loaded, cannot unload.')
    if self._status[key] == common_pb2.ModelStatus.FAILED:
      del self._status[key]
      del self._errors[key]
      return
    if key not in self._models:
      raise ValueError(f'Model {key} is not loaded, cannot unload.')
    self._status[key] = common_pb2.ModelStatus.UNLOADING
    try:
      self._models[key].unload()
    except Exception as e:  # pylint: disable=broad-except
      self._status[key] = common_pb2.ModelStatus.FAILED
      self._errors[key] = str(e)
      # Stash the error message here and return it in a more detailed GetStatus
      # response when requested.
      raise
    del self._status[key]
    del self._model_metadata[key]
    del self._models[key]

  def contains(self, key: str) -> bool:
    return key in self._status

  def get_status(self) -> Dict[str, 'common_pb2.ModelStatus']:
    return self._status

  def has_model(self, key: str) -> bool:
    return key in self._models

  def get_model(self, key: str) -> servable_model.ServableModel:
    return self._models[key]

  def get_models(self) -> Dict[str, servable_model.ServableModel]:
    return self._models

  def get_model_metadata(self, key: str) -> Mapping[str, Any]:
    return self._model_metadata[key]

  def maybe_get_model(self, key: str) -> Optional[servable_model.ServableModel]:
    return self._models.get(key)

  def has_error(self, key: str) -> bool:
    return key in self._errors

  def get_error(self, key: str) -> str:
    return self._errors[key]


class ModelService(metaclass=abc.ABCMeta):
  """Model RPC server base class."""

  def __init__(
      self,
      service_id: str,
      batcher: PerMethodBatcher,
      loader: LoadedModelManager,
      *args,
      **kwargs,
  ):
    self._batcher = batcher
    self._loader = loader
    self._service_id = service_id
    # Forward arguments to other parent classes.
    super().__init__(*args, **kwargs)  # pytype: disable=invalid-directive,wrong-keyword-args

  @classmethod
  def global_service_registry(cls) -> Mapping[str, List[Any]]:
    """Global mapping of service_id to model service class."""
    return _SERVICE_REGISTRY

  @abc.abstractmethod
  def ParseMethodRPCRequest(self, method_name: str, request: Any) -> Any:
    """Parses an RPC request into the input of a method before preprocessing."""

  @abc.abstractmethod
  def FillRPCResponse(
      self, method_name: str, method_outputs: Any, response: Any
  ) -> None:
    """Fills an RPC response based on a method's postprocessing outputs."""

  def _EnqueueRequestInternal(
      self,
      method: str,
      model_key: str,
      rpc_context: utils.RPCContext,
      req: message.Message,
      resp: message.Message,
      done: StatusCallback,
      streaming_output: bool,
  ):
    """Enqueues a request to the processing loop."""
    # Request may arrive before the corresponding _load_model() finishes or
    # after an unload. In this case, return NotFound.
    model = self._loader.maybe_get_model(model_key)
    if model is None or model.unloaded:
      done(utils.not_found(f'{model_key} is still loading or already unloaded'))
      return
    # Even if a model is loaded, the model may not support all methods.
    method_obj = model.methods.get(method)
    if method_obj is None:
      # Model may get unloaded since last maybe_get_model().
      if model.unloaded:
        done(utils.not_found(f'{model_key} is already unloaded'))
      else:
        done(utils.invalid_arg(f'{model_key} does not support {method}'))
      return

    # The method may not support streaming.
    if streaming_output and not method_obj.streamable_output:
      done(
          utils.invalid_arg(
              f'Method {method} does not support streaming output.'
          )
      )
      logging.error('Method %s does not support streaming output.', method)
      return

    # Only lm.generate supports continuous batching.
    if method_obj.continuous_batching:
      if method == 'lm.generate':
        batcher_item_key = MethodKey(
            MethodName.BATCHED_LM_GENERATE, method, self._service_id, model_key
        )
      else:
        done(
            utils.invalid_arg(
                f'Method {method} does not support continuous batching.'
            )
        )
        logging.error('Method %s does not support continuous batching.', method)
        return
    else:
      batcher_item_key = MethodKey(
          MethodName.MODEL, method, self._service_id, model_key
      )
    self._batcher.add_item(batcher_item_key, rpc_context, req, resp, done)


class ModelServiceGRPC(ModelService):
  """gRPC version of model service."""

  def ServiceName(self) -> str:
    """Returns the full name of the gRPC service, including package name."""
    raise NotImplementedError('ServiceName not implemented')

  @abc.abstractmethod
  def AddToServer(self, server: Any) -> None:
    """Adds the service to the GRPC server."""

  def EnqueueRequest(
      self,
      method: str,
      model_key: str,
      context: grpc.ServicerContext,
      req: message.Message,
      resp: message.Message,
  ) -> Awaitable[Any]:
    """Enqueues request, and returns a done future."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def _done(status: utils.Status, query_cost: Optional[int] = None):
      if query_cost:
        context.set_trailing_metadata([('query_cost_v0', str(query_cost))])
      if not status.ok():
        context.set_code(status.code)
        context.set_details(status.details)
      loop.call_soon_threadsafe(fut.set_result, None)

    self._EnqueueRequestInternal(
        method,
        model_key,
        utils.RPCContextGRPC(context),
        req,
        resp,
        _done,
        streaming_output=False,
    )
    return fut

  def EnqueueStreamRequest(
      self,
      method: str,
      model_key: str,
      context: grpc.ServicerContext,
      req: message.Message,
      empty_resp: message.Message,
  ) -> asyncio.Queue:
    """Enqueues a streaming request, and returns a done future."""
    loop = asyncio.get_running_loop()
    q = asyncio.Queue()

    def _done(
        status: utils.Status,
        resp: Optional[message.Message] = None,
        query_cost: Optional[int] = None,
    ):
      if query_cost:
        context.set_trailing_metadata([('query_cost_v0', str(query_cost))])
      if not status.ok():
        context.set_code(status.code)
        context.set_details(status.details)
      loop.call_soon_threadsafe(q.put_nowait, resp)

    self._EnqueueRequestInternal(
        method,
        model_key,
        utils.RPCContextGRPC(context),
        req,
        empty_resp,
        _done,
        streaming_output=True,
    )

    return q


def register_service(
    service_id: str,
) -> Callable[[Type[ModelService]], Type[ModelService]]:
  """Returns a decorator to register a service with a given service_id."""

  def _register(service_class: Type[ModelService]) -> Type[ModelService]:
    if service_id not in _SERVICE_REGISTRY:
      _SERVICE_REGISTRY[service_id] = []
    logging.info('Registering service %s for %s', service_class, service_id)
    _SERVICE_REGISTRY[service_id].append(service_class)
    return service_class

  return _register


class ModeletService:
  """Main service holding different model RPCs and loading/unloading models."""

  def __init__(
      self,
      service_port: int,
      debug_port: Optional[int],
      batcher: PerMethodBatcher,
      loader: LoadedModelManager,
      sax_cell: Optional[str],
      admin_port: Optional[int],
      platform_chip: Optional[str],
      platform_topology: Optional[str],
      *args,
      **kwargs,
  ):
    self._services = {}
    self._batcher = batcher
    self._loader = loader
    self._unload_lock = threading.Lock()
    self._models_being_unloaded = set()

    super().__init__(*args, **kwargs)
    self._batcher.register_method(
        None, MethodKey(MethodName.LOAD), batch_size=1, max_live_batches=4
    )
    self._batcher.register_method(
        None, MethodKey(MethodName.UNLOAD), batch_size=1, max_live_batches=4
    )
    self._batcher.register_method(
        None, MethodKey(MethodName.EXPORT), batch_size=1, max_live_batches=1
    )
    self._batcher.register_method(
        None, MethodKey(MethodName.SAVE), batch_size=1, max_live_batches=1
    )

    self._platform_chip = proto_util.to_chip_type(platform_chip)
    self._platform_topology = proto_util.to_chip_topology(platform_topology)
    logging.info('Sax cell %s', sax_cell)
    logging.info('Admin port %s', admin_port)
    logging.info('Platform chip %s, %s', platform_chip, self._platform_chip)
    logging.info(
        'Platform topology %s, %s', platform_topology, self._platform_topology
    )
    if sax_cell:
      if (
          self._platform_chip
          == admin_pb2.ModelServer.ChipType.CHIP_TYPE_UNKNOWN
      ):
        raise ValueError('Chip type unknown')
      if (
          self._platform_topology
          == admin_pb2.ModelServer.ChipTopology.CHIP_TOPOLOGY_UNKNOWN
      ):
        raise ValueError('Chip topology unknown')
    # If self._sax_cell is set to not None below, loadable model paths will get
    # imported, and location.Join will be called after the server starts.
    self._sax_cell = None
    self._admin_port = None
    if sax_cell is not None:
      if not sax_cell.startswith(_SAX_PREFIX):
        raise ValueError(f'Invalid sax_cell {sax_cell}. Should be /sax/<cell>')
      self._sax_cell = sax_cell
      self._admin_port = admin_port

    self._loadable_model_paths = []
    if self._sax_cell is not None:
      for k, v in servable_model_registry.get_all().items():
        if not issubclass(v, servable_model_params.ServableModelParams):
          continue
        status = v.check_serving_platform()
        if not status.ok():
          logging.info('Skipping unsupported model %s, %s', k, status.details)
          continue
        logging.info('Servable model %s', k)
        self._loadable_model_paths.append(k)
        for alias in servable_model_registry.get_aliases(k):
          logging.info('Servable model alias %s for %s', alias, k)
          self._loadable_model_paths.append(alias)

    self._ipport = ipaddr.Join(ipaddr.MyIPAddr(), service_port)
    self._debug_addr = (
        '' if debug_port is None else ipaddr.Join(ipaddr.MyIPAddr(), debug_port)
    )

  def model_services(self) -> Dict[str, ModelService]:
    return self._services

  @property
  def ipport(self) -> str:
    return self._ipport

  def after_start(self) -> None:
    """A callback that is supposed to be invoked after the server starts."""
    if self._sax_cell is not None:
      specs = admin_pb2.ModelServer(
          chip_type=self._platform_chip,
          chip_topology=self._platform_topology,
          servable_model_paths=list(self._loadable_model_paths),
      )
      try:
        location.Join(
            self._sax_cell,
            self._ipport,
            self._debug_addr,
            '',  # data_addr
            specs.SerializeToString(),
            admin_port=self._admin_port,
        )
      except RuntimeError as e:
        logging.exception('location.Join failed')
        raise e
      logging.info('Started joining SAX cell %s', self._sax_cell)

  def load(
      self,
      rpc_context: utils.RPCContext,
      req: modelet_pb2.LoadRequest,
      resp: modelet_pb2.LoadResponse,
      done_with_status: StatusCallback,
  ) -> None:
    """Loads a model."""
    self._batcher.add_item(
        MethodKey(MethodName.LOAD), rpc_context, req, resp, done_with_status
    )

  def update(
      self,
      req: modelet_pb2.UpdateLoadedRequest,
      done_with_status: StatusCallback,
  ) -> None:
    """Updates a model."""
    if not req.model_key:
      done_with_status(utils.invalid_arg('model_key is not specified.'))
      return
    self._loader.update(req.model_key, dict(req.acls.items))
    done_with_status(utils.ok())

  def unload(
      self,
      rpc_context: utils.RPCContext,
      req: modelet_pb2.UnloadRequest,
      resp: modelet_pb2.UnloadResponse,
      done_with_status: StatusCallback,
  ) -> None:
    """Unloads a model."""
    if not req.model_key:
      done_with_status(utils.invalid_arg('model_key is not specified.'))
      return
    with self._unload_lock:
      if req.model_key in self._models_being_unloaded:
        done_with_status(
            utils.invalid_arg(
                f'Model is already being unloaded. Key: {req.model_key}'
            )
        )
        return
      if not self._loader.contains(req.model_key):
        done_with_status(
            utils.invalid_arg(f'{req.model_key} not found, cannot unload.')
        )
        return
      self._models_being_unloaded.add(req.model_key)

    if self._loader.has_model(req.model_key):
      model = self._loader.get_model(req.model_key)
      for method_name in model.methods:
        self._batcher.unregister_method(
            MethodKey(
                MethodName.MODEL,
                method_name,
                model.method(method_name).service_id(),
                req.model_key,
            )
        )

    self._batcher.add_item(
        MethodKey(MethodName.UNLOAD), rpc_context, req, resp, done_with_status
    )
    with self._unload_lock:
      self._models_being_unloaded.remove(req.model_key)

  def export(
      self,
      rpc_context: utils.RPCContext,
      req: modelet_pb2.ExportRequest,
      resp: modelet_pb2.ExportResponse,
      done_with_status: StatusCallback,
  ) -> None:
    """Exports a model to a serialized format."""
    self._batcher.add_item(
        MethodKey(MethodName.EXPORT), rpc_context, req, resp, done_with_status
    )

  def save(
      self,
      rpc_context: utils.RPCContext,
      req: modelet_pb2.SaveRequest,
      resp: modelet_pb2.SaveResponse,
      done_with_status: StatusCallback,
  ) -> None:
    """Save a model."""
    self._batcher.add_item(
        MethodKey(MethodName.SAVE), rpc_context, req, resp, done_with_status
    )

  def get_status(
      self,
      req: modelet_pb2.GetStatusRequest,
      resp: modelet_pb2.GetStatusResponse,
  ) -> None:
    """Retrieves the server status."""
    model_by_key: dict[str, modelet_pb2.GetStatusResponse.ModelWithStatus] = {}
    for key, status in self._loader.get_status().items():
      model = modelet_pb2.GetStatusResponse.ModelWithStatus(
          model_key=key, model_status=status
      )
      model_by_key[key] = model
      if (
          status == common_pb2.ModelStatus.FAILED
          and req.include_failure_reasons
      ):
        model.failure_reason = self._loader.get_error(key)

    if req.include_method_stats:
      for (
          key,
          ok_stats,
          err_stats,
          recent_batch_sizes,
      ) in self._batcher.get_method_stats():
        if (
            key.service_id
            and (model := model_by_key.get(key.model_key)) is not None
        ):
          stats = model.method_stats.add(
              method=key.method_name(),
              errors_per_second=err_stats.rate(),
              successes_per_second=ok_stats.rate(),
              recent_batch_sizes=recent_batch_sizes,
          )
          if np.size(ok_stats.samples) > 0:
            stats.mean_latency_on_success_per_second = ok_stats.mean()
            percentiles = np.percentile(ok_stats.samples, [50, 95, 99])
            stats.p50_latency_on_success_per_second = percentiles[0]
            stats.p95_latency_on_success_per_second = percentiles[1]
            stats.p99_latency_on_success_per_second = percentiles[2]

    for model in model_by_key.values():
      resp.models.append(model)


class ModeletServiceGRPC(ModeletService, modelet_pb2_grpc.ModeletServicer):
  """gRPC interfaces for ModeletService."""

  def _future_and_done_cb(
      self, context: grpc.ServicerContext
  ) -> Tuple[Awaitable[Any], Callable[[utils.Status], None]]:
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def _done(status: utils.Status):
      if not status.ok():
        context.set_code(status.code)
        context.set_details(status.details)
      loop.call_soon_threadsafe(fut.set_result, None)

    return fut, _done

  def AddToServer(self, server: Any) -> None:
    modelet_pb2_grpc.add_ModeletServicer_to_server(self, server)

  async def Load(self, request, context):
    resp = modelet_pb2.LoadResponse()
    fut, done = self._future_and_done_cb(context)
    self.load(utils.RPCContextGRPC(context), request, resp, done)
    await fut
    return resp

  async def UpdateLoaded(self, request, context):
    resp = modelet_pb2.UpdateLoadedResponse()
    fut, done = self._future_and_done_cb(context)
    self.update(request, done)
    await fut
    return resp

  async def Unload(self, request, context):
    resp = modelet_pb2.UnloadResponse()
    fut, done = self._future_and_done_cb(context)
    self.unload(utils.RPCContextGRPC(context), request, resp, done)
    await fut
    return resp

  async def Export(self, request, context):
    resp = modelet_pb2.ExportResponse()
    fut, done = self._future_and_done_cb(context)
    self.export(utils.RPCContextGRPC(context), request, resp, done)
    await fut
    return resp

  async def Save(self, request, context):
    resp = modelet_pb2.SaveResponse()
    fut, done = self._future_and_done_cb(context)
    self.save(utils.RPCContextGRPC(context), request, resp, done)
    await fut
    return resp

  async def GetStatus(self, request, context):
    resp = modelet_pb2.GetStatusResponse()
    self.get_status(request, resp)
    return resp


class Exporter:
  """Injectable class for implementing exporting of models."""

  def __init__(self, model_services_runner):
    pass

  def parse_export_request(self, req: modelet_pb2.ExportRequest) -> List[str]:
    """Parses export request."""
    raise NotImplementedError()

  def export_model(self, *args):
    """Exports a model."""
    raise NotImplementedError()

  def finalize_export(self, *args) -> None:
    """Actions after model exporting."""
    pass

  def export_error_to_status(self, e: Exception) -> utils.Status:
    """Converts an error during model export to a Status."""
    msg = f'Exporting error: {e}'
    if isinstance(e, ValueError):
      return utils.invalid_arg(msg)
    elif isinstance(e, NotImplementedError):
      return utils.unimplemented(msg)
    elif isinstance(e, FileExistsError):
      return utils.already_exists(msg)
    else:
      return utils.internal_error(msg)


class ModelServicesRunner:
  """Implements the main loop for request processing with multiple services.

  Usage::
    runner = ModelServicesRunner(...)

    runner.start()
    ...
    runner.stop()
    runner.wait()
  """

  def __init__(
      self,
      is_primary_process: bool = True,
      port: int = 0,
      debug_port: Optional[int] = None,
      deterministic_prng_seed: Optional[int] = None,
      sax_cell: Optional[str] = None,
      admin_port: Optional[int] = None,
      platform_chip: Optional[str] = None,
      platform_topology: Optional[str] = None,
      backend: Optional[spmd_backend.SPMDBackend] = None,
      fail_on_error: bool = False,
  ):
    self._is_primary = is_primary_process
    # If deterministic_prng_seed is provided, all models will use this as the
    # initial seed.
    self._det_prng_seed = deterministic_prng_seed
    self._batcher = PerMethodBatcher()
    self._batcher.register_method(
        None,
        MethodKey(MethodName.KEEP_DEVICES_WARM),
        batch_size=1,
        max_live_batches=1,
    )
    self._log_exception = functools.partial(
        logging.fatal if fail_on_error else logging.error, exc_info=True
    )

    if backend is None:
      if not self._is_primary:
        raise NotImplementedError('No spmd_backend provided for multi-host.')
      self._spmd_backend = spmd_backend.SingleHostBackend()
    else:
      self._spmd_backend = backend

    if self._is_primary:
      # Thread pool for async postprocess on the host.
      self._postprocess_pool = utils.ThreadPool(
          num_threads=16, thread_name_prefix='model_service_runner_postprocess'
      )

      # Device execution ensures the enqueue operations of streaming outputs
      # across different requests are serializable. By constraining
      # num_thread=1, we can ensure the dequeue operations are also
      # serializable because utils.ThreadPool runs in FIFO order.
      self._stream_pool = utils.ThreadPool(
          num_threads=1,
          thread_name_prefix='model_service_runner_stream_dequeuer',
      )

      primary_host = self._spmd_backend.spmd_host_index()
      if self._spmd_backend.spmd_host_count() > 1:
        self._spmd_backend.send_via_device(str(primary_host))
      self._worker_thread = threading.Thread(
          target=self._run_primary_worker_loop,
          daemon=False,
          name='model_service_runner_primary_worker',
      )
      self._keep_warm_thread = threading.Thread(
          target=self._run_keep_warm_loop,
          daemon=True,
          name='model_service_runner_keep_warm',
      )

      # For continuous batching
      self._token_batching_state: Dict[str, ContinuousBatchingState] = {}
      self._generate_thread = threading.Thread(
          target=self._run_generation_loop,
          daemon=True,  # TODO(changlan): Handle shutdown properly
          name='model_service_runner_generation',
      )
    else:
      primary_id_str = self._spmd_backend.receive_via_device()
      primary_host = int(primary_id_str)
      logging.info('Secondary worker loop. Primary process: %d', primary_host)
      self._worker_thread = threading.Thread(
          target=self._run_secondary_worker_loop,
          daemon=False,
          name='model_service_runner_secondary_worker',
      )

    self._loaded_models = LoadedModelManager(primary_host)
    self._batcher.register_method(
        None, MethodKey(MethodName.TERMINATE), batch_size=1, max_live_batches=1
    )
    self._modelet_service = ModeletServiceGRPC(
        port,
        debug_port,
        batcher=self._batcher,
        loader=self._loaded_models,
        sax_cell=sax_cell,
        admin_port=admin_port,
        platform_chip=platform_chip,
        platform_topology=platform_topology,
    )
    self._platform_topology = platform_topology
    all_grpc_services = [self._modelet_service]
    service_names = [
        modelet_pb2.DESCRIPTOR.services_by_name['Modelet'].full_name,
        reflection.SERVICE_NAME,
    ]

    self._model_services = {}
    non_grpc_services = []
    if self._is_primary:
      for service_id, service_classes in _SERVICE_REGISTRY.items():
        for service_class in service_classes:
          service = service_class(
              service_id=service_id,
              batcher=self._batcher,
              loader=self._loaded_models,
          )
          if issubclass(service_class, ModelServiceGRPC):
            all_grpc_services.append(service)
            service_names.append(service.ServiceName())
          else:
            non_grpc_services.append(service)
          # It's OK to override previous value, since the gRPC and Stubby
          # services should have the same implementation for
          # ParseMethodRPCRequest and FillRPCResponse. We prefer
          # ModelServiceGRPC as it's opensource.
          if (
              issubclass(service_class, ModelServiceGRPC)
              or service_id not in self._model_services
          ):
            self._model_services[service_id] = service

    self._aio_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self._aio_loop)
    self._grpc_server = self._create_grpc_server(non_grpc_services)
    for service in all_grpc_services:
      service.AddToServer(self._grpc_server)
    reflection.enable_server_reflection(service_names, self._grpc_server)
    self._multihost_sync = multi_host_sync.MultiHostSync(
        is_primary_process,
        self._grpc_server,
        self._modelet_service.ipport,
        self._spmd_backend,
    )
    server_credentials = self._get_server_credentials()  # pylint: disable=assignment-from-none
    if server_credentials is None:
      self._grpc_server.add_insecure_port(f'[::]:{port}')
    else:
      self._grpc_server.add_secure_port(f'[::]:{port}', server_credentials)

    self._aio_thread = threading.Thread(target=self._run_aio_loop)
    self._terminate_future = self._aio_loop.create_future()
    # Any unhandled exception in the worker thread will be set here.
    self._worker_thread_exception: Optional[Exception] = None

  def _create_grpc_server(
      self, non_grpc_services: List[ModelService]
  ) -> grpc.aio.Server:
    assert not non_grpc_services
    return grpc.aio.server()

  def _get_server_credentials(self) -> Optional[grpc.ServerCredentials]:
    # TODO(sax-dev): Add credentials for OSS.
    return None

  def _get_client_channel_credentials(
      self,
  ) -> Optional[grpc.ChannelCredentials]:
    # TODO(sax-dev): Add credentials for OSS.
    return None

  def _run_keep_warm_loop(self):
    while True:
      self._batcher.add_item(MethodKey(MethodName.KEEP_DEVICES_WARM))
      time.sleep(_KEEP_WARM_SECONDS)

  @property
  def spmd_backend(self) -> spmd_backend.SPMDBackend:
    return self._spmd_backend

  @property
  def modelet_service(self):
    return self._modelet_service

  def model_service(self, service_id: str) -> ModelService:
    return self._model_services[service_id]

  def start(self) -> None:
    """Start the worker and server threads, non-blocking."""
    self._multihost_sync.initialize(self._get_client_channel_credentials())
    self._worker_thread.start()
    self._aio_thread.start()
    if self._is_primary:
      self._keep_warm_thread.start()
      self._generate_thread.start()

  def on_initial_models_load_completion(self) -> None:
    """Callback to invoke after all models are loaded."""

  def _run_aio_loop(self) -> None:
    """Runs the aio grpc loop."""

    async def _serve():
      await self._grpc_server.start()
      if self._is_primary:
        self._modelet_service.after_start()
      await self._terminate_future
      await self._grpc_server.stop(None)
      await self._grpc_server.wait_for_termination()

    self._aio_loop.run_until_complete(_serve())
    self._aio_loop.close()

  def wait(self) -> None:
    """Wait until all threads finishes."""
    self._worker_thread.join()
    if self._worker_thread_exception is not None:
      # TODO(yuanzx,jiawenhao): Consider to inform the admin server about this
      # crash.
      raise self._worker_thread_exception
    self._aio_thread.join()
    self._multihost_sync.wait()

  def stop(self) -> None:
    """Wait until all threads finishes."""
    self._aio_loop.call_soon_threadsafe(self._terminate_future.set_result, ())
    self._enqueue_terminate()
    self._multihost_sync.stop()

  def _generate_rng_seed(self):
    if self._det_prng_seed is not None:
      return self._det_prng_seed
    return uuid.uuid4().int & ((1 << 63) - 1)

  def _encode_message(self, *msgs: str) -> str:
    return json.dumps(msgs)  # tuples are serialized as lists in JSON.

  def _decode_message(self, encoded: str) -> List[str]:
    msgs = json.loads(encoded)
    assert isinstance(msgs, list)
    return msgs

  def _load_model(
      self,
      model_key: str,
      model_path: str,
      checkpoint_path: str,
      acls: Dict[str, str],
      overrides: Dict[str, str],
      prng_key: int,
  ) -> None:
    """Loads a model and initializes its methods."""
    if self._loaded_models.contains(model_key):
      return

    def register_methods(model: servable_model.ServableModel):
      for method_name in model.methods:
        method = model.method(method_name)
        service_id = method.service_id()
        service = self._model_services[service_id]

        def preprocess_rpc_tasks(
            rpc_tasks: Sequence[utils.RpcQueueTask],
            method=method,
            method_name=method_name,
            service=service,
            prefill=False,
        ):
          inputs = method.pre_processing([
              service.ParseMethodRPCRequest(method_name, t.request)
              for t in rpc_tasks
          ])
          extra_inputs = [
              method.get_extra_inputs_from_request_inputs(t.request)
              for t in rpc_tasks
          ]
          inputs = method.update_extra_inputs(
              inputs, len(rpc_tasks), extra_inputs
          )

          unpadded_shape = method.get_unpadded_shape(len(rpc_tasks), inputs)
          padded_shape = method.get_padded_input_shape(unpadded_shape)
          if prefill:
            res = method.input_to_device_for_continuous_batching(
                inputs, unpadded_shape
            )
          else:
            res = method.input_to_device(inputs, unpadded_shape, padded_shape)
          return res, padded_shape

        method_key = MethodKey(
            MethodName.MODEL, method_name, service_id, model_key
        )

        self._batcher.register_method(
            model,
            method_key,
            method.batch_size,
            preprocess_fn=preprocess_rpc_tasks,
            max_live_batches=method.max_live_batches,
            batching_wait_secs=method.batching_wait_secs,
        )

        # If a method supports continuous batching, additionally register the
        # continuous batching variation of the method.
        if method.continuous_batching:
          # Currently, only lm.generate supports continuous batching.
          if method_name != 'lm.generate':
            raise ValueError(
                f'Continuous batching is not supported for {method_name}'
            )
          batch_generate_method_key = MethodKey(
              MethodName.BATCHED_LM_GENERATE, method_name, service_id, model_key
          )
          self._batcher.register_method(
              model,
              batch_generate_method_key,
              1,
              preprocess_fn=functools.partial(
                  preprocess_rpc_tasks, prefill=True
              ),
              # Note: batch size of this method is always 1 so max_live_batches
              # denotes the number of live requests.
              max_live_batches=method.num_cache_slots * 2,
          )

          self._token_batching_state[model_key] = ContinuousBatchingState(
              method,
              model_key=model_key,
              model_method=method_name,
              service_id=service_id,
              num_cache_slots=method.num_cache_slots,
              max_decode_step=method.max_decode_steps,
          )

    self._loaded_models.load(
        model_key,
        model_path,
        checkpoint_path,
        acls,
        overrides,
        prng_key,
        register_methods,
    )

  def _save_model(self, model_key: str, checkpoint_path: str):
    """Saves a model checkpoint."""
    if not self._loaded_models.contains(model_key):
      logging.warning('Model %s is not loaded.', model_key)
      return
    self._loaded_models.get_model(model_key).save(checkpoint_path)

  def _enqueue_terminate(self):
    self._batcher.add_item(key=MethodKey(MethodName.TERMINATE))

  def _inform_secondary_hosts(self, *msgs: str, skip_host_sync=True) -> None:
    self._multihost_sync.send(self._encode_message(*msgs), skip_host_sync)

  def _postprocess_async(
      self,
      model: servable_model.ServableModel,
      batch: Batch,
      out_tensors: DeviceTensors,
      streaming_done: Optional[utils.Notification],
  ) -> None:
    """Runs post processing and RPC dones asynchronously."""
    # Use a list to allow deleting out_tensors earlier in the thread pool.
    out_tensors_container = [out_tensors]
    del out_tensors

    def _postprocess():
      if streaming_done is not None:
        logging.info('Waiting for streaming to finish.')
        streaming_done.wait()
      with batch:
        # We don't need to postprocess if preprocess failed where input_tensors
        # is set to None.
        pre_process_failure = batch.input_tensors is None
        # Free input tensors.
        batch.input_tensors = None
        done_rpcs = 0
        if not pre_process_failure:
          logging.info('Processing final results.')
        try:
          method_obj = model.method(batch.method.model_method)
          utils.traceprint_all(
              batch.rpc_tasks, f'in _postprocess_async: {batch.method}'
          )
          start_ts = time.time()
          host_tensors = method_obj.output_to_host(
              out_tensors_container[0], len(batch.rpc_tasks)
          )
          tpu_ms_per_chip = int(1000 * (time.time() - start_ts))
          chips_count = (
              proto_util.count_physical_chips(self._platform_topology) or 1
          )
          tpu_ms = (tpu_ms_per_chip * chips_count) // len(batch.rpc_tasks)
          # Free device tensors.
          del out_tensors_container[0]
          utils.traceprint_all(
              batch.rpc_tasks, f'After output_to_host: {batch.method}'
          )
          if not pre_process_failure:
            # No more result for streaming.
            if streaming_done is not None:
              for task in batch.rpc_tasks:
                task.done(utils.ok(), resp=None, query_cost=tpu_ms)
              return
            # TODO(zhifengc): Might make more sense to split this phase into
            # two. One calls output_to_host and the other calls post_processing.
            outputs = method_obj.post_processing(host_tensors)
            utils.traceprint_all(
                batch.rpc_tasks, f'After post_processing: {batch.method}'
            )
            if len(outputs) != len(batch.rpc_tasks):
              raise ValueError(
                  f'post_processing returned {outputs=}, expected'
                  f' {len(batch.rpc_tasks)} items.'
              )
            for out, task in zip(outputs, batch.rpc_tasks):
              self._model_services[batch.method.service_id].FillRPCResponse(
                  batch.method.model_method, out, task.response
              )
              task.done(utils.ok(), query_cost=tpu_ms)
              done_rpcs += 1
        except Exception as e:  # pylint: disable=broad-except
          if not pre_process_failure:
            self._log_exception(
                'Postprocessing error. model_key: %s, method: %s, error: %s',
                batch.method.model_key,
                batch.method.model_method,
                e,
            )
            error_msg = f'Postprocessing error: {e}\n{traceback.format_exc()}'
            for task in batch.rpc_tasks[done_rpcs:]:
              task.done(utils.internal_error(error_msg))

    self._postprocess_pool.run(_postprocess)

  # TODO(changlan): Merge with _postprocess_async
  def _postprocess_stream_async(
      self,
      model: servable_model.ServableModel,
      batch: Batch,
      streaming_done: utils.Notification,
  ):
    """Runs post processing and RPC dones on streamed tensors asynchronously."""

    def _postprocess():
      done = False
      postprocess_error = False
      stream_state = None
      while not done:
        b = len(batch.rpc_tasks)
        method_obj = model.method(batch.method.model_method)
        host_tensors = method_obj.dequeue_stream_output()
        if host_tensors is None:
          # Done with streaming.
          done = True
        else:
          host_tensors = method_obj.remove_batch_padding(host_tensors, b)
        if postprocess_error:
          # Postprocessing failed before.
          continue

        if batch.input_tensors is not None:
          done_rpcs = 0
          try:
            start_ts = time.time()
            outputs, stream_state = method_obj.post_processing_stream(
                host_tensors, stream_state
            )
            query_cost = (time.time() - start_ts) / len(batch.rpc_tasks)
            if host_tensors is not None and len(outputs) != len(
                batch.rpc_tasks
            ):
              raise ValueError(
                  f'post_processing_stream returned {outputs=}, expected'
                  f' {len(batch.rpc_tasks)} items.'
              )
            for out, task in zip(outputs, batch.rpc_tasks):
              # Use a new response each time.
              resp = copy.deepcopy(task.response)
              self._model_services[batch.method.service_id].FillRPCResponse(
                  batch.method.model_method, out, resp
              )
              task.done(utils.ok(), resp=resp, query_cost=query_cost)
              done_rpcs += 1
          except Exception as e:  # pylint: disable=broad-except
            self._log_exception(
                'Postprocessing error. model_key: %s, method: %s, error: %s',
                batch.method.model_key,
                batch.method.model_method,
                e,
            )
            error_msg = f'Postprocessing error: {e}\n{traceback.format_exc()}'
            for task in batch.rpc_tasks[done_rpcs:]:
              task.done(utils.internal_error(error_msg))
            postprocess_error = True

      streaming_done.notify()

    self._stream_pool.run(_postprocess)

  def _run_primary_worker_loop(self):
    """Main loop for processing batches."""
    while True:
      batch = self._batcher.get_batch()
      match batch.method.name:
        case MethodName.LOAD:
          with batch:
            assert len(batch.rpc_tasks) == 1
            task = batch.rpc_tasks[0]
            request = typing.cast(modelet_pb2.LoadRequest, task.request)
            assert batch.method.model_key is None
            model_key = request.model_key
            assert model_key
            try:
              # Generate a seed for the model and pass to secondary hosts.
              prng_seed = self._generate_rng_seed()
              self._inform_secondary_hosts(
                  batch.method.name,
                  model_key,
                  request.model_path,
                  request.checkpoint_path,
                  json.dumps({k: v for k, v in request.overrides.items()}),
                  str(prng_seed),
              )
              self._load_model(
                  model_key,
                  request.model_path,
                  request.checkpoint_path,
                  dict(request.acls.items),
                  dict(request.overrides),
                  prng_seed,
              )
              task.done(utils.ok())
            except ValueError as e:
              self._log_exception(
                  (
                      'Invalid load request. model_key: %s, model_path: %s,'
                      ' error: %s'
                  ),
                  model_key,
                  request.model_path,
                  e,
              )
              task.done(utils.invalid_arg(f'{e}'))
            except Exception as e:  # pylint: disable=broad-except
              self._log_exception(
                  (
                      'Internal error during loading. model_key: %s,'
                      ' model_path: %s, error: %s'
                  ),
                  model_key,
                  request.model_path,
                  e,
              )
              task.done(utils.internal_error(f'Loading error: {e}'))
        case MethodName.UNLOAD:
          with batch:
            assert len(batch.rpc_tasks) == 1
            task = batch.rpc_tasks[0]
            request = typing.cast(modelet_pb2.UnloadRequest, task.request)
            assert batch.method.model_key is None
            model_key = request.model_key
            try:
              if not model_key:
                raise ValueError('model_key is not specified.')
              self._inform_secondary_hosts(batch.method.name, model_key)
              self._loaded_models.unload(model_key)
              task.done(utils.ok())
            except ValueError as e:
              logging.exception(
                  'Invalid unload request. model_key %s, error: %s',
                  model_key,
                  e,
              )
              task.done(utils.invalid_arg(f'Unloading error: {e}'))
            except Exception as e:  # pylint: disable=broad-except
              self._log_exception(
                  'Internal error during unloading. model_key: %s, error: %s',
                  model_key,
                  e,
              )
              task.done(utils.internal_error(f'Unloading error: {e}'))
        case MethodName.EXPORT:
          with batch:
            assert len(batch.rpc_tasks) == 1
            task = batch.rpc_tasks[0]
            request = typing.cast(modelet_pb2.ExportRequest, task.request)
            exporter = Exporter(self)
            try:
              export_args = exporter.parse_export_request(request)
              # Starts a multi-host export job. Any exception must be from
              # low-level software/hardware stack and or secondary workers not
              # being informed. Thus we crash the server to avoid hanging if
              # there is any exception.
              try:
                self._inform_secondary_hosts(batch.method.name, *export_args)
                exporter.export_model(*export_args)
              except Exception as e:  # pylint: disable=broad-except
                self._worker_thread_exception = e
                break
              exporter.finalize_export(*export_args)
              task.done(utils.ok())
            except Exception as e:  # pylint: disable=broad-except
              self._log_exception(
                  (
                      '%s during Exporting. model_key: %s, method_names %s,'
                      ' export_path: %s, error: %s'
                  ),
                  type(e),
                  request.model_key,
                  request.method_names,
                  request.export_path,
                  e,
              )
              task.done(exporter.export_error_to_status(e))
        case MethodName.SAVE:
          with batch:
            assert len(batch.rpc_tasks) == 1
            task = batch.rpc_tasks[0]
            request = typing.cast(modelet_pb2.SaveRequest, task.request)
            try:
              self._inform_secondary_hosts(
                  batch.method.name, request.model_key, request.checkpoint_path
              )
              self._save_model(request.model_key, request.checkpoint_path)
              task.done(utils.ok())
            except ValueError as e:
              self._log_exception(
                  'Invalid save request. model_key %s, error: %s, ',
                  request.model_key,
                  e,
              )
              task.done(utils.invalid_arg(f'Save checkpoint error: {e}'))
            except Exception as e:  # pylint: disable=broad-except
              self._log_exception(
                  (
                      'Internal error during Saving checkpoint. model_key: %s, '
                      'error: %s'
                  ),
                  request.model_key,
                  e,
              )
              task.done(utils.internal_error(f'Saving checkpoint error: {e}'))
        case MethodName.TERMINATE:
          with batch:
            assert batch.method.model_key is None
            self._inform_secondary_hosts(batch.method.name)
            break
        case MethodName.KEEP_DEVICES_WARM:
          with batch:
            self._inform_secondary_hosts(batch.method.name)
        case MethodName.MODEL:
          # Model methods hosts are simply invoking a device program with
          # pre-processed inputs, so we do not expect any known exception here.
          # An exception must be from low-level software/hardware stack and we
          # crash the job.
          try:
            self._inform_secondary_hosts(
                batch.method.name,
                batch.method.model_key,
                batch.method.model_method,
                str(batch.unpadded_shape),
                str(batch.padded_shape),
                skip_host_sync=batch.skip_host_sync,
            )
            model = self._loaded_models.get_model(batch.method.model_key)
            batch.wait_for_ready()

            method_obj = model.method(batch.method.model_method)
            enable_token_streaming = method_obj.streamable_output
            streaming_done = (
                utils.Notification() if enable_token_streaming else None
            )
            if batch.input_tensors is None:
              # Failed preprosessing. Since we have already informed secondary
              # hosts, we need to compute on tensors.
              if self._spmd_backend.spmd_host_count() > 1:
                # JAX dispatches device_compute synchronously to XLA:GPU. Hence
                # _postprocess_stream_async must start before device_compute,
                # otherwise device_compute may block the consumption of
                # streaming queue until it finishes.
                if enable_token_streaming:
                  self._postprocess_stream_async(model, batch, streaming_done)
                assert batch.unpadded_shape is not None
                assert batch.padded_shape is None
                padded_shape = method_obj.get_padded_input_shape(
                    batch.unpadded_shape
                )
                result = method_obj.device_compute_with_dummy_data(padded_shape)
              else:
                # Skip device compute if there is no secondary host.
                result = None
            else:
              # Successful pre-processing. Compute on device using the padded
              # shape picked by pre-processing.
              if enable_token_streaming:
                self._postprocess_stream_async(model, batch, streaming_done)
              assert batch.unpadded_shape is None
              assert batch.padded_shape is not None
              result = method_obj.device_compute(
                  input_batch=batch.input_tensors,
                  padded_shape=batch.padded_shape,
              )
            if result is None:
              batch.finish()
            else:
              self._postprocess_async(model, batch, result, streaming_done)
              del result
          except Exception as e:  # pylint: disable=broad-except
            self._worker_thread_exception = e
            batch.finish()
            break
        case MethodName.BATCHED_LM_GENERATE:
          try:
            batch.wait_for_ready()
            assert len(batch.rpc_tasks) == 1
            state = self._token_batching_state[batch.method.model_key]

            state.prefill_queue.put(
                PrefillRequest(
                    preprocessed_inputs=batch.input_tensors,
                    rpc_task=batch.rpc_tasks[0],
                    enqueue_time=time.time(),
                )
            )
            batch.finish()
          except Exception as e:  # pylint: disable=broad-except
            self._worker_thread_exception = e
            batch.finish()
            break
        case _:
          batch.finish()
          self._log_exception('Unknown method: %s', batch.method.name)

  def _run_generation_loop(self):
    """Host loop for running continuous generation on the primary host."""
    logging.info('Running generation loop')
    while True:
      for model_key in self._loaded_models.get_models():
        if model_key not in self._token_batching_state:
          continue
        state = self._token_batching_state[model_key]
        assert model_key == state.model_key
        model = self._loaded_models.get_model(state.model_key)
        method_obj = model.method(state.model_method)

        # If there are prefill requests and empty cache slots, run prefills
        # and insert them to the cache first
        while not (state.slots_in_use.all() or state.prefill_queue.empty()):
          request = state.prefill_queue.get()
          prefill_dequeue_time = time.time()

          slot = np.argmin(state.slots_in_use)
          state.slots_in_use[slot] = 1

          self._inform_secondary_hosts(
              MethodName.PREFILL_INSERT,
              state.model_key,
              state.model_method,
              str(slot),
              skip_host_sync=False,
          )

          assert request.preprocessed_inputs is not None
          assert request.rpc_task is not None
          scores, tokens, prefix_cache = method_obj.prefill(
              inputs=request.preprocessed_inputs,
          )
          state.update_stats(
              prefill_wait_time=prefill_dequeue_time - request.enqueue_time,
              insert_wait_time=time.time() - prefill_dequeue_time,
          )
          method_obj.insert(prefix_cache, slot)
          state.decoded_tokens[slot][0] = np.array(tokens.addressable_data(0))
          state.scores[slot] = np.array(scores.addressable_data(0))
          state.rpc_tasks[slot] = request.rpc_task
          state.steps[slot] = 1

        # Skip generation if none of the cache slots are in use.
        if not state.slots_in_use.any():
          continue

        # Perform generation.
        start_ts = time.time()
        self._inform_secondary_hosts(
            MethodName.GENERATE,
            state.model_key,
            state.model_method,
            skip_host_sync=False,
        )
        token_batch = (
            state.decoded_tokens[
                np.arange(state.num_cache_slots), state.steps - 1
            ]
            * state.slots_in_use
        )
        token_batch = method_obj.input_to_device_for_continuous_batching(
            token_batch,
            InputShapeInfo(batch_size=state.num_cache_slots),
        )
        res = method_obj.generate(token_batch)
        scores, tokens, done = method_obj.output_to_host(
            res, unpadded_batch_size=state.num_cache_slots
        )
        state.generate_step_time.add(time.time() - start_ts)

        state.decoded_tokens[np.arange(state.num_cache_slots), state.steps] = (
            tokens
        )
        state.scores += scores * state.slots_in_use
        state.steps += state.slots_in_use

        done = done * state.slots_in_use
        done = np.logical_or(done, state.steps >= state.max_decode_steps)

        # If any of the sequences in the batch is done
        if np.any(done):
          sequences = state.decoded_tokens[done]
          output_strings = method_obj.detokenize(sequences)
          done_slots = np.flatnonzero(done)

          for idx, slot in enumerate(done_slots):
            outputs = [output_strings[idx]], [state.scores[slot]]
            self._model_services[state.service_id].FillRPCResponse(
                state.model_method, outputs, state.rpc_tasks[slot].response
            )
            try:
              state.rpc_tasks[slot].done(utils.ok(), query_cost=0)
            except Exception as e:  # pylint: disable=broad-except
              self._log_exception('Error occurred: %s, error: %s', model_key, e)
            state.rpc_tasks[slot] = None

          # Reset the cache slot state.
          state.decoded_tokens[done] = 0
          state.scores[done] = 0.0
          state.steps[done] = 0
          state.slots_in_use[done] = 0

  def _run_secondary_worker_loop(self):
    """Runs the processing loop in secondary hosts in a multi-host setup."""
    while True:
      sync_output = self._multihost_sync.receive()
      msgs = self._decode_message(sync_output)
      method = msgs.pop(0)
      match method:
        case MethodName.LOAD:
          model_key, model_path, ckpt_path, overrides_json, prng_seed = msgs
          overrides = json.loads(overrides_json)
          logging.info('Received load: %s', model_key)
          try:
            prng_seed = int(prng_seed)
            if not self._loaded_models.contains(model_key):
              self._loaded_models.load(
                  model_key,
                  model_path,
                  ckpt_path,
                  {},  # Empty ACLs because only the primary worker needs it.
                  overrides,
                  prng_seed,
              )
          except Exception as e:  # pylint: disable=broad-except
            self._log_exception(
                'Error occurred during loading: %s, error: %s', model_key, e
            )
        case MethodName.UNLOAD:
          model_key = msgs[0]
          logging.info('Received unload: %s', model_key)
          try:
            self._loaded_models.unload(model_key)
          except Exception as e:  # pylint: disable=broad-except
            self._log_exception(
                'Error occurred during unloading: %s, error: %s', model_key, e
            )
        case MethodName.TERMINATE:
          logging.info('Received terminate')
          break
        case MethodName.KEEP_DEVICES_WARM:
          logging.info('Received keep devices warm')
          continue
        case MethodName.SAVE:
          model_key, ckpt_path = msgs
          logging.info('Received save: %s', model_key)
          try:
            self._save_model(model_key, ckpt_path)
          except Exception as e:  # pylint: disable=broad-except
            self._log_exception(
                'Error occurred during save: %s, error: %s', model_key, e
            )
        case MethodName.EXPORT:
          # Starts a multi-host export job. Any exception must be from low-level
          # software/hardware stack. Thus we crash the server to avoid hanging
          # if there is any exception.
          exporter = Exporter(self)
          try:
            exporter.export_model(*msgs)
          except Exception as e:  # pylint: disable=broad-except
            self._worker_thread_exception = e
            break
        case MethodName.MODEL:
          # Model methods at secondary hosts are simply invoking a device
          # program with pre-defined dummy inputs, so we do not expect any known
          # exception here. An exception must be from low-level
          # software/hardware stack and we crash the job.
          try:
            model_key, model_method, unpadded_shape_str, padded_shape_str = msgs
            logging.info(
                'Received model_key %s, method %s, unpadded_shape %s,'
                ' padded_shape %s',
                model_key,
                model_method,
                unpadded_shape_str,
                padded_shape_str,
            )
            method_obj = self._loaded_models.get_model(model_key).method(
                model_method
            )
            if unpadded_shape_str == 'None':
              unpadded_shape = None
            else:
              unpadded_shape = method_obj.deserialize_input_shape(
                  unpadded_shape_str
              )
            if padded_shape_str == 'None':
              padded_shape = None
            else:
              padded_shape = method_obj.deserialize_input_shape(
                  padded_shape_str
              )
            if unpadded_shape is None:
              # Successful pre-processing.
              assert padded_shape is not None
            else:
              # Presync or failed-preprocessing.
              assert padded_shape is None
              padded_shape = method_obj.get_padded_input_shape(unpadded_shape)
            method_obj.device_compute_with_dummy_data(padded_shape)
          except Exception as e:  # pylint: disable=broad-except
            self._worker_thread_exception = e
            break
        case MethodName.PREFILL_INSERT:
          try:
            model_key, model_method, slot = msgs
            logging.info(
                'Received PREFILL_INSERT model_key %s method %s slot %s',
                model_key,
                model_method,
                slot,
            )
            slot = int(slot)
            method_obj = self._loaded_models.get_model(model_key).method(
                model_method
            )
            _, _, state = method_obj.prefill_with_dummy()
            method_obj.insert(state, slot)
          except Exception as e:  # pylint: disable=broad-except
            self._worker_thread_exception = e
            break
        case MethodName.GENERATE:
          try:
            model_key, model_method = msgs
            logging.info(
                'Received GENERATE model_key %s method %s',
                model_key,
                model_method,
            )
            method_obj = self._loaded_models.get_model(model_key).method(
                model_method
            )
            method_obj.generate_with_dummy()
          except Exception as e:  # pylint: disable=broad-except
            self._worker_thread_exception = e
            break
        case _:
          self._log_exception('Unknown method: %s, msgs: %s', method, msgs)
