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
"""Utilities routines."""

import dataclasses
import queue
import threading
import time
from typing import Any, Callable, Optional, Protocol, Sequence, Tuple

import grpc

from google.protobuf import message


@dataclasses.dataclass
class Status:
  code: grpc.StatusCode
  details: str = ''

  def ok(self) -> bool:
    return self.code == grpc.StatusCode.OK


class StatusCallback(Protocol):
  """A callback on a status, and optionally other arguments."""

  def __call__(self, status: Status, *args, **kwargs) -> None:
    ...


Callback = Callable[..., None]
TracerPrintCallback = Callable[[str], None]


def get_current_trace_printer() -> TracerPrintCallback:
  # No-op tracer.
  return lambda _: None


class RPCContext:
  """Object to query the current status of the RPC."""

  def username(self) -> Optional[str]:
    raise NotImplementedError()

  def should_cancel(self) -> bool:
    raise NotImplementedError()


class RPCContextGRPC(RPCContext):
  """gRPC version of RPCContext."""

  def __init__(self, context: grpc.ServicerContext):
    self._context = context
    self._username: Optional[str] = None
    self._init_cred(context)

  def _init_cred(self, context: grpc.ServicerContext) -> None:
    pass

  def username(self) -> Optional[str]:
    return self._username

  def should_cancel(self) -> bool:
    timeout = self._context.time_remaining()
    return timeout is not None and timeout <= 0


@dataclasses.dataclass
class RpcQueueTask:
  """Represents one RPC request in the queue."""
  rpc: Optional[RPCContext]
  request: Optional[message.Message]
  response: Optional[message.Message]
  done: Optional[StatusCallback]
  tc: Optional[TracerPrintCallback]


def traceprint_all(rpc_tasks: Sequence[RpcQueueTask], msg: str):
  """Prints `msg` in the tracer of all rpc_tasks, if present."""
  for rpc_task in rpc_tasks:
    if rpc_task.tc:
      rpc_task.tc(msg)


class RpcQueue():
  """A queue of RPC requests."""

  def __init__(self, batching_wait_secs: Optional[float] = None):
    self._queue: queue.SimpleQueue[RpcQueueTask] = queue.SimpleQueue()
    self._batching_wait_secs = batching_wait_secs

  def send(self,
           rpc: Optional[RPCContext],
           request: Optional[message.Message],
           response: Optional[message.Message],
           done: Optional[StatusCallback],
           tc: Optional[TracerPrintCallback] = None):
    """Called from RPC handler to schedule a task for processing.

    Args:
      rpc: the rpc object.
      request: request protocol message
      response: response protocol message
      done: A callback when the rpc handling is done.
      tc: optional TracerPrintCallback object.
    """
    self._queue.put(RpcQueueTask(rpc, request, response, done, tc))

  def take_batch(self, batch_size: int) -> list[RpcQueueTask]:
    """Returns up to batch_size RpcQueueTask objects from the queue.

    The call may block indefinitely when the queue is empty.

    Args:
      batch_size: number of tasks

    Returns:
      A list of RpcQueueTask.
    """
    batch = []
    batch_begin_time = time.time()
    while len(batch) < batch_size:
      try:
        # Only blocks for the 1st item in the batch.
        if batch:
          timeout = (
              self._batching_wait_secs - time.time() + batch_begin_time
              if self._batching_wait_secs
              else 0
          )
          if timeout <= 0:
            task = self._queue.get_nowait()
          else:
            task = self._queue.get(timeout=timeout)
        else:
          task = self._queue.get()
          batch_begin_time = time.time()
      except queue.Empty:
        break

      if task.rpc is not None and task.rpc.should_cancel():
        if task.done is not None:
          task.done(cancelled())
        continue

      if task.tc:
        task.tc(f'RpcQueueTask Dequeued (qlen: {self._queue.qsize()})')

      batch.append(task)

    return batch


class ThreadPool:
  """A simple fixed sized threadpool.

  NOTE: We don't bother to proper shutdown this pool. run() also doesn't provide
  mechanism for the caller to get notified when the task is done. These
  simplication make the implementation rather straightforward compared to
  multiprocessing.dummy.Pool or concurrent.futures.Executor.

  TODO(zhifengc): adds mechanism to shutdown the pool when needed.
  """

  def __init__(self, num_threads: int, thread_name_prefix: str = ''):
    self._queue: queue.SimpleQueue[Tuple[Callback, Any]] = queue.SimpleQueue()
    for i in range(num_threads):
      t = threading.Thread(
          target=self._do_work, daemon=True, name=f'{thread_name_prefix}_{i}')
      t.start()

  def run(self, func: Callback, args: Any = ()):
    assert func is not None
    assert args is not None
    self._queue.put((func, args))

  def _do_work(self):
    while True:
      func, args = self._queue.get()
      func(*args)


class Notification:
  """A notification helper."""

  def __init__(self):
    self._cv = threading.Condition()
    self._notified = False

  def notify(self):
    with self._cv:
      self._notified = True
      self._cv.notify_all()

  def wait(self):
    with self._cv:
      while not self._notified:
        self._cv.wait()


class Admissioner:
  """A semaphore with a shutdown method."""

  def __init__(self, limit):
    self._limit = limit
    self._count = 0
    self._cv = threading.Condition()
    self._active = True
    self._shutdown = False

  def acquire(self, blocking: bool = True) -> Tuple[bool, bool]:
    """Acquires resource.

    Args:
      blocking: whether the invocation is blocking.

    Returns:
      A tuple of 2 bools. The first indicates if it's successful, and the second
      indicates if the resource is still active.
    """
    with self._cv:
      while self._count >= self._limit:
        if not blocking:
          return False, self._active
        self._cv.wait()
      if not self._active:
        return False, False
      self._count += 1
      return True, True

  def release(self):
    with self._cv:
      self._count -= 1
      self._cv.notify_all()

  def is_shutdown(self):
    return self._shutdown

  def shutdown(self):
    with self._cv:
      self._active = False
      while self._count > 0:
        self._cv.wait()
      self._shutdown = True


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
