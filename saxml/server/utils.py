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

import collections
import dataclasses
import queue
import threading
import time
from typing import Any, Callable, Deque, List, Optional, Protocol, Sequence, Tuple

import grpc
import jax
import numpy as np
import numpy.typing as npt
from saxml.server import status as server_status

from google.protobuf import message


Status = server_status.Status
ok = server_status.ok
cancelled = server_status.cancelled
invalid_arg = server_status.invalid_arg
internal_error = server_status.internal_error
not_found = server_status.not_found
permission_denied = server_status.permission_denied
resource_exhausted = server_status.resource_exhausted
unimplemented = server_status.unimplemented
already_exists = server_status.already_exists
unavailable = server_status.unavailable

ClockTime = Callable[[], float]


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
  release_device_resource: Optional[Callable[[], None]]
  done: Optional[StatusCallback]
  tc: Optional[TracerPrintCallback]
  # Any additional data.
  aux: dict[str, Any] = dataclasses.field(default_factory=dict)


def traceprint_all(rpc_tasks: Sequence[RpcQueueTask], msg: str):
  """Prints `msg` in the tracer of all rpc_tasks, if present."""
  for rpc_task in rpc_tasks:
    if rpc_task.tc:
      rpc_task.tc(msg)


class RpcQueue:
  """A queue of RPC requests."""

  def __init__(self, batching_wait_secs: Optional[float] = None):
    self._queue: queue.SimpleQueue[RpcQueueTask] = queue.SimpleQueue()
    self._batching_wait_secs = batching_wait_secs

  def send(
      self,
      rpc: Optional[RPCContext],
      request: Optional[message.Message],
      response: Optional[message.Message],
      release_device_resource: Optional[Callable[[], None]],
      done: Optional[StatusCallback],
      tc: Optional[TracerPrintCallback] = None,
      aux: Optional[dict[str, Any]] = None,
  ):
    """Called from RPC handler to schedule a task for processing.

    Args:
      rpc: the rpc object.
      request: request protocol message
      response: response protocol message
      release_device_resource: callback for releasing device resource to control
        the maximum live batches.
      done: A callback when the rpc handling is done.
      tc: optional TracerPrintCallback object.
      aux: any additional data.
    """
    self._queue.put(
        RpcQueueTask(
            rpc,
            request,
            response,
            release_device_resource,
            done,
            tc,
            aux=aux,
        )
    )

  def take_batch(self, batch_size: int) -> List[RpcQueueTask]:
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
          target=self._do_work, daemon=True, name=f'{thread_name_prefix}_{i}'
      )
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

  def wait(self, timeout: float | None = None):
    with self._cv:
      while not self._notified:
        if self._cv.wait(timeout):
          assert self._notified
        else:
          raise TimeoutError()


class Admissioner:
  """A semaphore with a shutdown method."""

  def __init__(self, limit):
    self._limit = limit
    self._count = 0
    self._cv = threading.Condition()
    self._active = True
    self._shutdown = False

  def acquire(self, blocking: bool = True, count: int = 1) -> Tuple[bool, bool]:
    """Acquires resource.

    Args:
      blocking: whether the invocation is blocking.
      count: number of resources to acquire.

    Returns:
      A tuple of 2 bools. The first indicates if it's successful, and the second
      indicates if the resource is still active.
    """
    with self._cv:
      while self._count + count > self._limit:
        if not blocking:
          return False, self._active
        self._cv.wait()
      if not self._active:
        return False, False
      self._count += count
      return True, True

  def release(self, count: int = 1):
    with self._cv:
      self._count -= count
      self._cv.notify_all()

  def is_shutdown(self):
    return self._shutdown

  def shutdown(self):
    with self._cv:
      self._active = False
      while self._count > 0:
        self._cv.wait()
      self._shutdown = True


class RequestStats:
  """RequestStats keeps track of request latencies in the recent past.

  E.g.,
    # Keeps track of request latencies in the last 60 seconds.
    stats = RequestStats(60)

    # request just finished
    stats.add(request_finish_time - request_start_time)

    # Looks at the statistics such as mean/std of the latency together
    # with a few samples. These samples can be used in computing latency
    # percentile.
    result = stats.get(100)
    print(result.mean(), result.std, np.percentile(result.samples, 50))
  """

  clock_time: ClockTime
  timespan_sec: np.float64

  @dataclasses.dataclass(frozen=True)
  class _Item:
    timestamp_sec: np.float64
    duration_sec: np.float64

  queue: Deque[_Item]

  # Basic statistics of items in deque.
  total: np.int64  # sum(deque[*].duration_sec)
  summ: np.float64  # sum(deque[*].duration_sec)
  summ2: np.float64  # sum(deque[*].duration_sec^2)

  def __init__(self, timespan_sec: np.float64, clock: ClockTime = time.time):
    """Constructs a RequestStats object.

    Args:
      timespan_sec: Keeps track latencies observed in the last these many
        seconds.
      clock: A callback returns the current time. Useful for testing.
    """
    assert timespan_sec > 0.0
    self.timespan_sec = timespan_sec
    self.clock_time = clock
    self.queue = collections.deque()
    self.total = 0  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    self.summ = 0.0  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    self.summ2 = 0.0  # pytype: disable=annotation-type-mismatch  # numpy-scalars

  def _gc(self, now_sec):
    """Garbage collection routine to keep self.queue finite size."""
    while self.queue and (
        now_sec - self.queue[0].timestamp_sec >= self.timespan_sec
    ):
      item = self.queue.popleft()
      self.total -= 1  # pytype: disable=annotation-type-mismatch  # numpy-scalars
      self.summ -= item.duration_sec
      self.summ2 -= np.square(item.duration_sec)

  def add(self, duration_sec: float):
    """Records one request's latency."""
    now_sec = self.clock_time()
    if self.queue:
      # Makes sure clock doesn't go back.
      now_sec = max(now_sec, self.queue[-1].timestamp_sec)
    item = self._Item(timestamp_sec=now_sec, duration_sec=duration_sec)  # pytype: disable=wrong-arg-types  # numpy-scalars
    self.queue.append(item)
    self.total += 1  # pytype: disable=annotation-type-mismatch  # numpy-scalars
    self.summ += item.duration_sec
    self.summ2 += np.square(item.duration_sec)
    self._gc(now_sec)

  @dataclasses.dataclass(frozen=True)
  class Stats:
    """Statistics summary for request latencies during the timespan."""

    # The time span this stats covers.
    timespan_sec: np.float64

    # The total number of recorded requests during the timespan.
    total: np.int64

    # The sum of request latencies during the timespan.
    summ: np.float64

    # The sum of request latencies squared during the timespan.
    summ2: np.float64

    # The selected samples of request latencies.
    samples: npt.NDArray[np.float64]

    def rate(self) -> np.float64:
      return self.total / self.timespan_sec

    def mean(self) -> np.float64:
      if self.total == 0:
        return 0.0  # pytype: disable=bad-return-type  # numpy-scalars
      else:
        return self.summ / self.total

    def std(self) -> np.float64:
      if self.total == 0:
        return 0.0  # pytype: disable=bad-return-type  # numpy-scalars
      else:
        return np.sqrt(self.summ2 / self.total - np.square(self.mean()))

  def get(self, max_samples: Optional[int] = None) -> Stats:
    """Returns a summarized view of the latency statistics."""
    self._gc(self.clock_time())
    samples = np.array([i.duration_sec for i in self.queue])
    if max_samples is not None and len(samples) > max_samples:
      samples = np.random.choice(samples, max_samples, replace=False)
    return self.Stats(  # pytype: disable=wrong-arg-types  # numpy-scalars
        timespan_sec=self.timespan_sec,
        total=self.total,
        summ=self.summ,
        summ2=self.summ2,
        samples=samples,
    )


def is_mock_tpu_backend() -> bool:
  """Checks if a mock TPU backend is detected.

  Returns:
    True if Mock TPU backend detected.
  """
  return 'MOCK' in str(jax.devices()[0])


def is_backend_dormant() -> bool:
  """Checks if the backend is dormant.

  Computation framework supports temporary suspend/resume functionality can
  override this function so provide a more accurate status signal to admin
  server for dormant detection.

  Returns:
    True if and only if the backend is dormant.
  """
  return False


def trigger_backend_resumption() -> None:
  """Trigger backend resumption if it is suspended."""
  pass
