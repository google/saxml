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
"""Utilities for cross-host synchronization in multi-host setup."""

import threading
from typing import List, Optional, Tuple

from absl import logging
import grpc
from saxml.protobuf import internal_pb2
from saxml.protobuf import internal_pb2_grpc
from saxml.server import utils
from saxml.server.spmd_backend import SPMDBackend

_MAX_RING_BUF_SIZE = 128
_RPC_SYNC_TIMEOUT = 5.0


class MessageRingBuffer:
  """A ring buffer for message syncs.

  Messages can be added (pushed) out of order, and a message can be added
  multiple times where duplicates are dropped. Dequeue (pop) always happens in
  order.
  """

  def __init__(self, max_size: int):
    self._max_size = max_size
    self._buffer: List[Optional[str]] = [None] * max_size
    self._cv = threading.Condition()
    # _head_seqno is 1 + the largest successful push seqno in the past.
    self._head_seqno = 0
    # _tail_seqno is the next seqno to pop.
    self._tail_seqno = 0
    # _pending_out_of_order is the set of unseen seqnos which are smaller than
    # the largest successful push seqno in the past.
    # Invariants:
    #  _tail_seqno <= elements in _pending_out_of_order < self._head_seqno - 1
    self._pending_out_of_order = set()

  def push(self, seqno: int, message: str, blocking: bool) -> bool:
    """Adds a message to the ring buffer, and returns whether push succeeds."""
    offset = seqno % self._max_size
    with self._cv:
      if self._head_seqno > seqno and seqno not in self._pending_out_of_order:
        # This is a duplicate. Consider this as a successful push.
        return True
      while (
          self._buffer[offset] is not None
          or seqno >= self._tail_seqno + self._max_size
      ):
        if not blocking:
          return False
        self._cv.wait()
      self._buffer[offset] = message
      if self._head_seqno == seqno:
        self._head_seqno += 1
      elif self._head_seqno > seqno:
        self._pending_out_of_order.remove(seqno)
      else:
        for unseen in range(self._head_seqno, seqno):
          self._pending_out_of_order.add(unseen)
        self._head_seqno = seqno + 1
      self._cv.notify_all()
      return True

  def pop(self) -> str:
    """Dequeues a message from the ring buffer."""
    with self._cv:
      offset = self._tail_seqno % self._max_size
      while self._buffer[offset] is None:
        self._cv.wait()
      message = self._buffer[offset]
      self._buffer[offset] = None
      self._tail_seqno += 1
      self._cv.notify_all()
      assert message is not None
      return message


class MultiHostSyncService(internal_pb2_grpc.MultiHostSyncService):
  """Service that runs at secondary hosts to accept sync messages."""

  def __init__(self, rb: MessageRingBuffer):
    self._rb = rb

  async def Sync(
      self, request: internal_pb2.SyncRequest, context: grpc.ServicerContext
  ):
    if not self._rb.push(request.seqno, request.text, blocking=False):
      context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
      context.set_details(f'Message ringbuffer full for seqno {request.seqno}')
    return internal_pb2.SyncResponse()


class MultiHostSync:
  """Implements cross-host synchronization on a sequence of messages."""

  def __init__(
      self,
      is_primary: bool,
      server: grpc.aio.Server,
      my_ipport: str,
      spmd_backend: SPMDBackend,
  ):
    self._rb = MessageRingBuffer(_MAX_RING_BUF_SIZE)
    self._is_primary = is_primary
    self._my_ipport = my_ipport
    self._spmd_backend = spmd_backend
    self._primary_seqno: int = 0
    self._device_receive_seqno: int = 0
    self._device_receive_thread_pool: Optional[utils.ThreadPool] = None
    if not is_primary:
      internal_pb2_grpc.add_MultiHostSyncServiceServicer_to_server(
          MultiHostSyncService(self._rb), server
      )
      self._device_receive_thread_pool = utils.ThreadPool(num_threads=2)
    self._host_send_threads: List[threading.Thread] = []
    self._stubs = []
    self._host_send_cv: threading.Condition = threading.Condition()
    self._next_message_via_host: Tuple[int, str] = (-1, '')
    # A special sequence number to tell sending threads to stop.
    self._end_special_seqno = -2
    self._live_outgoing_rpcs: int = 0

  def initialize(
      self, channel_creds: Optional[grpc.ChannelCredentials]
  ) -> None:
    """Initializes the connections between hosts."""
    if self._spmd_backend.spmd_host_count() == 1:
      return
    # Communicate the host addresses to each other via device sync.
    other_addrs = []
    for idx in range(self._spmd_backend.spmd_host_count()):
      if idx == self._spmd_backend.spmd_host_index():
        # Send my address.
        self._spmd_backend.send_via_device(self._my_ipport)
      else:
        # Receive peer's address.
        other_addrs.append(self._spmd_backend.receive_via_device())
        logging.info('Received SPMD peer address %s', other_addrs[-1])

    def _send_loop(stub):
      last_seqno = -1
      while True:
        with self._host_send_cv:
          while True:
            seqno, msg = self._next_message_via_host
            if seqno == self._end_special_seqno:
              return
            if seqno <= last_seqno:
              self._host_send_cv.wait()
            else:
              break
        try:
          stub.Sync(internal_pb2.SyncRequest(seqno=seqno, text=msg))
        except Exception as e:  # pylint: disable=broad-except
          # This is not critical because it can use device sync.
          logging.warning('Error during RPC sync with secondary host: %s', e)
        last_seqno = seqno
        with self._host_send_cv:
          self._live_outgoing_rpcs -= 1
          if self._live_outgoing_rpcs <= 0:
            self._host_send_cv.notify_all()

    if self._is_primary:
      # Create RPC stubs to secondary hosts and start sender threads.
      channels = []
      channel_ready_futures = []
      # Create the channel and stubs here instead of in the sender threads, so
      # that exceptions would be thrown in the current thread.
      for addr in other_addrs:
        logging.info('Connecting to %s', addr)
        if channel_creds is None:
          channels.append(grpc.insecure_channel(addr))
        else:
          channels.append(grpc.secure_channel(addr, channel_creds))
        channel_ready_futures.append(grpc.channel_ready_future(channels[-1]))
      for channel, ready_future in zip(channels, channel_ready_futures):
        ready_future.result()
        stub = internal_pb2_grpc.MultiHostSyncServiceStub(channel)
        self._host_send_threads.append(
            threading.Thread(target=_send_loop, args=(stub,))
        )
        self._host_send_threads[-1].start()
      logging.info('Connected to all secondary hosts')

  def stop(self) -> None:
    if not self._host_send_threads:
      return
    with self._host_send_cv:
      while self._live_outgoing_rpcs > 0:
        self._host_send_cv.wait()
      self._next_message_via_host = (self._end_special_seqno, '')
      self._host_send_cv.notify_all()

  def wait(self) -> None:
    for thread in self._host_send_threads:
      thread.join()

  def send(self, message: str, skip_host_sync: bool) -> None:
    """Sends a message to secondary hosts."""
    if self._spmd_backend.spmd_host_count() == 1:
      return
    assert self._is_primary
    seqno = self._primary_seqno
    self._primary_seqno += 1

    # Exclude seqno in device sync to allow cache hits.
    self._spmd_backend.send_via_device(message)

    if skip_host_sync:
      return
    with self._host_send_cv:
      if self._live_outgoing_rpcs > 0:
        logging.info(
            'Skipping send via host, %s unfinished RPC requests',
            self._live_outgoing_rpcs,
        )
        return
      # Notify the sender threads of the new message.
      self._live_outgoing_rpcs = len(self._host_send_threads)
      self._next_message_via_host = (seqno, message)
      self._host_send_cv.notify_all()

  def receive(self) -> str:
    """Receives a message from the primary host."""
    assert not self._is_primary
    seqno = self._device_receive_seqno

    def _done(message: str) -> None:
      self._rb.push(seqno, message, blocking=True)

    self._spmd_backend.receive_via_device_async(
        self._device_receive_thread_pool, _done
    )
    self._device_receive_seqno += 1
    return self._rb.pop()
