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
"""Tests for multi_host_sync."""

import threading
import time

from absl.testing import absltest
from saxml.server import multi_host_sync


class MessageRingBufferTest(absltest.TestCase):

  def test_in_order_push(self):
    rb = multi_host_sync.MessageRingBuffer(2)
    rb.push(0, 'a', blocking=True)
    self.assertTrue(rb.push(1, 'b', blocking=False))
    # Duplicate.
    self.assertTrue(rb.push(1, 'b', blocking=False))
    # Out of space.
    self.assertFalse(rb.push(2, 'c', blocking=False))

    self.assertEqual(rb.pop(), 'a')
    self.assertEqual(rb.pop(), 'b')

    # Now there is space for 2.
    self.assertTrue(rb.push(2, 'c', blocking=False))

  def test_out_of_order_push(self):
    rb = multi_host_sync.MessageRingBuffer(2)
    # Cannot push to position 2 (== max_size) before 0 is consumed.
    self.assertFalse(rb.push(2, 'c', blocking=False))
    rb.push(1, 'b', blocking=True)
    self.assertTrue(rb.push(0, 'a', blocking=False))

    self.assertEqual(rb.pop(), 'a')
    self.assertEqual(rb.pop(), 'b')

    # Now there is space for 2.
    self.assertTrue(rb.push(2, 'c', blocking=False))

  def test_concurrent_push_and_pop(self):
    rb = multi_host_sync.MessageRingBuffer(4)

    def _pusher1():
      for seqno in [1, 3, 2, 7, 5, 6, 4]:
        rb.push(seqno, str(seqno), blocking=True)
        time.sleep(0.01)

    def _pusher2():
      for seqno in [1, 2, 0, 3, 8]:
        rb.push(seqno, str(seqno), blocking=True)
        time.sleep(0.01)

    pop_results = []

    def _popper():
      for _ in range(9):
        pop_results.append(rb.pop())

    pusher1 = threading.Thread(target=_pusher1)
    pusher2 = threading.Thread(target=_pusher2)
    popper = threading.Thread(target=_popper)
    pusher1.start()
    pusher2.start()
    popper.start()
    pusher1.join()
    pusher2.join()
    popper.join()
    self.assertEqual(pop_results, [str(i) for i in range(9)])


if __name__ == '__main__':
  absltest.main()
