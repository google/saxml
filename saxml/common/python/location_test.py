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
"""Tests for location."""

import time

from absl.testing import absltest
import grpc
import portpicker
from saxml.common.platform import env
from saxml.common.python import location
from saxml.common.python import testutil
from saxml.protobuf import admin_pb2
from saxml.protobuf import admin_pb2_grpc


class LocationTest(absltest.TestCase):

  def test_join(self):
    sax_cell = '/sax/test-join-py'
    port = portpicker.pick_unused_port()
    testutil.StartLocalTestCluster(sax_cell, testutil.ModelType.Language, port)

    model_addr = 'localhost:10000'
    specs = admin_pb2.ModelServer()
    location.Join(sax_cell, model_addr, '', specs.SerializeToString())

    time.sleep(3)  # wait for the initial Join to happen
    admin_addr = 'localhost:' + str(port)
    with env.create_grpc_channel(admin_addr) as channel:
      grpc.channel_ready_future(channel).result()
      stub = admin_pb2_grpc.AdminStub(channel)
      req = admin_pb2.WatchLocRequest(seqno=0)
      resp = stub.WatchLoc(req)
      result = resp.result
      self.assertTrue(result.has_fullset)
      model_addrs = []
      for value in result.values:
        model_addrs.append(value)
      for change in result.changelog:
        if change.HasField('addition'):
          model_addrs.append(change.addition)
        if change.HasField('deletion'):
          model_addrs.remove(change.deletion)
      # NOTE: testutil.StartLocalTestCluster starts 1 model server. We let
      # another server join. So the total is 2.
      self.assertLen(model_addrs, 2)
      self.assertIn(model_addr, model_addrs)

    testutil.StopLocalTestCluster(sax_cell)

  def test_join_start_admin(self):
    sax_cell = '/sax/test-join-start-admin-py'
    port = portpicker.pick_unused_port()
    testutil.SetUp(sax_cell)

    model_addr = 'localhost:10000'
    specs = admin_pb2.ModelServer()
    location.Join(
        sax_cell, model_addr, '', specs.SerializeToString(), admin_port=port
    )

    time.sleep(3)  # wait for the initial Join to happen
    admin_addr = 'localhost:' + str(port)
    with env.create_grpc_channel(admin_addr) as channel:
      grpc.channel_ready_future(channel).result()
      stub = admin_pb2_grpc.AdminStub(channel)
      req = admin_pb2.ListRequest()
      # Make sure the newly started admin can respond to List.
      stub.List(req)

  def test_join_fail(self):
    with self.assertRaises(RuntimeError):
      location.Join('/sax/test-join-fail-py', 'localhost:10000', '', '')


if __name__ == '__main__':
  absltest.main()
