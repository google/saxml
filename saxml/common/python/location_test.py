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
    location.Join(sax_cell, model_addr, specs.SerializeToString())

    time.sleep(3)  # wait for the initial Join to happen
    admin_addr = 'localhost:' + str(port)
    with env.create_grpc_channel(admin_addr) as channel:
      grpc.channel_ready_future(channel).result()
      stub = admin_pb2_grpc.AdminStub(channel)
      req = admin_pb2.FindLocRequest(up_to=2)
      resp = stub.FindLoc(req)
      # NOTE: testutil.StartLocalTestCluster starts 1 model server. We let
      # another server join. So the total is 2.
      self.assertLen(resp.modelet_addresses, 2)
      self.assertIn(model_addr, resp.modelet_addresses)

    testutil.StopLocalTestCluster(sax_cell)

  def test_join_fail(self):
    with self.assertRaises(RuntimeError):
      location.Join('/sax/test', 'localhost:10000', '')


if __name__ == '__main__':
  absltest.main()
