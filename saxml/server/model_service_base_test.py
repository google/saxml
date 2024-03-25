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
"""Tests for model_service_base."""

from unittest import mock

from absl.testing import absltest
import numpy as np
import portpicker
from saxml.protobuf import common_pb2
from saxml.protobuf import modelet_pb2
from saxml.server import model_service_base
from saxml.server import utils


class MethodKeyTest(absltest.TestCase):

  def test_method_key_name(self):
    key = model_service_base.MethodKey(model_service_base.MethodName.LOAD)
    self.assertEqual(key.method_name(), 'load')

    key = model_service_base.MethodKey(
        model_service_base.MethodName.MODEL,
        'method',
    )
    self.assertEqual(key.method_name(), 'method')


class GetStatusTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._service = model_service_base.ModeletService(
        service_port=portpicker.pick_unused_port(),
        debug_port=None,
        batcher=model_service_base.PerMethodBatcher(),
        loader=model_service_base.LoadedModelManager(0),
        sax_cell='/sax/foo',
        admin_port=portpicker.pick_unused_port(),
        platform_chip='cpu',
        platform_topology='1',
        tags=[]
    )
    mock_loader = self.enter_context(
        mock.patch.object(self._service, '_loader', autospec=True)
    )
    mock_loader.get_status.return_value = {
        '/sax/foo/bar': common_pb2.ModelStatus.LOADED,
    }
    mock_batcher = self.enter_context(
        mock.patch.object(self._service, '_batcher', autospec=True)
    )
    mock_batcher.get_method_stats.return_value = [
        (
            model_service_base.MethodKey(
                model_service_base.MethodName.MODEL,
                'method',
                'service',
                '/sax/foo/bar',
            ),
            utils.RequestStats.Stats(
                timespan_sec=8,
                total=100,
                summ=4950,
                summ2=328350,
                samples=np.array(list(range(100))),
            ),
            utils.RequestStats.Stats(
                timespan_sec=8, total=1, summ=1, summ2=1, samples=np.array([1])
            ),
            10,
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        ),
    ]

  def test_has_method_stats_if_requested(self):
    request = modelet_pb2.GetStatusRequest(include_method_stats=True)
    response = modelet_pb2.GetStatusResponse()

    self._service.get_status(request, response)

    self.assertLen(response.models, 1)
    model = response.models[0]
    self.assertLen(model.method_stats, 1)
    method_stats = model.method_stats[0]
    self.assertEqual('method', method_stats.method)
    self.assertEqual(0.125, method_stats.errors_per_second)
    self.assertEqual(12.5, method_stats.successes_per_second)
    self.assertEqual(49.5, method_stats.mean_latency_on_success_per_second)
    self.assertEqual(49.5, method_stats.p50_latency_on_success_per_second)
    self.assertAlmostEqual(
        94.05, method_stats.p95_latency_on_success_per_second, 3
    )
    self.assertAlmostEqual(
        98.01, method_stats.p99_latency_on_success_per_second, 3
    )

  def test_no_method_stats_if_not_requested(self):
    request = modelet_pb2.GetStatusRequest()
    response = modelet_pb2.GetStatusResponse()

    self._service.get_status(request, response)

    self.assertLen(response.models, 1)
    model = response.models[0]
    self.assertEmpty(model.method_stats)


if __name__ == '__main__':
  absltest.main()
