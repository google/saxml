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

import asyncio
import threading
from typing import Any, Callable
from unittest import mock

from absl.testing import absltest
import grpc
import numpy as np
import portpicker
from saxml.protobuf import common_pb2
from saxml.protobuf import lm_pb2
from saxml.protobuf import lm_pb2_grpc
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
        bouncer=model_service_base.DormantServerBouncer(
            is_backend_dormant=lambda: False,
            wake_up_backend=lambda: None,
            enable_early_rejection=False,
        ),
        sax_cell='/sax/foo',
        admin_port=portpicker.pick_unused_port(),
        platform_chip='cpu',
        platform_topology='1',
        tags=[],
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

  def test_report_server_dormant_state(self):
    mock_bouncer = self.enter_context(
        mock.patch.object(self._service, '_bouncer', autospec=True)
    )
    mock_bouncer.is_server_dormant.return_value = True

    request = modelet_pb2.GetStatusRequest()
    response = modelet_pb2.GetStatusResponse()

    self._service.get_status(request, response)

    status = response.server_status
    self.assertEqual(
        status.state, modelet_pb2.GetStatusResponse.ServerStatus.DORMANT
    )
    self.assertNotEmpty(status.explanation)

  def test_report_server_active_state(self):
    request = modelet_pb2.GetStatusRequest()
    response = modelet_pb2.GetStatusResponse()

    self._service.get_status(request, response)

    status = response.server_status
    self.assertEqual(
        status.state, modelet_pb2.GetStatusResponse.ServerStatus.ACTIVE
    )
    self.assertNotEmpty(status.explanation)

  def test_report_server_early_rejection_state(self):
    mock_bouncer = self.enter_context(
        mock.patch.object(self._service, '_bouncer', autospec=True)
    )
    mock_bouncer.get_rejected_request_stats.return_value = (
        utils.RequestStats.Stats(
            timespan_sec=20,
            total=400,
            summ=400,
            summ2=400,
            samples=np.array(list(range(400))),
        )
    )

    request = modelet_pb2.GetStatusRequest()
    response = modelet_pb2.GetStatusResponse()

    self._service.get_status(request, response)

    status = response.server_status
    self.assertEqual(
        status.state, modelet_pb2.GetStatusResponse.ServerStatus.DORMANT
    )
    self.assertEqual(status.stats.early_rejection_errors_per_second, 20)


class DormantServerBouncerTest(absltest.TestCase):

  def test_reject_requests_respecting_enabled_flag(self):
    early_rejection_disabled_bouncer = model_service_base.DormantServerBouncer(
        is_backend_dormant=lambda: True,
        wake_up_backend=lambda: None,
        enable_early_rejection=False,
    )

    self.assertTrue(early_rejection_disabled_bouncer.is_server_dormant())
    self.assertFalse(early_rejection_disabled_bouncer.should_reject_request())

  def test_reject_requests_respecting_server_dormant(self):
    dormant_server_bouncer = model_service_base.DormantServerBouncer(
        lambda: True, lambda: None, True
    )
    self.assertTrue(dormant_server_bouncer.is_server_dormant())
    self.assertTrue(dormant_server_bouncer.should_reject_request())
    self.assertEqual(
        dormant_server_bouncer.get_rejected_request_stats(),
        utils.RequestStats.Stats(
            timespan_sec=10.0,
            total=1,
            summ=1.0,
            summ2=1.0,
            samples=np.array([1]),
        ),
    )

  def test_reject_requests_respecting_server_active(self):
    active_server_bouncer = model_service_base.DormantServerBouncer(
        is_backend_dormant=lambda: False,
        wake_up_backend=lambda: None,
        enable_early_rejection=True,
    )
    self.assertFalse(active_server_bouncer.is_server_dormant())
    self.assertFalse(active_server_bouncer.should_reject_request())
    self.assertEqual(
        active_server_bouncer.get_rejected_request_stats().total, 0
    )

  def test_bouncer_wakes_up_server(self):
    is_dormant = True

    def is_backend_dormant() -> bool:
      nonlocal is_dormant
      return is_dormant

    bouncer = model_service_base.DormantServerBouncer(
        is_backend_dormant=is_backend_dormant,
        wake_up_backend=lambda: asyncio.run(asyncio.sleep(0.5)),
        enable_early_rejection=True,
    )

    def simulate_background_server_wake_up() -> None:
      asyncio.run(asyncio.sleep(3.0))
      nonlocal is_dormant
      is_dormant = False

    self.assertTrue(bouncer.is_server_dormant())
    bouncer.wake_up_if_dormant()
    self.assertTrue(bouncer._pending_wake_up)

    simulate_background_server_wake_up()
    bouncer._backend_waker.join()  # Wait for the waking-up settle down.
    self.assertFalse(bouncer.is_server_dormant())
    self.assertFalse(bouncer._pending_wake_up)


class DormantModelServiceTest(absltest.TestCase):

  class FakeModelService(
      model_service_base.ModelServiceGRPC, lm_pb2_grpc.LMServiceServicer
  ):

    def ParseMethodRPCRequest(self, method_name: str, request: Any) -> Any:
      pass

    def FillRPCResponse(
        self, method_name: str, method_outputs: Any, response: Any
    ) -> None:
      pass

    def ServiceName(self) -> str:
      return 'FakeModelService'

    def AddToServer(self, server: Any) -> None:
      pass

    async def Generate(self, request, context):
      resp = lm_pb2.GenerateResponse()

      await self.EnqueueRequest(
          'lm.generate', request.model_key, context, request, resp
      )
      return resp

  def setUp(self):
    super().setUp()
    self._service = self.FakeModelService(
        service_id='fake_model_service',
        batcher=model_service_base.PerMethodBatcher(),
        loader=model_service_base.LoadedModelManager(0),
        bouncer=model_service_base.DormantServerBouncer(
            is_backend_dormant=lambda: True,
            wake_up_backend=lambda: None,
            enable_early_rejection=True,
        ),
    )

  # A tedious fake of grpc.ServicerContext.
  class _FakeRPCServerCtx(grpc.ServicerContext):

    def __init__(self):
      self._code = None
      self._details = None

    def is_active(self) -> bool:
      raise NotImplementedError()

    def time_remaining(self) -> float:
      raise NotImplementedError()

    def cancel(self) -> None:
      raise NotImplementedError()

    def add_callback(self, callback: Callable[[], None]) -> None:
      raise NotImplementedError()

    def invocation_metadata(self) -> None:
      raise NotImplementedError()

    def peer(self) -> str:
      raise NotImplementedError()

    def peer_identities(self) -> None:
      raise NotImplementedError()

    def peer_identity_key(self) -> None:
      raise NotImplementedError()

    def auth_context(self) -> dict[str, list[bytes]]:
      raise NotImplementedError()

    def send_initial_metadata(self, initial_metadata: Any) -> None:
      raise NotImplementedError()

    def set_trailing_metadata(self, trailing_metadata: Any) -> None:
      raise NotImplementedError()

    def trailing_metadata(self) -> None:
      raise NotImplementedError()

    def abort(self, code: grpc.StatusCode, details: str) -> None:
      raise NotImplementedError()

    def abort_with_status(self, status: grpc.Status) -> None:
      raise NotImplementedError()

    def set_code(self, code: grpc.StatusCode) -> None:
      self._code = code

    def set_details(self, details: str) -> None:
      self._details = details

    def details(self) -> str:
      return self._details

    def disable_next_message_compression(self) -> None:
      raise NotImplementedError()

    def code(self) -> grpc.StatusCode:
      return self._code

  def test_dormant_server_rejects_model_requests(self):
    context = self._FakeRPCServerCtx()
    asyncio.run(
        self._service.Generate(
            lm_pb2.GenerateRequest(model_key='anymodel', text='hello world'),
            context,
        )
    )
    self.assertEqual(context.code(), grpc.StatusCode.UNAVAILABLE)
    self.assertIn('dormant', context.details())


class ModelServicesRunnerJoinTimingTest(absltest.TestCase):

  def test_delays_after_start_when_keeping_statically_loaded_models(self):
    port = portpicker.pick_unused_port()
    with mock.patch.object(
        model_service_base.ModeletService, 'after_start'
    ) as mock_after_start:
      runner = model_service_base.ModelServicesRunner(
          is_primary_process=True,
          port=port,
          sax_cell='/sax/foo',
          keep_statically_loaded_models=True,
      )
      orig_grpc_start = runner._grpc_server.start
      grpc_started = threading.Event()

      async def _start_and_signal():
        await orig_grpc_start()
        grpc_started.set()

      with mock.patch.object(
          runner._grpc_server, 'start', side_effect=_start_and_signal
      ):
        runner.start()
        try:
          self.assertTrue(grpc_started.wait(timeout=10))
          # Flush the event loop so _serve() advances past after_start().
          asyncio.run_coroutine_threadsafe(
              asyncio.sleep(0), runner._aio_loop
          ).result(timeout=10)
          mock_after_start.assert_not_called()

          runner.on_initial_models_load_completion()
          mock_after_start.assert_called_once()
        finally:
          runner.stop()
          runner.wait()

  def test_calls_after_start_on_start_without_static_models(self):
    port = portpicker.pick_unused_port()
    joined = threading.Event()
    with mock.patch.object(
        model_service_base.ModeletService,
        'after_start',
        side_effect=joined.set,
    ) as mock_after_start:
      runner = model_service_base.ModelServicesRunner(
          is_primary_process=True,
          port=port,
          sax_cell='/sax/foo',
          keep_statically_loaded_models=False,
      )
      runner.start()
      try:
        self.assertTrue(joined.wait(timeout=10))
        mock_after_start.assert_called_once()

        runner.on_initial_models_load_completion()
        mock_after_start.assert_called_once()
      finally:
        runner.stop()
        runner.wait()


class UpdateLoadedTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._service = model_service_base.ModeletServiceGRPC(
        service_port=portpicker.pick_unused_port(),
        debug_port=None,
        batcher=model_service_base.PerMethodBatcher(),
        loader=model_service_base.LoadedModelManager(0),
        bouncer=model_service_base.DormantServerBouncer(
            is_backend_dormant=lambda: False,
            wake_up_backend=lambda: None,
            enable_early_rejection=False,
        ),
        sax_cell='/sax/foo',
        admin_port=portpicker.pick_unused_port(),
        platform_chip='cpu',
        platform_topology='1',
        tags=[],
    )
    mock_loader = self.enter_context(
        mock.patch.object(self._service, '_loader', autospec=True)
    )
    mock_loader.maybe_get_model.return_value = mock.Mock()
    mock_loader.maybe_get_model_metadata.return_value = dict(
        checkpoint_path='/ckpt/step_0',
        model_path='test.Model',
        acls={},
        overrides={},
    )
    self._mock_unload = self.enter_context(
        mock.patch.object(
            self._service,
            'unload',
            side_effect=lambda rpc, req, resp, done: done(utils.ok()),
        )
    )
    self._mock_load = self.enter_context(
        mock.patch.object(self._service, 'Load', new_callable=mock.AsyncMock)
    )
    self._mock_update = self.enter_context(
        mock.patch.object(
            self._service,
            'update',
            side_effect=lambda req, done: done(utils.ok()),
        )
    )

  def _update_loaded(self, checkpoint_path: str) -> None:
    asyncio.run(
        self._service.UpdateLoaded(
            modelet_pb2.UpdateLoadedRequest(
                model_key='/sax/foo/bar', checkpoint_path=checkpoint_path
            ),
            mock.create_autospec(grpc.ServicerContext, instance=True),
        )
    )

  def test_skips_reload_when_checkpoint_already_loaded(self):
    self._update_loaded('/ckpt/step_0')

    self._mock_unload.assert_not_called()
    self._mock_load.assert_not_called()
    self._mock_update.assert_called_once()

  def test_reloads_when_checkpoint_changes(self):
    self._update_loaded('/ckpt/step_80')

    self._mock_unload.assert_called_once()
    self._mock_load.assert_awaited_once()
    self.assertEqual(
        self._mock_load.await_args.args[0].checkpoint_path, '/ckpt/step_80'
    )
    self._mock_update.assert_not_called()


if __name__ == '__main__':
  absltest.main()
