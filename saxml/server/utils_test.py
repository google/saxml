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
"""Tests for utils."""

from absl.testing import absltest

import numpy as np
from saxml.server import utils


class _TestClock:
  _now: float = 1.6e9

  def advance(self, seconds: float):
    self._now += seconds

  def now(self):
    return self._now


class RequestStatsTest(absltest.TestCase):

  def testBasic(self):
    clock = _TestClock()
    timespan = 60
    tick = 1.0
    stats = utils.RequestStats(timespan, clock.now)

    result = stats.get(100)
    self.assertEqual(0.0, result.mean())
    self.assertEqual(0.0, result.std())

    for i in range(100):
      clock.advance(tick)
      stats.add(i * 0.1)

    result = stats.get(100)
    self.assertEqual(result.total, timespan / tick)
    self.assertLen(result.samples, result.total)  # All sampled.
    self.assertEqual(result.summ, np.sum(np.arange(40, 100)) * 0.1)
    np.testing.assert_allclose(
        result.summ2,
        np.sum(np.square(np.arange(40, 100) * 0.1)),
    )

    result = stats.get(30)
    self.assertEqual(result.total, timespan / tick)
    self.assertLen(result.samples, 30)
    self.assertEqual(result.summ, np.sum(np.arange(40, 100)) * 0.1)
    np.testing.assert_allclose(
        result.summ2,
        np.sum(np.square(np.arange(40, 100) * 0.1)),
    )

  def testNormal(self):
    clock = _TestClock()
    timespan = 600
    tick = 0.01
    stats = utils.RequestStats(timespan, clock.now)

    mean, std = 0.1, 0.05
    for dur in np.random.normal(mean, std, int(timespan / tick)):
      clock.advance(tick)
      stats.add(dur)

    result = stats.get(5000)
    np.testing.assert_allclose(mean, result.mean(), rtol=1e-1)
    np.testing.assert_allclose(mean, np.mean(result.samples), rtol=1e-1)
    np.testing.assert_allclose(std, result.std(), rtol=1e-1)
    np.testing.assert_allclose(std, np.std(result.samples), rtol=1e-1)
    np.testing.assert_allclose(1.0 / tick, result.rate())


if __name__ == '__main__':
  absltest.main()
