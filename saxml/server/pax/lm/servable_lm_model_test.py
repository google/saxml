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
"""Tests for servable_lm_model."""

import os

from absl import flags
from absl import logging
from absl.testing import absltest
import jax
import numpy as np
from paxml import checkpoints
from paxml.tasks.lm.params import lm_cloud
from praxis import pax_fiddle
from praxis import py_utils
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_model


class LmCloudSpmdSmall(
    lm_cloud.LmCloudSpmd, servable_lm_model.ServableLMModelParams
):
  """SPMD model with small params."""

  NUM_LAYERS = 3
  MODEL_DIMS = 128
  DIMS_PER_HEAD = 32
  HIDDEN_DIMS = MODEL_DIMS * 4

  ICI_MESH_SHAPE = [1, 1, 1]

  @classmethod
  def serving_mesh_shape(cls):
    return cls.ICI_MESH_SHAPE

  def serving_tokenizer(self):
    spm_model = os.path.join(
        flags.FLAGS.test_srcdir,
        'google3/third_party/py/saxml/server/pax/lm/test_data',
        'test_model.model',
    )
    return pax_fiddle.Config(
        lm_tokenizer.LMTokenizer,
        spm_model=spm_model,
        target_sos_id=0,
        target_eos_id=1,
    )

  def score(self):
    return servable_lm_model.ScoreHParams(
        batch_size=4, max_input_seq_len=8, max_suffix_seq_len=8
    )

  def input_for_model_init(self):
    batch_size, seq_len = 4, 16
    targets = np.ones([batch_size, seq_len], dtype=np.int32)
    input_batch = py_utils.NestedMap()
    input_batch.ids = targets
    input_batch.paddings = np.zeros_like(targets)
    input_batch.weights = np.ones_like(targets)
    input_batch.labels = targets
    input_batch.segment_ids = targets
    input_batch.segment_pos = np.tile(
        np.arange(0, seq_len)[np.newaxis, :], [batch_size, 1]
    )
    return input_batch

  def task(self):
    task_p = super().task()
    task_p.model.lm_tpl.packed_input = False
    return task_p


class ServableLMModelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._prng_key = jax.random.PRNGKey(1234)

  def test_load_model_score(self):
    model = servable_lm_model.ServableLMModel(
        LmCloudSpmdSmall(),
        primary_process_id=0,
        ckpt_type=checkpoints.CheckpointType.GDA,
        test_mode=True,
    )
    model.load(checkpoint_path=None, prng_key=self._prng_key)
    score_result = model.method(servable_lm_model.LMMethodName.SCORE).compute(
        [('k', ['a b']), ('k', ['b d e'])]
    )
    logging.info('score_result: %s', score_result)
    self.assertLen(score_result, 2)
    score_result = model.method(servable_lm_model.LMMethodName.SCORE).compute(
        [('p', ['a b c']), ('p', ['d e f']), ('p', ['g h i'])]
    )
    logging.info('score_result: %s', score_result)
    self.assertLen(score_result, 3)


if __name__ == '__main__':
  absltest.main()
