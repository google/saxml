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
"""Tests for template."""

from praxis import test_utils
from saxml.server.pax.lm.params import lm_cloud
from saxml.server.pax.lm.params import template
import tensorflow as tf


@template.make_servable()
class TestModel(lm_cloud.LmCloudSpmd2BTest):
  ENABLE_GENERATE_STREAM = True
  MAX_SEQ_LEN = None


@template.make_servable()
class TestModelMaxSequenceLength(TestModel):
  MAX_SEQ_LEN = 128


@template.make_servable()
class TestLayerwiseModel(TestModel):

  def task(self):
    task_p = super().task()
    transformer_p = (
        task_p.model.lm_tpl.stacked_transformer_tpl.transformer_layer_params_tpl
    )
    transformer_p_list = []

    for _ in range(self.NUM_LAYERS):
      single_tr_p = transformer_p.clone()
      transformer_p_list.append(single_tr_p)

    task_p.model.lm_tpl.stacked_transformer_tpl.transformer_layer_params_tpl = (
        transformer_p_list
    )
    return task_p


class TemplateTest(tf.test.TestCase, test_utils.TestCase):

  def test_seqlen(self):
    config = TestModelMaxSequenceLength()
    self.assertEqual(
        config.generate().decoder.seqlen, TestModelMaxSequenceLength.MAX_SEQ_LEN
    )
    self.assertEqual(
        config.generate_stream().decoder.seqlen,
        TestModelMaxSequenceLength.MAX_SEQ_LEN,
    )

    config = TestModel()
    self.assertEqual(
        config.generate().decoder.seqlen,
        TestModel.INPUT_SEQ_LEN + TestModel.MAX_DECODE_STEPS,
    )
    self.assertEqual(
        config.generate_stream().decoder.seqlen,
        TestModel.INPUT_SEQ_LEN + TestModel.MAX_DECODE_STEPS,
    )

    config = TestLayerwiseModel()
    self.assertEqual(
        config.generate().decoder.seqlen,
        TestLayerwiseModel.INPUT_SEQ_LEN + TestLayerwiseModel.MAX_DECODE_STEPS,
    )
    self.assertEqual(
        config.generate_stream().decoder.seqlen,
        TestLayerwiseModel.INPUT_SEQ_LEN + TestLayerwiseModel.MAX_DECODE_STEPS,
    )


if __name__ == '__main__':
  tf.test.main()
