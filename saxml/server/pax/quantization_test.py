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

"""Tests for quantization methods."""

from absl.testing import absltest
from absl.testing import parameterized

from praxis import layers
from praxis import test_utils
from praxis.layers import quantization as qlayer
from praxis.layers.quantization import quantization_hparams

from saxml.server.pax import quantization
from saxml.server.pax.lm.params.lm_cloud import LmCloudSpmd2B


class QuantizationTest(test_utils.TestCase):

  @parameterized.named_parameters(
      ('combine_qkv', True, qlayer.attentions.CombinedQKVProjectionLayer),
      ('separate_qkv', False, layers.attentions.CombinedQKVProjectionLayer),
  )
  def test_update_transformer(self, use_combine_qkv, qkv_cls):
    p = layers.transformers.Transformer.HParams(
        name='jax_transformer_layer', input_dims=12, hidden_dims=4, num_heads=8)
    p.tr_atten_tpl.combine_qkv = use_combine_qkv
    quantization.quantize_transformer_layer_weights(
        p, quantization_hparams.QuantizationType.PTQ,
        quantization_hparams.QuantizationMode.MATERIALIZE)
    self.assertEqual(p.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls,
                     qlayer.linears.Linear)
    self.assertEqual(p.tr_atten_tpl.proj_tpl.cls,
                     qlayer.attentions.AttentionProjection)
    self.assertEqual(p.tr_atten_tpl.combined_qkv_proj_tpl.cls, qkv_cls)


@quantization.for_transformer()
class QuantizationModel(LmCloudSpmd2B):
  """Quantize transformer for GLaM100MTarzanC30PF1x1x2Serving."""
  pass


class DecoratorTest(test_utils.TestCase):

  def test_for_transformer(self):
    config = QuantizationModel()
    task_p = config.task()
    self.assertEqual(
        task_p.model.lm_tpl.stacked_transformer_tpl.block
        .transformer_layer_params_tpl.tr_atten_tpl.proj_tpl.cls,
        qlayer.attentions.AttentionProjection)
    self.assertEqual(
        task_p.model.lm_tpl.stacked_transformer_tpl.block
        .transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.linear_tpl.cls,
        qlayer.linears.Linear)
    self.assertEqual(
        task_p.model.lm_tpl.stacked_transformer_tpl.block
        .transformer_layer_params_tpl.tr_atten_tpl.proj_tpl.quantization
        .quantization_type, quantization_hparams.QuantizationType.PTQ)

    self.assertEqual(
        task_p.model.lm_tpl.stacked_transformer_tpl.block
        .transformer_layer_params_tpl.tr_atten_tpl.proj_tpl.quantization.mode,
        quantization_hparams.QuantizationMode.MATERIALIZE)

    self.assertEqual(
        task_p.model.lm_tpl.stacked_transformer_tpl.block
        .transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.linear_tpl
        .quantization.quantization_type,
        quantization_hparams.QuantizationType.PTQ)
    self.assertEqual(
        task_p.model.lm_tpl.stacked_transformer_tpl.block
        .transformer_layer_params_tpl.tr_fflayer_tpl.fflayer_tpl.linear_tpl
        .quantization.mode, quantization_hparams.QuantizationMode.MATERIALIZE)


if __name__ == '__main__':
  absltest.main()
