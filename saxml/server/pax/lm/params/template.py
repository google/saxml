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

"""Serving template params."""

import os
from typing import Optional

from absl import flags
import numpy as np
from paxml import base_task
from praxis import decoder_hparams
from praxis import py_utils
from praxis.layers import attentions
from praxis.layers import transformers
from saxml.server.pax.lm import lm_tokenizer
from saxml.server.pax.lm import servable_lm_model

# Unused internal library

# pytype: disable=attribute-error


class ServingTemplate(servable_lm_model.ServableLMModelParams):
  # pylint: disable=line-too-long
  """Template servable config."""
  # pylint: enable=line-too-long

  ICI_MESH_SHAPE = [1, 1, 8]
  USE_BEAM_SEARCH = False
  BATCH_SIZE = 1
  INPUT_SEQ_LEN = 256
  MAX_DECODE_STEPS = 32
  NUM_SAMPLES = 2
  TOP_K = 40
  BEAM_SIZE = 4
  FPROP_FOR_PREFIX = False
  SPM = None
  VOCAB_SIZE = 32000
  LENGTH_NORM_ALPHA = 0.8
  SCORE_ONLY = False
  SPM_MODEL = None
  SOS_ID = 0
  EOS_ID = 1
  SLICE_LEFT = True
  EXTRA_INPUTS = {'temperature': 0.1}
  BUCKET_KEYS = None
  INCLUDE_PREFIX_IN_RESULT = False
  MAX_LIVE_BATCHES = 4

  def input_for_model_init(self):
    batch_size = self.BATCH_SIZE
    if isinstance(batch_size, (list, tuple)):
      batch_size = batch_size[0]
    seq_len = self.INPUT_SEQ_LEN
    targets = np.ones([batch_size, seq_len], dtype=np.int32)
    input_batch = py_utils.NestedMap()
    input_batch.ids = targets
    input_batch.paddings = np.zeros_like(targets)
    input_batch.inputs_indicator = np.ones_like(targets)
    input_batch.weights = np.ones_like(targets)
    input_batch.labels = targets
    input_batch.segment_ids = targets
    input_batch.segment_pos = np.tile(
        np.arange(0, seq_len)[np.newaxis, :], [batch_size, 1])
    return input_batch

  @classmethod
  def serving_mesh_shape(cls):
    return cls.ICI_MESH_SHAPE

  def score(self):
    return servable_lm_model.ScoreHParams(
        batch_size=self.BATCH_SIZE, max_input_seq_len=self.INPUT_SEQ_LEN)

  def serving_tokenizer(self):
    if self.SPM_MODEL is None:
      spm_model = self._dataset_train().input.tokenizer.spm_model
    else:
      spm_model = self.SPM_MODEL

    return lm_tokenizer.LMTokenizer.HParams(
        spm_model=spm_model,
        target_sos_id=self.SOS_ID,
        target_eos_id=self.EOS_ID,
        slice_left=self.SLICE_LEFT)

  def generate(self) -> Optional[servable_lm_model.DecodeHParams]:
    if self.SCORE_ONLY:
      return None

    if self.USE_BEAM_SEARCH:
      generate_hparams = decoder_hparams.BeamSearchHParams(
          fprop_for_prefix=True,
          max_decode_steps=self.MAX_DECODE_STEPS,
          seqlen=self.INPUT_SEQ_LEN + self.MAX_DECODE_STEPS,
          beam_size=self.BEAM_SIZE,
          eos_id=self.EOS_ID,
          length_norm_alpha=self.LENGTH_NORM_ALPHA)
    else:
      generate_hparams = decoder_hparams.SampleDecoderHParams(
          fprop_for_prefix=self.FPROP_FOR_PREFIX,
          # Use LPB for whenever FPROP_FOR_PREFIX is enabled.
          lazy_prefix_broadcast=(self.FPROP_FOR_PREFIX and
                                 self.NUM_SAMPLES > 1),
          max_decode_steps=self.MAX_DECODE_STEPS,
          seqlen=self.INPUT_SEQ_LEN + self.MAX_DECODE_STEPS,
          num_samples=self.NUM_SAMPLES,
          temperature=None,
          eos_id=self.EOS_ID,
          k=self.TOP_K)
    return servable_lm_model.DecodeHParams(
        batch_size=self.BATCH_SIZE,
        max_input_seq_len=self.INPUT_SEQ_LEN,
        bucket_keys=self.BUCKET_KEYS,
        decoder=generate_hparams,
        include_prefix_in_result=self.INCLUDE_PREFIX_IN_RESULT,
        max_live_batches=self.MAX_LIVE_BATCHES,
        extra_inputs=self.EXTRA_INPUTS)

  def set_serving_params(self, task_p: base_task.BaseTask.HParams) -> None:
    # Override attention with lazy prefix broadcast.
    lazy_prefix_broadcast = False
    decode_params = self.generate()
    if decode_params is not None:
      if decode_params.decoder.lazy_prefix_broadcast:
        assert decode_params.decoder.num_samples > 1  # pytype: disable=attribute-error
        lazy_prefix_broadcast = True

    if lazy_prefix_broadcast:
      xformer = task_p.model.lm_tpl.stacked_transformer_tpl  # pytype: disable=attribute-error  # enable-nested-classes
      if xformer.cls == transformers.StackedTransformerRepeated:
        xformer = xformer.block
      assert xformer.cls == transformers.StackedTransformer
      layer_p = xformer.transformer_layer_params_tpl
      lbp_tr_atten_tpl = attentions.DotProductAttentionWithLPB.HParams()
      if layer_p.tr_atten_tpl.cls == attentions.DotProductAttention:
        lbp_tr_atten_tpl.copy_fields_from(layer_p.tr_atten_tpl)
        layer_p.tr_atten_tpl = lbp_tr_atten_tpl
      else:
        assert (layer_p.tr_atten_tpl.cls == lbp_tr_atten_tpl.cls), (
            f'Attention layer does not support lazy prefix broadcast '
            f'{layer_p.tr_atten_tpl.cls}.')
