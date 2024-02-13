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
"""Serving model parameters for lm_cloud."""

# OSS import placeholder
from typing import List

from jax import numpy as jnp
from paxml import base_experiment
from paxml import tasks_lib
from praxis import base_input
from praxis import layers
from praxis import optimizers
from praxis import pax_fiddle
from praxis import schedules
from praxis.layers import activations
from saxml.server import servable_model_registry
from saxml.server.pax.lm import layers as sax_layers
from saxml.server.pax.lm.params import template


@servable_model_registry.register
@template.make_servable()
class GPTJ(base_experiment.BaseExperiment):
  """GPTJ Transformer LM configuration."""

  VOCABULARY_CLASS = 'GPT2BPEVocabulary'
  VOCABULARY_PATH = 'gs://saxml-e2e-tests/mlperf-gptj-bpe-vocabulary'

  # Match the HF model tokenizer configs.
  SOS_ID = 50256
  EOS_ID = 50256
  EOS_PADDING_AND_NO_SOS = True

  # architecture related
  NUM_LAYERS = 28
  VOCAB_SIZE = 50401
  DIMS_PER_HEAD = 256
  NUM_HEADS = 16
  MODEL_DIMS = 4096
  HIDDEN_DIMS = MODEL_DIMS * 4
  ROTARY_DIM = 64
  MAX_POSITION_EMBEDDINGS = 2048
  NORM_POLICY = 'pre'
  FPROP_DTYPE = jnp.float32
  MODEL_DTYPE = jnp.float32

  ACTIVATION_CLS = activations.GELU
  LAYER_NORM_EPSILON = 1.0e-05

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 1, 1]
  TRAINING_OPTIMIZED_SHARDING = False
  DCN_MESH_SHAPE = None
  DECODE_MESH_TRANSPOSE = None
  LENGTH_NORM_ALPHA = 0.5

  BATCH_SIZE = 1
  NUM_SAMPLES = 1
  FPROP_FOR_PREFIX = True
  INPUT_SEQ_LEN = 1919
  BUCKET_KEYS = None
  MIN_DECODE_STEPS = 30
  MAX_DECODE_STEPS = 128
  EXTRA_INPUTS = {
      'temperature': 0.5,
      'per_example_max_decode_steps': 128,
      'per_example_top_k': 200,
      'per_example_top_p': 0.95,
  }
  USE_BEAM_SEARCH = True
  BEAM_SEARCH_EARLY_EXIT = False
  BEAM_SIZE = 4
  TOKENS_PER_BEAM = 4
  BATCH_WAIT_SECS = 0.2
  MAX_LIVE_BATCHES = 2

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    return []

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='xformer_task')
    task_p.model = pax_fiddle.Config(layers.LanguageModel, name='xformer_lm')
    model_p = task_p.model
    model_p.lm_tpl.packed_input = False
    model_p.lm_tpl.model_dims = self.MODEL_DIMS
    model_p.lm_tpl.vocab_size = self.VOCAB_SIZE
    model_p.lm_tpl.position_emb_tpl = None
    model_p.lm_tpl.softmax_tpl = pax_fiddle.Config(
        layers.FullSoftmax,
        name='output',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    model_p.lm_tpl.softmax_tpl.feed_forward_tpl.has_bias = True
    model_p.lm_tpl.separate_embedding_tpl = pax_fiddle.Config(
        layers.Embedding,
        name='tok_embeddings',
        input_dims=self.MODEL_DIMS,
        num_classes=self.VOCAB_SIZE,
    )
    ln_tpl = pax_fiddle.Config(
        layers.LayerNorm,
        name='norm',
        direct_scale=True,
        epsilon=1.0e-05,
    )
    model_p.lm_tpl.final_ln_tpl = ln_tpl.clone()

    stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = self.MODEL_DIMS
    stacked_transformer_tpl.hidden_dims = self.HIDDEN_DIMS
    stacked_transformer_tpl.num_layers = self.NUM_LAYERS
    stacked_transformer_tpl.num_heads = self.NUM_HEADS
    stacked_transformer_tpl.dim_per_head = self.DIMS_PER_HEAD

    transformer_layer_p = pax_fiddle.Config(
        sax_layers.ParallelTransformerOnlyNormAttentionInputs
    )
    transformer_layer_p.norm_policy = self.NORM_POLICY
    transformer_layer_p.ln_tpl = ln_tpl.clone()
    transformer_layer_p.ln_tpl.epsilon = self.LAYER_NORM_EPSILON
    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    transformer_layer_p.tr_atten_tpl.internal_enable_query_scale = True
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    rotary_position_emb_p = pax_fiddle.Config(sax_layers.GPTJRotaryEmbedding)
    rotary_position_emb_p.max_position_embeddings = self.MAX_POSITION_EMBEDDINGS
    rotary_position_emb_p.rotary_dim = self.ROTARY_DIM
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True
    transformer_layer_p.tr_atten_tpl.consolidate_rope_key_state = True
    transformer_layer_p.tr_atten_tpl.rotary_position_emb_tpl = (
        rotary_position_emb_p
    )

    transformer_layer_p.tr_fflayer_tpl.has_bias = True
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        self.ACTIVATION_CLS
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = False
    transformer_layer_p.tr_fflayer_tpl.add_skip_connection = False
    stacked_transformer_tpl.transformer_layer_params_tpl = transformer_layer_p

    model_p.lm_tpl.stacked_transformer_tpl = stacked_transformer_tpl

    model_p.fprop_dtype = self.FPROP_DTYPE
    model_p.dtype = self.MODEL_DTYPE

    # Set sharding
    task_p = template.set_decoding_sharding_hparams(
        task_p,
        mesh_shape=self.ICI_MESH_SHAPE,
        decode_mesh_transpose=self.DECODE_MESH_TRANSPOSE,
    )
    # Unused.
    lp = task_p.train.learner
    lp.loss_name = 'total_loss'
    lp.optimizer = pax_fiddle.Config(
        optimizers.ShardedSgd,
        learning_rate=1e-3,
        lr_schedule=pax_fiddle.Config(schedules.Constant),
    )
    return task_p


@servable_model_registry.register
class GPTJ4BF16BS32(GPTJ):
  """GPTJ Transformer LM tokenized configuration."""

  ICI_MESH_SHAPE = [1, 1, 4]
  BATCH_SIZE = 32
  FPROP_DTYPE = jnp.bfloat16
  MODEL_DTYPE = jnp.bfloat16
  USE_MATMUL_BEAM_SHUFFLE = True
  BUCKET_KEYS = [512, 1024, 1919]
  MAX_DECODE_STEPS = [32, 64, 128]
  BATCH_WAIT_SECS = 4.0
  MAX_LIVE_BATCHES = 6

  DECODE_MESH_TRANSPOSE = {
      'fprop_mdl': 'mdl',
      'mdl': 'fprop_mdl',
  }

  @classmethod
  def serving_mesh_shape(cls) -> list[int]:
    # replica, data_mdl2, mdl, fprop_data, fprop_mdl
    return [1, 1, 1, 1, 4]


@servable_model_registry.register
class GPTJ4TokenizedBF16BS32(GPTJ4BF16BS32):
  """GPTJ Transformer LM tokenized configuration."""

  TOKENIZED_INPUT = True
  TOKENIZED_OUTPUT = True
