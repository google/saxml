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
"""Serving model parameters for Gemma."""

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
from saxml.server import servable_model_registry
from saxml.server.pax import quantization
from saxml.server.pax.lm import transformer_models
from saxml.server.pax.lm.params import template


@servable_model_registry.register
@template.make_servable(template.ServingTemplate)
class GemmaBase(base_experiment.BaseExperiment):
  """Gemma Transformer LM configuration."""

  SPM_MODEL = 'gs://cloud-tpu-inference-public/sax-tokenizers/gemma/gemma-tokenizer.model'
  SOS_ID = 2
  EOS_ID = 1
  GENERATE_ONLY = True  # No need to compute loss.

  # Architecture-related.
  NUM_LAYERS = 18
  VOCAB_SIZE = 256128
  DIMS_PER_HEAD = 256
  NUM_HEADS = 8
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 8
  FPROP_DTYPE = jnp.float32
  MODEL_DTYPE = jnp.float32
  USE_MQA = False
  USE_MXU_ATTENTION = False  # whether to force MHA attention to use MXU.

  # Sub-class has to specify a mesh.
  ICI_MESH_SHAPE = [1, 1, 1]
  DCN_MESH_SHAPE = None
  DECODE_MESH_TRANSPOSE = None

  BATCH_SIZE = 1
  NUM_CACHE_SLOTS = 0
  NUM_SAMPLES = 1
  ENABLE_GENERATE_STREAM = True
  STREAM_INTERVAL_STEPS = 16
  FPROP_FOR_PREFIX = True
  INPUT_SEQ_LEN = 4096
  BUCKET_KEYS = [128, 1024, 4096]
  MAX_DECODE_STEPS = [128, 512, 1024]
  EXTRA_INPUTS = {
      'temperature': 0.5,
      'per_example_max_decode_steps': 128,
      'per_example_top_k': 200,
      'per_example_top_p': 0.95,
  }

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    return []

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='xformer_task')
    if self.NUM_CACHE_SLOTS > 0:
      task_p.model = pax_fiddle.Config(
          layers.LanguageModelContinuousBatching, name='xformer_lm'
      )
    else:
      task_p.model = pax_fiddle.Config(layers.LanguageModel, name='xformer_lm')
    model_p = task_p.model
    model_p.lm_tpl = transformer_models.gemma(
        vocab_size=self.VOCAB_SIZE,
        model_dims=self.MODEL_DIMS,
        hidden_dims=self.HIDDEN_DIMS,
        num_layers=self.NUM_LAYERS,
        num_heads=self.NUM_HEADS,
        dim_per_head=self.DIMS_PER_HEAD,
        use_mqa=self.USE_MQA,
        use_mxu_attention=self.USE_MXU_ATTENTION,
    )

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
class Gemma2BFP16(GemmaBase):
  """Gemma2B model."""

  NUM_LAYERS = 18
  VOCAB_SIZE = 256128
  DIMS_PER_HEAD = 256
  NUM_HEADS = 8
  MODEL_DIMS = 2048
  HIDDEN_DIMS = MODEL_DIMS * 8
  USE_MQA = True

  BATCH_SIZE = [1, 64, 192]
  MAX_LIVE_BATCHES = 16
  NUM_SAMPLES = 1
  INPUT_SEQ_LEN = 1024
  BUCKET_KEYS = None
  MAX_DECODE_STEPS = 1024
  ENABLE_GENERATE_STREAM = False

  FPROP_DTYPE = jnp.bfloat16
  MODEL_DTYPE = jnp.bfloat16

  @classmethod
  def serving_mesh_shape(cls):
    return [
        [1, 1, 1],
        [1, 1, 4],
        [1, 1, 8],
    ]


@servable_model_registry.register
class Gemma2BFP16Exp(Gemma2BFP16):
  BATCH_SIZE = 1
  NUM_CACHE_SLOTS = 256
  MAX_LIVE_BATCHES = 256 * 4  # BATCH_SIZE is always 1 in this case.


@servable_model_registry.register
class Gemma7BFP16(GemmaBase):
  """Gemma7B model."""

  USE_MXU_ATTENTION = True
  NUM_LAYERS = 28
  VOCAB_SIZE = 256128
  DIMS_PER_HEAD = 256
  NUM_HEADS = 16
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 8
  USE_MQA = False

  BATCH_SIZE = [1, 32]
  NUM_SAMPLES = 1
  INPUT_SEQ_LEN = 1024
  BUCKET_KEYS = None
  MAX_DECODE_STEPS = 1024
  ENABLE_GENERATE_STREAM = False

  FPROP_DTYPE = jnp.bfloat16
  MODEL_DTYPE = jnp.bfloat16

  @classmethod
  def serving_mesh_shape(cls):
    return [
        # A single device or just on CPU. Note: it's not fitted in v5e-1.
        [1, 1, 1],
        [1, 1, 4],
        [1, 1, 8],
    ]


@servable_model_registry.register
class Gemma7BFP16Exp(Gemma7BFP16):
  BATCH_SIZE = 1
  NUM_CACHE_SLOTS = 48
  MAX_LIVE_BATCHES = 48 * 4  # BATCH_SIZE is always 1 in this case.
  # Need to add --extra_flags=xla_tpu_enable_dot_strength_reduction=false


@servable_model_registry.register
class Gemma2BFP16With8Replicas(Gemma2BFP16):
  """Gemma2B model on v5e-8 with 8 replications."""

  @classmethod
  def serving_mesh_shape(cls):
    return [
        [8, 1, 1],
    ]


@servable_model_registry.register
class Gemma2BFP16With4Replicas(Gemma2BFP16):
  """Gemma2B model on v4-8 or v5e-4 both with 4 replications."""

  @classmethod
  def serving_mesh_shape(cls):
    return [
        [4, 1, 1],
    ]


@servable_model_registry.register
class Gemma7BFP16With2Replicas(Gemma7BFP16):
  """Gemma7B model on v5e-8 with 2 replications."""

  @classmethod
  def serving_mesh_shape(cls):
    return [
        [2, 1, 4],
    ]


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class Gemma2BInt8(Gemma2BFP16):
  """Gemma2B model with int8 weight quantization."""


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class Gemma7BInt8(Gemma7BFP16):
  """Gemma7B model with int8 weight quantization."""


@servable_model_registry.register
class Gemma2BInt8With8Replicas(Gemma2BInt8):
  """Gemma2B model with int8 quantization on v4-8 or v5e-8 with 8 replications."""

  @classmethod
  def serving_mesh_shape(cls):
    return [
        [8, 1, 1],
    ]


@servable_model_registry.register
class Gemma7BInt8With2Replicas(Gemma7BInt8):
  """Gemma7B model with int8 quantization on v4-8 or v5e-8 with 2 replications."""

  @classmethod
  def serving_mesh_shape(cls):
    return [
        [2, 1, 4],
    ]


@servable_model_registry.register
class Gemma2BFP16Test(Gemma2BFP16):
  """Gemma2B model for testing without ckpt."""

  test_mode = True


@servable_model_registry.register
class Gemma7BFP16Test(Gemma7BFP16):
  """Gemma7B model for testing without ckpt."""

  test_mode = True


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class Gemma2BInt8Test(Gemma2BInt8):
  """Gemma2B model with int8 weight quantization for testing without ckpt."""

  test_mode = True


@servable_model_registry.register
@quantization.for_transformer(quantize_on_the_fly=False)
class Gemma7BInt8Test(Gemma7BInt8):
  """Gemma7B model with int8 weight quantization for testing without ckpt."""

  test_mode = True
