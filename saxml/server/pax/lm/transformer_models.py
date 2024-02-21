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
"""Customize transformer models for sax."""

from praxis import layers
from praxis import pax_fiddle
from praxis.layers import multi_query_attention
from saxml.server.pax.lm import layers as sax_layers


def gemma(
    vocab_size,
    model_dims,
    hidden_dims,
    num_layers,
    num_heads,
    dim_per_head,
    use_mqa,
) -> pax_fiddle.Config[layers.TransformerLm]:
  """Create a TransformerLm config(template) for Gemma model family.

  Args:
    vocab_size: Size of vocabulary.
    model_dims: Model dimension.
    hidden_dims: Hidden dimension for the ffw layer.
    num_layers: Number of layers.
    num_heads: Number of heads.
    dim_per_head: Dimension per head.
    use_mqa: Whether use Multi-Query Attention.

  Returns:
    TransformerLm for Gemma.
  """
  model_p = pax_fiddle.Config(layers.TransformerLm)
  model_p.vocab_size = vocab_size
  model_p.model_dims = model_dims
  model_p.softmax_tpl = pax_fiddle.Config(
      layers.embedding_softmax.NClassMajorSharedEmbeddingSoftmax,
      scale_sqrt_depth=True,
      use_bias=False,
  )
  model_p.position_emb_tpl = None
  ln_tpl = pax_fiddle.Config(
      layers.RmsNorm,
      name='rms_norm',
      direct_scale=False,
  )
  model_p.final_ln_tpl = ln_tpl.clone()

  stacked_transformer_tpl = pax_fiddle.Config(layers.StackedTransformer)
  stacked_transformer_tpl.model_dims = model_dims
  stacked_transformer_tpl.hidden_dims = hidden_dims
  stacked_transformer_tpl.num_layers = num_layers
  stacked_transformer_tpl.num_heads = num_heads
  stacked_transformer_tpl.dim_per_head = dim_per_head
  transformer_layer_p = pax_fiddle.Config(layers.Transformer)
  transformer_layer_p.ln_tpl = ln_tpl.clone()
  # Attention Layer.
  if use_mqa:
    transformer_layer_p.tr_atten_tpl = pax_fiddle.Config(
        multi_query_attention.MultiQueryDotProductAttention,
        num_kv_heads=1,
        use_bias=False,
        use_rotary_position_emb=True,
        consolidate_rope_key_state=True,
        scale_query_by_dim_per_head=True,
    )
  else:
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True
    transformer_layer_p.tr_atten_tpl.consolidate_rope_key_state = True
    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    transformer_layer_p.tr_atten_tpl.scale_query_by_dim_per_head = True
  # FeedForward
  transformer_layer_p.tr_fflayer_tpl = pax_fiddle.Config(
      sax_layers.TransformerFeedForwardWithSeqSplit
  )
  transformer_layer_p.tr_fflayer_tpl.ln_tpl = ln_tpl.clone()
  transformer_layer_p.tr_fflayer_tpl.has_bias = False
  transformer_layer_p.tr_fflayer_tpl.use_gated_activation = True
  transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
      layers.activations.GELU,
  )

  stacked_transformer_tpl.transformer_layer_params_tpl = transformer_layer_p
  model_p.stacked_transformer_tpl = stacked_transformer_tpl

  return model_p
