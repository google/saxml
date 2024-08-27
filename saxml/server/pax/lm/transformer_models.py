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

from typing import Optional, Sequence, Union

from praxis import layers
from praxis import pax_fiddle
from praxis.layers import multi_query_attention
from saxml.server.pax.lm import layers as sax_layers


def gemma(
    vocab_size: int,
    model_dims: int,
    hidden_dims: int,
    num_layers: int,
    num_heads: int,
    dim_per_head: int,
    use_mqa: bool,
    num_kv_heads: int = 1,
    attn_logit_softcap: float | None = None,
    final_logit_softcap: float | None = None,
    use_post_attn_norm: bool = False,
    use_post_ffn_norm: bool = False,
    scale_query_by_dim_per_head: bool = True,
    sliding_window_sizes: Optional[Union[int, Sequence[Optional[int]]]] = None,
    chunked_one_step_attn_num_seq_split=1,
    chunked_ffn_num_seq_split=1,
) -> pax_fiddle.Config[layers.TransformerLm]:
  """Create a TransformerLm config(template) for Gmini model family.

  Args:
    vocab_size: Size of vocabulary.
    model_dims: Model dimension.
    hidden_dims: Hidden dimension for the ffw layer.
    num_layers: Number of layers.
    num_heads: Number of heads for query.
    dim_per_head: Dimension per head.
    use_mqa: Whether use Multi-Query Attention.
    num_kv_heads: Number of heads for key and value.
    attn_logit_softcap: Softcap for attention logit.
    final_logit_softcap: Softcap for final logit.
    use_post_attn_norm: Whether to use post attention norm.
    use_post_ffn_norm: Whether to use post ffn norm.
    scale_query_by_dim_per_head: Whether to scale query by dim_per_head.
      Otherwise, it is scaled by hidden_dim // num_heads.
    sliding_window_sizes: Sliding window sizes for local attention.
    chunked_one_step_attn_num_seq_split: split attention computation in chunks.
    chunked_ffn_num_seq_split: chunk ff weight computation.

  Returns:
    TransformerLm for Gmini.
  """
  if num_kv_heads > 1:
    assert use_mqa, 'num_kv_heads > 1 is only supported with MQA.'

  model_p = pax_fiddle.Config(layers.TransformerLm)
  model_p.vocab_size = vocab_size
  model_p.model_dims = model_dims
  model_p.softmax_tpl = pax_fiddle.Config(
      layers.embedding_softmax.NClassMajorSharedEmbeddingSoftmax,
      scale_sqrt_depth=True,
      use_bias=False,
      soft_cap_logits=final_logit_softcap,
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
  if sliding_window_sizes is not None:
    stacked_transformer_tpl.local_window_size = sliding_window_sizes
  transformer_layer_p = pax_fiddle.Config(layers.Transformer)
  transformer_layer_p.ln_tpl = ln_tpl.clone()
  transformer_layer_p.norm_policy = (
      'primer_hybrid' if use_post_attn_norm else 'pre'
  )
  # Attention Layer.
  if sliding_window_sizes is not None:
    transformer_layer_p.tr_atten_tpl = pax_fiddle.Config(
        multi_query_attention.MultiQueryDotProductAttention,
        num_kv_heads=num_kv_heads,
        chunked_attn_num_seq_split=chunked_one_step_attn_num_seq_split,
    )
  elif use_mqa:
    transformer_layer_p.tr_atten_tpl = pax_fiddle.Config(
        sax_layers.ChunkedMQA,
        num_kv_heads=num_kv_heads,
        chunked_one_step_attn_num_seq_split=chunked_one_step_attn_num_seq_split,
    )
  else:
    transformer_layer_p.tr_atten_tpl = pax_fiddle.Config(
        sax_layers.MXUDotProductAttention,
        combine_qkv=True,
        internal_enable_per_dim_scale=False,
    )
  transformer_layer_p.tr_atten_tpl.use_bias = False
  transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True
  transformer_layer_p.tr_atten_tpl.consolidate_rope_key_state = True
  transformer_layer_p.tr_atten_tpl.scale_query_by_dim_per_head = (
      scale_query_by_dim_per_head
  )
  transformer_layer_p.tr_atten_tpl.atten_logit_cap = attn_logit_softcap
  # FeedForward
  transformer_layer_p.tr_fflayer_tpl = pax_fiddle.Config(
      sax_layers.TransformerFeedForwardWithSeqSplit,
      chunked_ffn_num_seq_split=chunked_ffn_num_seq_split
  )
  transformer_layer_p.tr_fflayer_tpl.ln_tpl = ln_tpl.clone()
  transformer_layer_p.tr_fflayer_tpl.has_bias = False
  transformer_layer_p.tr_fflayer_tpl.use_gated_activation = True
  transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
      layers.activations.GELU,
  )
  transformer_layer_p.tr_fflayer_tpl.norm_policy = (
      'primer_hybrid' if use_post_ffn_norm else 'pre'
  )

  stacked_transformer_tpl.transformer_layer_params_tpl = transformer_layer_p
  model_p.stacked_transformer_tpl = stacked_transformer_tpl

  return model_p
