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
"""Compatibility test between pax and JAX_gptj."""
from functools import partial  # pylint: disable=g-importing-member

from absl.testing import absltest
from flax.core.frozen_dict import freeze
import flax.linen as nn
from flax.linen import combine_masks
from flax.linen import make_causal_mask
from flax.linen.attention import dot_product_attention_weights
import jax
from jax import lax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import layers as praxis_layers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import activations
from praxis.layers import attentions
from saxml.server.pax.lm import layers as sax_layers
from saxml.server.pax.lm.experimental import layers as gptj_layers


# Copied from HF GitHub https://github.com/huggingface/transformers/blob/v4.30.2/src/transformers/models/gptj/modeling_flax_gptj.py#L605 # pylint: disable=line-too-long
def create_sinusoidal_positions(num_pos, dim):
  inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
  sinusoid_inp = np.einsum('i , j -> i j', np.arange(num_pos), inv_freq).astype(
      'float32'
  )
  sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

  sentinel = dim // 2 + dim % 2
  out = np.zeros((num_pos, dim))
  out[:, 0:sentinel] = sin
  out[:, sentinel:] = cos

  return jnp.array(out)


def rotate_every_two(tensor):
  rotate_half_tensor = jnp.stack(
      (-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1
  )
  rotate_half_tensor = rotate_half_tensor.reshape(
      rotate_half_tensor.shape[:-2] + (-1,)
  )
  return rotate_half_tensor


def apply_rotary_pos_emb(tensor, sincos):
  sin_pos, cos_pos = sincos
  sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
  cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
  return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)


class FlaxGPTJAttention(nn.Module):
  hidden_size: int
  num_attention_heads: int
  rotary_dim: int
  initializer_range: float
  resid_pdrop: float
  max_position_embeddings: int
  attn_pdrop: float
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.embed_dim = self.hidden_size
    self.num_heads = self.num_attention_heads
    self.head_dim = self.embed_dim // self.num_heads

    dense = partial(
        nn.Dense,
        self.embed_dim,
        use_bias=False,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(self.initializer_range),
    )

    self.q_proj, self.k_proj, self.v_proj = dense(), dense(), dense()
    self.out_proj = dense()

    self.resid_dropout = nn.Dropout(rate=self.resid_pdrop)

    self.causal_mask = make_causal_mask(
        jnp.ones((1, self.max_position_embeddings), dtype='bool'), dtype='bool'
    )

    pos_embd_dim = self.rotary_dim or self.embed_dim
    self.embed_positions = create_sinusoidal_positions(
        self.max_position_embeddings, pos_embd_dim
    )

  def _split_heads(self, hidden_states):
    return hidden_states.reshape(
        hidden_states.shape[:2] + (self.num_heads, self.head_dim)
    )

  def _merge_heads(self, hidden_states):
    return hidden_states.reshape(hidden_states.shape[:2] + (self.embed_dim,))

  @nn.compact
  def _concatenate_to_cache(self, key, value, query, attention_mask):
    # detect if we're initializing by absence of existing cache data.
    is_initialized = self.has_variable('cache', 'cached_key')
    cached_key = self.variable(
        'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
    )
    cached_value = self.variable(
        'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
    )
    cache_index = self.variable(
        'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
    )

    if is_initialized:
      *batch_dims, max_length, _, _ = (
          cached_key.value.shape
      )
      # update key, value caches with our new 1d spatial slices
      cur_index = cache_index.value
      indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
      key = lax.dynamic_update_slice(cached_key.value, key, indices)
      value = lax.dynamic_update_slice(cached_value.value, value, indices)
      cached_key.value = key
      cached_value.value = value
      num_updated_cache_vectors = query.shape[1]
      cache_index.value = cache_index.value + num_updated_cache_vectors
      # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
      pad_mask = jnp.broadcast_to(
          jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
          tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
      )
      attention_mask = combine_masks(pad_mask, attention_mask)
    return key, value, attention_mask

  def __call__(
      self,
      hidden_states,
      attention_mask,
      position_ids,
      deterministic: bool = True,
      init_cache: bool = False,
      output_attentions: bool = False,
  ):
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query)
    key = self._split_heads(key)
    value = self._split_heads(value)

    sincos = jnp.take(self.embed_positions, position_ids, axis=0)
    sincos = jnp.split(sincos, 2, axis=-1)

    if self.rotary_dim is not None:
      k_rot = key[:, :, :, : self.rotary_dim]
      k_pass = key[:, :, :, self.rotary_dim :]

      q_rot = query[:, :, :, : self.rotary_dim]
      q_pass = query[:, :, :, self.rotary_dim :]

      k_rot = apply_rotary_pos_emb(k_rot, sincos)
      q_rot = apply_rotary_pos_emb(q_rot, sincos)

      key = jnp.concatenate([k_rot, k_pass], axis=-1)
      query = jnp.concatenate([q_rot, q_pass], axis=-1)
    else:
      key = apply_rotary_pos_emb(key, sincos)
      query = apply_rotary_pos_emb(query, sincos)

    query_length, key_length = query.shape[1], key.shape[1]

    if self.has_variable('cache', 'cached_key'):
      mask_shift = self.variables['cache']['cache_index']
      max_decoder_length = self.variables['cache']['cached_key'].shape[1]
      causal_mask = lax.dynamic_slice(
          self.causal_mask,
          (0, 0, mask_shift, 0),
          (1, 1, query_length, max_decoder_length),
      )
    else:
      causal_mask = self.causal_mask[:, :, :query_length, :key_length]

    batch_size = hidden_states.shape[0]
    causal_mask = jnp.broadcast_to(
        causal_mask, (batch_size,) + causal_mask.shape[1:]
    )

    attention_mask = jnp.broadcast_to(
        jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_mask.shape
    )
    attention_mask = combine_masks(attention_mask, causal_mask)

    dropout_rng = None
    if not deterministic and self.attn_pdrop > 0.0:
      dropout_rng = self.make_rng('dropout')

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.has_variable('cache', 'cached_key') or init_cache:
      key, value, attention_mask = self._concatenate_to_cache(
          key, value, query, attention_mask
      )

    # transform boolean mask into float mask
    attention_bias = lax.select(
        attention_mask > 0,
        jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
        jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(
            self.dtype
        ),
    )

    # usual dot product attention
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias=attention_bias,
        dropout_rng=dropout_rng,
        dropout_rate=self.attn_pdrop,
        deterministic=deterministic,
        dtype=self.dtype,
        precision=None,
    )

    attn_output = jnp.einsum('...hqk,...khd->...qhd', attn_weights, value)
    attn_output = self._merge_heads(attn_output)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

    outputs = (
        (attn_output, attn_weights) if output_attentions else (attn_output,)
    )
    return outputs


class FlaxGPTJMLP(nn.Module):
  intermediate_size: int
  hidden_size: int
  initializer_range: float
  resid_pdrop: float
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    embed_dim = self.hidden_size
    kernel_init = jax.nn.initializers.normal(self.initializer_range)

    self.fc_in = nn.Dense(
        self.intermediate_size, dtype=self.dtype, kernel_init=kernel_init
    )
    self.fc_out = nn.Dense(embed_dim, dtype=self.dtype, kernel_init=kernel_init)

    self.act = partial(nn.gelu, approximate=True)
    self.dropout = nn.Dropout(rate=self.resid_pdrop)

  def __call__(self, hidden_states, deterministic: bool = True):
    hidden_states = self.fc_in(hidden_states)
    hidden_states = self.act(hidden_states)
    hidden_states = self.fc_out(hidden_states)
    hidden_states = self.dropout(hidden_states, deterministic=deterministic)
    return hidden_states


class FlaxGPTJBlock(nn.Module):
  hidden_size: int
  n_inner: int
  layer_norm_epsilon: float
  initializer_range: float
  resid_pdrop: float
  num_attention_heads: int
  rotary_dim: int
  max_position_embeddings: int
  attn_pdrop: float
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    hidden_size = self.hidden_size
    inner_dim = self.n_inner if self.n_inner is not None else 4 * hidden_size

    self.ln_1 = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype)
    self.attn = FlaxGPTJAttention(
        dtype=self.dtype,
        hidden_size=hidden_size,
        num_attention_heads=self.num_attention_heads,
        rotary_dim=self.rotary_dim,
        initializer_range=self.initializer_range,
        resid_pdrop=self.resid_pdrop,
        max_position_embeddings=self.max_position_embeddings,
        attn_pdrop=self.attn_pdrop,
    )

    self.mlp = FlaxGPTJMLP(
        intermediate_size=inner_dim,
        dtype=self.dtype,
        hidden_size=hidden_size,
        initializer_range=self.initializer_range,
        resid_pdrop=self.resid_pdrop,
    )

  def __call__(
      self,
      hidden_states,
      attention_mask=None,
      position_ids=None,
      deterministic: bool = True,
      init_cache: bool = False,
      output_attentions: bool = False,
  ):
    residual = hidden_states
    hidden_states = self.ln_1(hidden_states)
    attn_outputs = self.attn(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        deterministic=deterministic,
        init_cache=init_cache,
        output_attentions=output_attentions,
    )
    attn_output = attn_outputs[0]

    feed_forward_hidden_states = self.mlp(
        hidden_states, deterministic=deterministic
    )

    # residual connection
    hidden_states = attn_output + feed_forward_hidden_states + residual

    return (hidden_states,) + attn_outputs[1:]


class FlaxGPTJBlockCollection(nn.Module):
  num_hidden_layers: int
  hidden_size: int
  n_inner: int
  layer_norm_epsilon: float
  initializer_range: float
  resid_pdrop: float
  num_attention_heads: int
  rotary_dim: int
  max_position_embeddings: int
  attn_pdrop: float
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    kwargs = {
        'hidden_size': self.hidden_size,
        'n_inner': self.n_inner,
        'layer_norm_epsilon': self.layer_norm_epsilon,
        'num_attention_heads': self.num_attention_heads,
        'rotary_dim': self.rotary_dim,
        'initializer_range': self.initializer_range,
        'resid_pdrop': self.resid_pdrop,
        'max_position_embeddings': self.max_position_embeddings,
        'attn_pdrop': self.attn_pdrop,
        'dtype': self.dtype,
    }
    self.blocks = [
        FlaxGPTJBlock(**kwargs, name=str(i))
        for i in range(self.num_hidden_layers)
    ]

  def __call__(
      self,
      hidden_states,
      attention_mask=None,
      position_ids=None,
      deterministic: bool = True,
      init_cache: bool = False,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
    all_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None

    for block in self.blocks:
      if output_hidden_states:
        all_hidden_states += (hidden_states,)

      layer_outputs = block(
          hidden_states,
          attention_mask,
          position_ids=position_ids,
          deterministic=deterministic,
          init_cache=init_cache,
          output_attentions=output_attentions,
      )
      hidden_states = layer_outputs[0]

      if output_attentions:
        all_attentions += (layer_outputs[1],)

    # this contains possible `None` values - `FlaxGPTJModule` will filter them out
    outputs = (hidden_states, all_hidden_states, all_attentions)

    return outputs


class FlaxGPTJModule(nn.Module):
  vocab_size: int
  embd_pdrop: float
  num_hidden_layers: int
  hidden_size: int
  n_inner: int
  layer_norm_epsilon: float
  initializer_range: float
  resid_pdrop: float
  num_attention_heads: int
  rotary_dim: int
  max_position_embeddings: int
  attn_pdrop: float
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.embed_dim = self.hidden_size

    self.wte = nn.Embed(
        self.vocab_size,
        self.hidden_size,
        embedding_init=jax.nn.initializers.normal(
            stddev=self.initializer_range
        ),
    )
    self.dropout = nn.Dropout(rate=self.embd_pdrop)
    self.h = FlaxGPTJBlockCollection(
        num_hidden_layers=self.num_hidden_layers,
        hidden_size=self.hidden_size,
        n_inner=self.n_inner,
        layer_norm_epsilon=self.layer_norm_epsilon,
        num_attention_heads=self.num_attention_heads,
        rotary_dim=self.rotary_dim,
        initializer_range=self.initializer_range,
        resid_pdrop=self.resid_pdrop,
        max_position_embeddings=self.max_position_embeddings,
        attn_pdrop=self.attn_pdrop,
        dtype=self.dtype,
    )
    self.ln_f = nn.LayerNorm(epsilon=self.layer_norm_epsilon, dtype=self.dtype)

  def __call__(
      self,
      input_ids,
      attention_mask,
      position_ids,
      deterministic=True,
      init_cache: bool = False,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
    input_embeds = self.wte(input_ids.astype('i4'))

    hidden_states = self.dropout(input_embeds, deterministic=deterministic)

    outputs = self.h(
        hidden_states,
        attention_mask,
        position_ids=position_ids,
        deterministic=deterministic,
        init_cache=init_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    hidden_states = self.ln_f(hidden_states)

    if output_hidden_states:
      all_hidden_states = outputs[1] + (hidden_states,)
      outputs = (hidden_states, all_hidden_states) + outputs[2:]
    else:
      outputs = (hidden_states,) + outputs[1:]

    if not return_dict:
      return tuple(v for v in outputs if v is not None)

    return hidden_states


class FlaxGPTJForCausalLMModule(nn.Module):
  tie_word_embeddings: bool
  vocab_size: int
  embd_pdrop: float
  num_hidden_layers: int
  hidden_size: int
  n_inner: int
  layer_norm_epsilon: float
  initializer_range: float
  resid_pdrop: float
  num_attention_heads: int
  rotary_dim: int
  max_position_embeddings: int
  attn_pdrop: float
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.transformer = FlaxGPTJModule(
        dtype=self.dtype,
        vocab_size=self.vocab_size,
        embd_pdrop=self.embd_pdrop,
        num_hidden_layers=self.num_hidden_layers,
        hidden_size=self.hidden_size,
        n_inner=self.n_inner,
        layer_norm_epsilon=self.layer_norm_epsilon,
        num_attention_heads=self.num_attention_heads,
        rotary_dim=self.rotary_dim,
        initializer_range=self.initializer_range,
        resid_pdrop=self.resid_pdrop,
        max_position_embeddings=self.max_position_embeddings,
        attn_pdrop=self.attn_pdrop,
    )
    self.lm_head = nn.Dense(
        self.vocab_size,
        dtype=self.dtype,
        kernel_init=jax.nn.initializers.normal(stddev=self.initializer_range),
    )

  def __call__(
      self,
      input_ids,
      attention_mask,
      position_ids,
      deterministic: bool = True,
      init_cache: bool = False,
      output_attentions: bool = False,
      output_hidden_states: bool = False,
      return_dict: bool = True,
  ):
    outputs = self.transformer(
        input_ids,
        attention_mask,
        position_ids,
        deterministic=deterministic,
        init_cache=init_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]

    if self.tie_word_embeddings:
      shared_kernel = self.transformer.variables['params']['wte']['embedding'].T
      lm_logits = self.lm_head.apply(
          {'params': {'kernel': shared_kernel}}, hidden_states
      )
    else:
      lm_logits = self.lm_head(hidden_states)

    if not return_dict:
      return (lm_logits,) + outputs[1:]

    return lm_logits


class GPTJTest(test_utils.TestCase):
  # Test against FlaxAutoModelForCausalLM from EleutherAI/gpt-j-6b

  def testAttenLayer(self):
    dim = 128
    rotary_dim = 16
    max_seq_len = 8
    max_batch_size = 2
    np.random.seed(123456)
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    x = np.random.randn(max_batch_size, max_seq_len, dim).astype(np.float32)
    wq = np.random.randn(dim, dim).astype(np.float32)
    wk = np.random.randn(dim, dim).astype(np.float32)
    wv = np.random.randn(dim, dim).astype(np.float32)
    wo = np.random.randn(dim, dim).astype(np.float32)

    atten_mask = jnp.ones((max_batch_size, max_seq_len), dtype=np.int32)
    segment_pos = jnp.broadcast_to(
        jnp.arange(max_seq_len, dtype=np.int32)[None, :],
        (max_batch_size, max_seq_len),
    )

    jax_attention = FlaxGPTJAttention(
        rotary_dim=rotary_dim, hidden_size=dim, num_attention_heads=4,
        max_position_embeddings=max_seq_len, initializer_range=0.2,
        resid_pdrop=0.0, attn_pdrop=0.0
    )

    jax_params = freeze({
        'q_proj': {'kernel': jnp.asarray(wq)},
        'k_proj': {'kernel': jnp.asarray(wk)},
        'v_proj': {'kernel': jnp.asarray(wv)},
        'out_proj': {'kernel': jnp.asarray(wo)},
    })

    jax_output = jax_attention.apply(
        {'params': jax_params},
        jnp.asarray(x),
        atten_mask,
        segment_pos,
    )
    pax_atten_p = pax_fiddle.Config(
        praxis_layers.DotProductAttention,
        input_dim=dim,
        hidden_dim=dim,
        num_heads=4,
        dim_per_head=dim // 4,
        internal_enable_per_dim_scale=False,
        internal_enable_query_scale=True,
        use_bias=False,
        combine_qkv=True,
        use_rotary_position_emb=True,
        consolidate_rope_key_state=True,
    )
    rotary_position_emb_p = (
        pax_fiddle.Config(gptj_layers.GPTJRotaryEmbedding)
    )
    rotary_position_emb_p.max_position_embeddings = max_seq_len
    rotary_position_emb_p.rotary_dim = rotary_dim
    pax_atten_p.rotary_position_emb_tpl = rotary_position_emb_p
    pax_attention = pax_atten_p.Instantiate()

    pax_atten_mask = attentions.causal_mask(
        jnp.zeros([max_batch_size, max_seq_len, dim], dtype=jnp.float32))
    initial_vars = pax_attention.init(init_key, x, x, x, pax_atten_mask)
    wc = np.stack([wq, wk, wv], axis=0)
    wc = np.reshape(wc, [3, dim, 4, dim // 4])

    initial_vars['params']['combined_qkv']['w'] = wc
    initial_vars['params']['post']['w'] = np.reshape(
        wo.transpose(), [dim, 4, dim // 4]
    )
    pax_output, _ = pax_attention.apply(
        initial_vars,
        x,
        x,
        x,
        atten_mask=pax_atten_mask,
        query_segment_pos=segment_pos,
        key_segment_pos=segment_pos,
    )
    flex_causal_mask = make_causal_mask(
        jnp.ones((1, max_seq_len), dtype='bool'), dtype='bool'
    )
    self.assertAllClose(
        pax_atten_mask > -0.7 * jnp.finfo(jnp.float32).max, flex_causal_mask
    )
    self.assertAllClose(pax_output, jax_output[0], atol=0.0001, rtol=0.001)

  # Uses the actual HF Flax model with random weights
  def testTransformerLmAgainstHFFlaxModel(self):
    vocab_size = 50401
    max_batch_size = 1
    max_seq_len = 1919

    model_dims = 4096
    hidden_dims = 16384
    num_heads = 16

    rotary_dim = 64
    embd_pdrop = 0.0
    initializer_range = 0.2
    resid_pdrop = 0.0
    attn_pdrop = 0.0
    dtype = jnp.float32
    layer_norm_epsilon = 1.0e-05

    fprop_dtype = jnp.float32
    model_dtype = jnp.float32

    np.random.seed(123456)
    prng_key = jax.random.PRNGKey(seed=123456)
    prng_key, init_key = jax.random.split(prng_key)
    x = np.random.randint(
        low=0,
        high=vocab_size,
        size=(max_batch_size, max_seq_len),
        dtype=np.int32,
    )
    paddings = jnp.zeros_like(x, dtype=jnp.int32)
    segment_pos = jnp.broadcast_to(
        jnp.arange(max_seq_len, dtype=np.int32)[None, :],
        (max_batch_size, max_seq_len),
    )
    wq = np.random.randn(model_dims, model_dims).astype(np.float32)
    wk = np.random.randn(model_dims, model_dims).astype(np.float32)
    wv = np.random.randn(model_dims, model_dims).astype(np.float32)
    wo = np.random.randn(model_dims, model_dims).astype(np.float32)
    w1 = np.random.randn(model_dims, hidden_dims).astype(np.float32)
    b1 = np.random.randn(hidden_dims).astype(np.float32)
    w2 = np.random.randn(hidden_dims, model_dims).astype(np.float32)
    b2 = np.random.randn(model_dims).astype(np.float32)
    layernorm_w = np.random.randn(model_dims).astype(np.float32)
    layernorm_b = np.random.randn(model_dims).astype(np.float32)
    final_norm_w = np.random.randn(model_dims).astype(np.float32)
    final_norm_b = np.random.randn(model_dims).astype(np.float32)
    tok_embeddings_w = np.random.randn(vocab_size, model_dims).astype(
        np.float32
    )
    softmax_w = np.random.randn(model_dims, vocab_size).astype(np.float32)
    softmax_b = np.random.randn(vocab_size).astype(np.float32)

    # GPTJ config
    layer_weights = [{
        'attn': {
            'q_proj': {'kernel': wq},
            'k_proj': {'kernel': wk},
            'v_proj': {'kernel': wv},
            'out_proj': {'kernel': wo},
        },
        'mlp': {
            'fc_in': {'kernel': w1, 'bias': b1},
            'fc_out': {'kernel': w2, 'bias': b2},
        },
        'ln_1': {'scale': layernorm_w, 'bias': layernorm_b},
    }]
    gptj_xformer = FlaxGPTJForCausalLMModule(
        tie_word_embeddings=False,
        num_hidden_layers=1,
        vocab_size=vocab_size,
        embd_pdrop=embd_pdrop,
        hidden_size=model_dims,
        n_inner=hidden_dims,
        layer_norm_epsilon=layer_norm_epsilon,
        num_attention_heads=num_heads,
        rotary_dim=rotary_dim,
        initializer_range=initializer_range,
        resid_pdrop=resid_pdrop,
        max_position_embeddings=max_seq_len,
        attn_pdrop=attn_pdrop,
        dtype=dtype,
    )
    gptj_params = freeze({
        'transformer': {
            'wte': {'embedding': jnp.asarray(tok_embeddings_w)},
            'ln_f': {
                'scale': jnp.asarray(final_norm_w),
                'bias': jnp.asarray(final_norm_b),
            },
            'h': {'0': layer_weights[0]},
        },
        'lm_head': {
            'kernel': jnp.asarray(softmax_w),
            'bias': jnp.asarray(softmax_b),
        },
    })
    gptj_logits = gptj_xformer.apply(
        {'params': gptj_params},
        jnp.asarray(x),
        jnp.ones((max_batch_size, max_seq_len), dtype=np.int32),
        segment_pos,
    )

    # PAX model
    model_p = pax_fiddle.Config(praxis_layers.TransformerLm)
    model_p.packed_input = False
    model_p.position_emb_tpl = None
    model_p.model_dims = model_dims
    model_p.vocab_size = vocab_size
    model_p.softmax_tpl = pax_fiddle.Config(
        praxis_layers.FullSoftmax,
        name='output',
        input_dims=model_dims,
        num_classes=vocab_size,
    )
    model_p.softmax_tpl.feed_forward_tpl.has_bias = True
    model_p.separate_embedding_tpl = pax_fiddle.Config(
        praxis_layers.Embedding,
        name='tok_embeddings',
        input_dims=model_dims,
        num_classes=vocab_size,
    )
    ln_tpl = pax_fiddle.Config(
        praxis_layers.LayerNorm,
        name='norm',
        direct_scale=True,
        epsilon=layer_norm_epsilon,
    )
    model_p.final_ln_tpl = ln_tpl.clone()

    stacked_transformer_tpl = pax_fiddle.Config(praxis_layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = model_dims
    stacked_transformer_tpl.hidden_dims = hidden_dims
    stacked_transformer_tpl.num_layers = 1
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.dim_per_head = model_dims // num_heads

    transformer_layer_p = pax_fiddle.Config(
        sax_layers.ParallelTransformerOnlyNormAttentionInputs
    )
    transformer_layer_p.norm_policy = 'pre'
    transformer_layer_p.ln_tpl = ln_tpl.clone()
    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    transformer_layer_p.tr_atten_tpl.internal_enable_query_scale = True
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    rotary_position_emb_p = (
        pax_fiddle.Config(gptj_layers.GPTJRotaryEmbedding)
    )
    rotary_position_emb_p.max_position_embeddings = max_seq_len
    rotary_position_emb_p.rotary_dim = rotary_dim
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True
    transformer_layer_p.tr_atten_tpl.consolidate_rope_key_state = True
    transformer_layer_p.tr_atten_tpl.rotary_position_emb_tpl = (
        rotary_position_emb_p)

    transformer_layer_p.tr_fflayer_tpl.has_bias = True
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        activations.GELU
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = False
    transformer_layer_p.tr_fflayer_tpl.add_skip_connection = False
    stacked_transformer_tpl.transformer_layer_params_tpl = transformer_layer_p

    model_p.stacked_transformer_tpl = stacked_transformer_tpl
    model_p.fprop_dtype = fprop_dtype
    model_p.dtype = model_dtype
    model = model_p.Instantiate()
    initial_vars = model.init(init_key, x, paddings)

    wc = np.stack([wq, wk, wv], axis=0)
    wc = np.reshape(wc, [3, model_dims, num_heads, model_dims // num_heads])
    xformer_layer_w = initial_vars['params']['transformer']['x_layers_0']
    xformer_layer_w['self_attention']['combined_qkv']['w'] = wc
    xformer_layer_w['self_attention']['post']['w'] = np.reshape(
        wo.transpose(), [model_dims, num_heads, model_dims // num_heads]
    )
    xformer_layer_w['ff_layer']['ffn_layer1']['linear']['w'] = w1
    xformer_layer_w['ff_layer']['ffn_layer1']['bias']['b'] = b1
    xformer_layer_w['ff_layer']['ffn_layer2']['linear']['w'] = w2
    xformer_layer_w['ff_layer']['ffn_layer2']['bias']['b'] = b2
    xformer_layer_w['layer_norm']['scale'] = layernorm_w
    xformer_layer_w['layer_norm']['bias'] = layernorm_b
    initial_vars['params']['final_ln']['scale'] = final_norm_w
    initial_vars['params']['final_ln']['bias'] = final_norm_b
    initial_vars['params']['softmax']['logits_ffn']['linear']['w'] = softmax_w
    initial_vars['params']['softmax']['logits_ffn']['bias']['b'] = softmax_b
    initial_vars['params']['embedding_lookup']['emb_var'] = tok_embeddings_w
    pax_out = model.apply(
        initial_vars,
        x,
        paddings,
    )
    pax_logits = pax_out.logits

    # Argmax (Top 1)
    flax_probs_1, flax_indices_1 = jax.lax.top_k(gptj_logits, 1)
    pax_probs_1, pax_indices_1 = jax.lax.top_k(pax_logits[0], 1)
    self.assertAllClose(flax_probs_1, pax_probs_1, atol=.4)
    self.assertAllClose(flax_indices_1, pax_indices_1)

    # Top 4
    flax_probs_4, flax_indices_4 = jax.lax.top_k(gptj_logits, 4)
    pax_probs_4, pax_indices_4 = jax.lax.top_k(pax_logits[0], 4)
    flax_indices_4 = jnp.sort(flax_indices_4, axis=1)
    pax_indices_4 = jnp.sort(pax_indices_4, axis=1)
    self.assertAllClose(flax_probs_4, pax_probs_4, atol=.4)
    self.assertAllClose(flax_indices_4, pax_indices_4)

    # Top 8
    flax_probs_8, flax_indices_8 = jax.lax.top_k(gptj_logits, 8)
    pax_probs_8, pax_indices_8 = jax.lax.top_k(pax_logits[0], 8)
    flax_indices_8 = jnp.sort(flax_indices_8, axis=1)
    pax_indices_8 = jnp.sort(pax_indices_8, axis=1)
    self.assertAllClose(flax_probs_8, pax_probs_8, atol=.4)
    # Top 8 indices are flaky
    # self.assertAllClose(flax_indices_8, pax_indices_8, atol=1)

    # Top 10
    flax_probs_10, flax_indices_10 = jax.lax.top_k(gptj_logits, 10)
    pax_probs_10, pax_indices_10 = jax.lax.top_k(pax_logits[0], 10)
    flax_indices_10 = jnp.sort(flax_indices_10, axis=1)
    pax_indices_10 = jnp.sort(pax_indices_10, axis=1)
    self.assertAllClose(flax_probs_10, pax_probs_10, atol=.4)
    # Top 10 indices are flaky
    # self.assertAllClose(flax_indices_10, pax_indices_10, atol=1)

    # Top 20
    flax_probs_20, flax_indices_20 = jax.lax.top_k(gptj_logits, 20)
    pax_probs_20, pax_indices_20 = jax.lax.top_k(pax_logits[0], 20)
    flax_indices_20 = jnp.sort(flax_indices_20, axis=1)
    pax_indices_20 = jnp.sort(pax_indices_20, axis=1)
    self.assertAllClose(flax_probs_20, pax_probs_20, atol=.6)
    # Top 20 indices are flaky
    # self.assertAllClose(flax_indices_20, pax_indices_20, atol=1)

  # Matching PAX's extend_step() with forward pass (__call__). i.e. extend_steps
  # T times should be  numerically the same as call to call directly.
  def testGPTJExtendStep(self):
    vocab_size = 50401
    max_batch_size = 1
    max_seq_len = 1919

    model_dims = 4096
    hidden_dims = 16384
    num_heads = 16

    rotary_dim = 64
    layer_norm_epsilon = 1.0e-05

    np.random.seed(123456)

    decode_cache = base_layer.DECODE_CACHE

    # PAX model
    model_p = pax_fiddle.Config(praxis_layers.TransformerLm)
    model_p.packed_input = False
    model_p.position_emb_tpl = None
    model_p.model_dims = model_dims
    model_p.vocab_size = vocab_size
    model_p.softmax_tpl = pax_fiddle.Config(
        praxis_layers.FullSoftmax,
        name='output',
        input_dims=model_dims,
        num_classes=vocab_size,
    )
    model_p.softmax_tpl.feed_forward_tpl.has_bias = True
    model_p.separate_embedding_tpl = pax_fiddle.Config(
        praxis_layers.Embedding,
        name='tok_embeddings',
        input_dims=model_dims,
        num_classes=vocab_size,
    )
    model_p.final_ln_tpl.epsilon = layer_norm_epsilon

    stacked_transformer_tpl = pax_fiddle.Config(praxis_layers.StackedTransformer)
    stacked_transformer_tpl.model_dims = model_dims
    stacked_transformer_tpl.hidden_dims = hidden_dims
    stacked_transformer_tpl.num_layers = 1
    stacked_transformer_tpl.num_heads = num_heads
    stacked_transformer_tpl.dim_per_head = model_dims // num_heads

    transformer_layer_p = pax_fiddle.Config(
        sax_layers.ParallelTransformerOnlyNormAttentionInputs
    )
    transformer_layer_p.norm_policy = 'pre'
    transformer_layer_p.ln_tpl.epsilon = layer_norm_epsilon
    transformer_layer_p.tr_atten_tpl.internal_enable_per_dim_scale = False
    transformer_layer_p.tr_atten_tpl.internal_enable_query_scale = True
    transformer_layer_p.tr_atten_tpl.use_bias = False
    transformer_layer_p.tr_atten_tpl.combine_qkv = True
    rotary_position_emb_p = (
        pax_fiddle.Config(gptj_layers.GPTJRotaryEmbedding)
    )
    rotary_position_emb_p.max_position_embeddings = max_seq_len
    rotary_position_emb_p.rotary_dim = rotary_dim
    transformer_layer_p.tr_atten_tpl.use_rotary_position_emb = True
    transformer_layer_p.tr_atten_tpl.consolidate_rope_key_state = True
    transformer_layer_p.tr_atten_tpl.rotary_position_emb_tpl = (
        rotary_position_emb_p)

    transformer_layer_p.tr_fflayer_tpl.has_bias = True
    transformer_layer_p.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        activations.GELU
    )
    transformer_layer_p.tr_fflayer_tpl.use_gated_activation = False
    transformer_layer_p.tr_fflayer_tpl.add_skip_connection = False
    stacked_transformer_tpl.transformer_layer_params_tpl = transformer_layer_p

    model_p.stacked_transformer_tpl = stacked_transformer_tpl
    model = model_p.Instantiate()

    # Prepare inputs
    start_step = max_seq_len - 4
    npy_inputs = np.random.randint(
        vocab_size, size=(max_batch_size, max_seq_len)
    ).astype('int32')
    inputs = jnp.asarray(npy_inputs)
    # No paddings
    nopaddings = jnp.zeros_like(inputs)
    # First (max_seq_len - 4) tokens are prefixes.
    paddings_suffix = jnp.array(
        [[0.0]*start_step + [1.0, 1.0, 1.0, 1.0]], dtype=inputs.dtype
    )

    # Compare fprop vs 4 iterations of extend_step
    context_params = base_layer.JaxContext.HParams(do_eval=True)
    with base_layer.JaxContext.new_context(hparams=context_params):
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = model.init(
          prng_key,
          inputs,
          nopaddings,
      )
      fprop_outputs = model.apply(
          initial_vars,
          inputs,
          # Compute fprop output on the entire sequence (1919 tokens no padding)
          nopaddings,
          causal_attention_mask=paddings_suffix.astype(jnp.int32),
      )
      # Init decode states.
      _, decoder_state = model.apply(
          initial_vars,
          inputs,
          # Compute fprop output on the first 1915 tokens.
          # (hence 1915 padded tokens)
          paddings_suffix,
          causal_attention_mask=paddings_suffix.astype(jnp.int32),
          start_time_step=start_step,
          mutable=[decode_cache],
      )
      logits = fprop_outputs.logits
      updated_vars = py_utils.merge_dict(decoder_state, initial_vars)
      for t in range(start_step, max_seq_len):
        xent_output, decoder_state = model.apply(
            updated_vars,
            inputs[:, t],
            method=model.extend_step,
            mutable=[decode_cache],
        )
        updated_vars = py_utils.merge_dict(decoder_state, initial_vars)
        self.assertAllClose(logits[:, t, :], xent_output.logits)


if __name__ == '__main__':
  absltest.main()
