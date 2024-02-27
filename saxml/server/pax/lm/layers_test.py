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
"""Tests for sax optimized lm layers used in decoding."""

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from praxis import base_layer
from praxis import layers as praxis_layers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import test_utils
from praxis.layers import attentions
from saxml.server.pax.lm import layers


class LayersTest(test_utils.TestCase, parameterized.TestCase):

  def test_transformer_feedforward(self):
    p = pax_fiddle.Config(
        layers.TransformerFeedForwardWithSeqSplit,
        name='ffwd',
        input_dims=8,
        hidden_dims=32,
        use_gated_activation=True,
    )
    batch_size = 8
    seq_len = 512

    npy_inputs = np.random.normal(
        1.0, 0.5, [batch_size, seq_len, p.input_dims]
    ).astype('float32')
    npy_inputs[0:4, :, :] = 0
    inputs = jnp.asarray(npy_inputs)
    npy_paddings = np.zeros([batch_size, seq_len], dtype=np.float32)
    npy_paddings[0:4, :] = 1
    input_paddings = jnp.asarray(npy_paddings)

    with base_layer.JaxContext.new_context():
      ffwd = base_layer.instantiate(p)
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars = ffwd.init(prng_key, inputs, input_paddings)
      fprop_out = ffwd.apply(initial_vars, inputs, input_paddings)

    p2 = pax_fiddle.Config(
        praxis_layers.TransformerFeedForward,
        name='ffwd',
        input_dims=8,
        hidden_dims=32,
        use_gated_activation=True,
    )
    with base_layer.JaxContext.new_context():
      ffwd2 = base_layer.instantiate(p2)
      prng_key = jax.random.PRNGKey(seed=123)
      initial_vars2 = ffwd2.init(prng_key, inputs, input_paddings)
      fprop_out2 = ffwd2.apply(initial_vars2, inputs, input_paddings)
    self.assertAllClose(fprop_out, fprop_out2)

  @parameterized.parameters([
      (False, None),
      (True, None),
      (False, 1.0),
      (True, 1.0),

  ])
  def test_attention(
      self,
      use_rotary_position_emb,
      attention_extra_logit,
  ):
    mdl_dim = 16
    hidden_dim = 32
    num_heads = 4
    target_batch_size = 3
    source_max_length = 16
    target_max_length = 16
    layer_p = layers.MXUDotProductAttention
    test_layer_p = pax_fiddle.Config(
        layer_p,
        name='self_atten',
        input_dim=mdl_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dim_per_head=16 if use_rotary_position_emb else None,
        atten_logit_cap=20.0,
        attention_extra_logit=attention_extra_logit,
        use_rotary_position_emb=use_rotary_position_emb,
    )
    layer = base_layer.instantiate(test_layer_p)
    prng_key = jax.random.PRNGKey(seed=123)
    _, init_key = jax.random.split(prng_key)

    query_vec = np.random.normal(
        size=[target_batch_size, source_max_length, mdl_dim]
    ).astype(np.float32)
    query_vec[0, :, :] = 0
    key_vec = query_vec
    value_vec = query_vec
    fake_query_vec = jnp.zeros_like(query_vec)
    atten_mask = attentions.causal_mask(query_vec)
    segment_pos = np.tile(np.arange(source_max_length), (target_batch_size, 1))

    starting_index = 0

    with base_layer.JaxContext.new_context():
      initial_vars = layer.init(
          init_key,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          jnp.log(jnp.zeros_like(atten_mask)),
      )
      logging.info('initial_vars: %s', initial_vars)
      _, attention_states = layer.apply(
          initial_vars,
          fake_query_vec,
          fake_query_vec,
          fake_query_vec,
          jnp.log(jnp.zeros_like(atten_mask)),
          method=layer.__call__,
          mutable=[base_layer.DECODE_CACHE],
      )
      fprop_out, _ = layer.apply(
          initial_vars,
          query_vec,
          key_vec,
          value_vec,
          atten_mask,
          query_segment_pos=segment_pos,
          key_segment_pos=segment_pos,
          method=layer.__call__,
      )

      decoder_output = jnp.zeros(
          shape=[target_max_length, target_batch_size, mdl_dim]
      )

      updated_vars = py_utils.merge_dict(attention_states, initial_vars)
      for t in range(starting_index, target_max_length):
        encoded, attention_states = layer.apply(
            updated_vars,
            query_vec=query_vec[:, t, :],
            atten_mask=atten_mask[:, :, t, :],
            time_step=t,
            segment_pos=None,
            method=layer.extend_step,
            mutable=[base_layer.DECODE_CACHE],
        )
        updated_vars = py_utils.merge_dict(attention_states, initial_vars)
        decoder_output = decoder_output.at[t].set(encoded)

    decoder_output = decoder_output[starting_index:]
    decoder_out_transposed = jnp.transpose(decoder_output, [1, 0, 2])
    fprop_out = fprop_out[:, starting_index:]

    logging.info('fprop_out: %s', fprop_out)
    logging.info('decoder_out: %s', decoder_output)
    self.assertAllClose(fprop_out, decoder_out_transposed)


if __name__ == '__main__':
  absltest.main()
