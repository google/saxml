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
"""Compatibility test between PAX and Flax gemma transformer."""

from absl.testing import absltest
from flax import linen as nn
from flax.core import frozen_dict
from gemma import modules as gemma_modules
from gemma import transformer as gemma_transformer
from jax import numpy as jnp
import numpy as np
from praxis import test_utils
from saxml.server.pax.lm import transformer_models as pax_transformer_models


class GemmaTest(test_utils.TestCase):

  def _dummy_flax_gemma(
      self,
      vocab_size,
      model_dims,
      hidden_dims,
      num_layers,
      num_heads,
      dim_per_head,
      num_kv_heads,
      max_cache_length,
  ) -> nn.Module:
    return gemma_transformer.Transformer(
        config=gemma_transformer.TransformerConfig(
            num_embed=vocab_size,
            embed_dim=model_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=dim_per_head,
            hidden_dim=hidden_dims,
            num_kv_heads=num_kv_heads,
            max_cache_length=max_cache_length,
            use_post_ffw_norm=False,
            use_post_attn_norm=False,
            attention_types=[gemma_modules.AttentionType.GLOBAL]
            * num_layers,
            final_logit_softcap=None,
        )
    )

  def _dummy_pax_gemma(
      self,
      vocab_size,
      model_dims,
      hidden_dims,
      num_layers,
      num_heads,
      dim_per_head,
      use_mqa,
      chunked_one_step_attn_num_seq_split=4,
  ) -> nn.Module:
    model_p = pax_transformer_models.gemma(
        vocab_size,
        model_dims,
        hidden_dims,
        num_layers,
        num_heads,
        dim_per_head,
        use_mqa=use_mqa,
        chunked_one_step_attn_num_seq_split=chunked_one_step_attn_num_seq_split,
    )

    return model_p.Instantiate()

  def test_mha_transformer(self):
    vocab_size = 500
    model_dims = 40
    hidden_dims = model_dims * 16
    num_layers = 1
    num_heads = 4
    dim_per_head = 8
    max_seq_len = 1
    flax_gemma_test_model = self._dummy_flax_gemma(
        vocab_size=vocab_size,
        model_dims=model_dims,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        num_kv_heads=num_heads,
        max_cache_length=max_seq_len,
    )
    pax_gemma_test_model = self._dummy_pax_gemma(
        vocab_size=vocab_size,
        model_dims=model_dims,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        use_mqa=False,
    )
    max_batch_size = 1
    np.random.seed(123456)
    x = np.random.randint(
        low=0,
        high=vocab_size,
        size=(max_batch_size, max_seq_len),
        dtype=np.int32,
    )
    attention_mask = jnp.ones(1)
    cache = {
        f"layer_{i}": gemma_modules.Attention.init_cache(
            max_seq_len,
            num_heads,
            dim_per_head,
            max_batch_size,
            dtype=jnp.float32,
        )
        for i in range(num_layers)
    }
    paddings = jnp.zeros_like(x, dtype=jnp.int32)

    wq = np.random.randn(model_dims, num_heads, dim_per_head).astype(np.float32)
    wk = np.random.randn(model_dims, num_heads, dim_per_head).astype(np.float32)
    wv = np.random.randn(model_dims, num_heads, dim_per_head).astype(np.float32)
    wc = np.stack([wq, wk, wv], axis=0)
    flax_wc = np.moveaxis(wc, 1, 2)
    pax_wc = wc
    wo = np.random.randn(model_dims, num_heads, dim_per_head).astype(np.float32)
    flax_wo = np.transpose(wo, [1, 2, 0])
    pax_wo = wo

    w1_gate = np.random.randn(model_dims, hidden_dims).astype(np.float32)
    w1 = np.random.randn(model_dims, hidden_dims).astype(np.float32)

    w2 = np.random.randn(hidden_dims, model_dims).astype(np.float32)

    pre_attention_norm = np.random.randn(model_dims).astype(np.float32)
    pre_ffw_norm = np.random.randn(model_dims).astype(np.float32)
    final_norm_w = np.random.randn(model_dims).astype(np.float32)
    tok_embeddings_w = np.random.randn(vocab_size, model_dims).astype(
        np.float32
    )

    flax_params = frozen_dict.freeze(
        {
            "embedder": {"input_embedding": tok_embeddings_w},
            "final_norm": {"scale": final_norm_w},
            "layer_0": {
                "attn": {
                    "attn_vec_einsum": {
                        "w": flax_wo,
                    },
                    "qkv_einsum": {"w": flax_wc},
                },
                "mlp": {
                    "gating_einsum": np.stack((w1_gate, w1), axis=0),
                    "linear": w2,
                },
                "pre_attention_norm": {"scale": pre_attention_norm},
                "pre_ffw_norm": {"scale": pre_ffw_norm},
            },
        },
    )
    positions = gemma_transformer.build_positions_from_mask(paddings)
    flax_logits, _ = flax_gemma_test_model.apply(
        {"params": flax_params}, x, positions, cache, attention_mask
    )

    pax_params = frozen_dict.freeze({
        "final_ln": {"scale": final_norm_w},
        "softmax": {"w": tok_embeddings_w},
        "transformer": {
            "x_layers_0": {
                "ff_layer": {
                    "ffn_layer1": {"linear": {"w": w1}},
                    "ffn_layer1_gate": {"linear": {"w": w1_gate}},
                    "ffn_layer2": {"linear": {"w": w2}},
                    "layer_norm": {"scale": pre_ffw_norm},
                },
                "layer_norm": {"scale": pre_attention_norm},
                "self_attention": {
                    "combined_qkv": {"w": pax_wc},
                    "post": {
                        "w": pax_wo,
                    },
                },
            }
        },
    })
    pax_logits = pax_gemma_test_model.apply(
        {"params": pax_params},
        x,
        paddings,
    ).logits
    self.assertArraysEqual(
        flax_logits.reshape(max_batch_size, vocab_size),
        pax_logits.reshape(max_batch_size, vocab_size),
    )

  def test_mqa_transformer(self):
    vocab_size = 500
    model_dims = 40
    hidden_dims = model_dims * 16
    num_layers = 1
    num_heads = 4
    dim_per_head = 8
    max_seq_len = 1
    flax_gemma_test_model = self._dummy_flax_gemma(
        vocab_size=vocab_size,
        model_dims=model_dims,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        num_kv_heads=1,
        max_cache_length=max_seq_len,
    )
    pax_gemma_test_model = self._dummy_pax_gemma(
        vocab_size=vocab_size,
        model_dims=model_dims,
        hidden_dims=hidden_dims,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_per_head=dim_per_head,
        use_mqa=True,
    )
    max_batch_size = 1
    np.random.seed(123456)
    x = np.random.randint(
        low=0,
        high=vocab_size,
        size=(max_batch_size, max_seq_len),
        dtype=np.int32,
    )
    attention_mask = jnp.ones(1)
    cache = {
        f"layer_{i}": gemma_modules.Attention.init_cache(
            max_seq_len,
            1,
            dim_per_head,
            max_batch_size,
            dtype=jnp.float32,
        )
        for i in range(num_layers)
    }
    paddings = jnp.zeros_like(x, dtype=jnp.int32)

    wq = np.random.randn(model_dims, num_heads, dim_per_head).astype(np.float32)
    wk = np.random.randn(model_dims, dim_per_head).astype(np.float32)
    wv = np.random.randn(model_dims, dim_per_head).astype(np.float32)
    flax_wq = np.moveaxis(wq, 0, 1)
    flax_wkv = np.expand_dims(np.stack([wk, wv], axis=0), axis=1)
    wo = np.random.randn(model_dims, num_heads, dim_per_head).astype(np.float32)
    flax_wo = np.transpose(wo, [1, 2, 0])
    pax_wo = wo

    w1_gate = np.random.randn(model_dims, hidden_dims).astype(np.float32)
    w1 = np.random.randn(model_dims, hidden_dims).astype(np.float32)

    w2 = np.random.randn(hidden_dims, model_dims).astype(np.float32)

    pre_attention_norm = np.random.randn(model_dims).astype(np.float32)
    pre_ffw_norm = np.random.randn(model_dims).astype(np.float32)
    final_norm_w = np.random.randn(model_dims).astype(np.float32)
    tok_embeddings_w = np.random.randn(vocab_size, model_dims).astype(
        np.float32
    )

    flax_params = frozen_dict.freeze(
        {
            "embedder": {"input_embedding": tok_embeddings_w},
            "final_norm": {"scale": final_norm_w},
            "layer_0": {
                "attn": {
                    "attn_vec_einsum": {
                        "w": flax_wo,
                    },
                    "kv_einsum": {"w": flax_wkv},
                    "q_einsum": {"w": flax_wq},
                },
                "mlp": {
                    "gating_einsum": np.stack((w1_gate, w1), axis=0),
                    "linear": w2,
                },
                "pre_attention_norm": {"scale": pre_attention_norm},
                "pre_ffw_norm": {"scale": pre_ffw_norm},
            },
        },
    )
    positions = gemma_transformer.build_positions_from_mask(paddings)
    flax_logits, _ = flax_gemma_test_model.apply(
        {"params": flax_params}, x, positions, cache, attention_mask
    )

    pax_params = frozen_dict.freeze({
        "final_ln": {"scale": final_norm_w},
        "softmax": {"w": tok_embeddings_w},
        "transformer": {
            "x_layers_0": {
                "ff_layer": {
                    "ffn_layer1": {"linear": {"w": w1}},
                    "ffn_layer1_gate": {"linear": {"w": w1_gate}},
                    "ffn_layer2": {"linear": {"w": w2}},
                    "layer_norm": {"scale": pre_ffw_norm},
                },
                "layer_norm": {"scale": pre_attention_norm},
                "self_attention": {
                    "query": {"w": wq},
                    "key": {"w": wk},
                    "value": {"w": wv},
                    "post": {
                        "w": pax_wo,
                    },
                },
            }
        },
    })
    pax_logits = pax_gemma_test_model.apply(
        {"params": pax_params},
        x,
        paddings,
    ).logits
    self.assertArraysEqual(
        flax_logits.reshape(max_batch_size, vocab_size),
        pax_logits.reshape(max_batch_size, vocab_size),
    )


if __name__ == "__main__":
  absltest.main()
