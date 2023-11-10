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
r"""Convert weights from a llama/vicuna model to a pax one.

Usage:

# Install the latest main branch of huggingface/transformers
pip3 install git+https://github.com/huggingface/transformers

# Get LLaMA ckpts from Meta, or follow the steps
# in https://github.com/lm-sys/FastChat to get vicuna weights.

# Example cmd:
python3 -m convert_vicuna_ckpt --base vicuna_7b --pax pax_7b
"""
# pylint: disable=g-line-too-long
import argparse
import jax
from jax.experimental import pjit
import numpy as np

from paxml import checkpoints
from paxml import train_states
from praxis import py_utils

from transformers import AutoModelForCausalLM


num_layers = 32
num_heads = 32
dims_per_head = 128
vocab = 32000
num_gpus = 1


def convert(base_model_path, pax_model_path):
  """Convert from vicuna to pax."""
  print(f'Loading the base model from {base_model_path}')

  base = AutoModelForCausalLM.from_pretrained(
      base_model_path, low_cpu_mem_usage=True)

  jax_weights = {
      'lm': {
          'embedding_lookup': {
              'emb_var': base.state_dict()['model.embed_tokens.weight'].data.numpy()[:vocab,:]
              },
          'softmax': {
              'logits_ffn': {
                  'linear': {
                      'w': base.state_dict()['lm_head.weight'].data.numpy().transpose()[:, :vocab]
                      }
                  }
              },
          'final_ln': {
              'scale': base.state_dict()['model.norm.weight'].data.numpy()
              },
          'transformer': {}
          }
      }
  for layer_idx in range(num_layers):
    wq = base.state_dict()['model.layers.%d.self_attn.q_proj.weight' % layer_idx].data.numpy().transpose()
    wk = base.state_dict()['model.layers.%d.self_attn.k_proj.weight' % layer_idx].data.numpy().transpose()
    wv = base.state_dict()['model.layers.%d.self_attn.v_proj.weight' % layer_idx].data.numpy().transpose()
    wc = np.stack([wq, wk, wv], axis=0)
    wc = np.reshape(wc, [3, num_heads * dims_per_head, num_heads, dims_per_head])

    w_post = base.state_dict()[
        'model.layers.%d.self_attn.o_proj.weight' % layer_idx
    ].data.numpy()
    w_post = np.reshape(w_post, [num_heads * dims_per_head, num_heads, dims_per_head])
    layer_weight = {
        'self_attention': {
            'combined_qkv': {
                'w': wc
                },
            'post': {
                'w': w_post
                }
            },
        'ff_layer': {
            'ffn_layer1_gate': {
                'linear': {
                    'w': base.state_dict()['model.layers.%d.mlp.gate_proj.weight' % layer_idx].data.numpy().transpose()
                    }
                },
            'ffn_layer1': {
                'linear': {
                    'w': base.state_dict()['model.layers.%d.mlp.up_proj.weight' % layer_idx].data.numpy().transpose()
                    }
                },
            'ffn_layer2': {
                'linear': {
                    'w': base.state_dict()['model.layers.%d.mlp.down_proj.weight' % layer_idx].data.numpy().transpose()
                    }
                },
            'layer_norm': {
                'scale': base.state_dict()['model.layers.%d.post_attention_layernorm.weight' % layer_idx].data.numpy()
                }
            },
        'layer_norm': {
            'scale': base.state_dict()['model.layers.%d.input_layernorm.weight' % layer_idx].data.numpy()
            }
        }
    jax_weights['lm']['transformer']['x_layers_%d' % layer_idx] = layer_weight

  print(f'Saving the pax model to {pax_model_path}')
  jax_states = train_states.TrainState(
      step=0,
      mdl_vars={'params': jax_weights},
      opt_states={})

  device_mesh = py_utils.create_device_mesh([1, 1, num_gpus])
  global_mesh = jax.sharding.Mesh(
      device_mesh, ['replica', 'data_mdl2', 'mdl'])

  # Identity pjit is needed to output a GDA model_states.
  def identity(x):
    return x

  pjitted_identity = pjit.pjit(identity,
                               in_shardings=None,
                               out_shardings=None)

  with global_mesh:
    jax_states_gda = pjitted_identity(jax_states)

  checkpoints.save_checkpoint(
      jax_states_gda, pax_model_path,
      checkpoint_type=checkpoints.CheckpointType.GDA)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base-model-path', type=str, required=True)
  parser.add_argument('--pax-model-path', type=str, required=True)
  args = parser.parse_args()

  convert(args.base_model_path, args.pax_model_path)
