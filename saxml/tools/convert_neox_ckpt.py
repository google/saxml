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

# Get a checkpiont from the GPT-NEOX family
https://huggingface.co/databricks/dolly-v2-3b

# Example cmd:
python3 -m convert_neox_ckpt --base dolly-v2-3b --pax pax_3b
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

# 3B example
num_layers = 32
num_heads = 32
dims_per_head = 80
vocab = 50280
num_gpus = 1


def convert(base_model_path, pax_model_path):
  """Convert from vicuna to pax."""
  print(f'Loading the base model from {base_model_path}')

  base = AutoModelForCausalLM.from_pretrained(base_model_path,
                                              low_cpu_mem_usage=True)
  for key, value in base.state_dict():
    print('%s %s' % (key, value.data.numpy().shape))

  jax_weights = {
      'lm': {
          'embedding_lookup': {
              'emb_var': base.state_dict()['gpt_neox.embed_in.weight'].data.numpy()[:vocab,:]
              },
          'softmax': {
              'logits_ffn': {
                  'linear': {
                      'w': base.state_dict()['embed_out.weight'].data.numpy().transpose()[:, :vocab]
                      }
                  }
              },
          'final_ln': {
              'scale': base.state_dict()['gpt_neox.final_layer_norm.weight'].data.numpy(),
              'bias': base.state_dict()['gpt_neox.final_layer_norm.bias'].data.numpy()
              },
          'transformer': {}
          }
      }
  for layer_idx in range(num_layers):
    wc = base.state_dict()['gpt_neox.layers.%d.attention.query_key_value.weight' % layer_idx].data.numpy()
    wc = np.reshape(wc, [3, num_heads, dims_per_head, num_heads * dims_per_head])
    wc = np.transpose(wc, (0, 3, 1, 2))

    bc = base.state_dict()['gpt_neox.layers.%d.attention.query_key_value.bias' % layer_idx].data.numpy()

    w_post = base.state_dict()[
        'gpt_neox.layers.%d.attention.dense.weight' % layer_idx
    ].data.numpy()
    w_post = np.reshape(w_post, [num_heads * dims_per_head, num_heads, dims_per_head])
    b_post = base.state_dict()[
        'gpt_neox.layers.%d.attention.dense.bias' % layer_idx
    ].data.numpy()
    layer_weight = {
        'self_attention': {
            'combined_qkv': {
                'w': wc,
                'b': bc,
                },
            'post': {
                'w': w_post,
                'b': b_post,
                }
            },
        'ff_layer': {
            'ffn_layer1': {
                'linear': {
                    'w': base.state_dict()['model.layers.%d.mlp.dense_h_to_4h.weight' % layer_idx].data.numpy().transpose(),
                    'b': base.state_dict()['model.layers.%d.mlp.dense_h_to_4h.bias' % layer_idx].data.numpy(),
                    }
                },
            'ffn_layer2': {
                'linear': {
                    'w': base.state_dict()['model.layers.%d.mlp.dense_4h_to_h.weight' % layer_idx].data.numpy().transpose(),
                    'b': base.state_dict()['model.layers.%d.mlp.dense_4h_to_h.bias' % layer_idx].data.numpy()
                    }
                },
            },
        'pre_layer_norm': {
            'scale': base.state_dict()['model.layers.%d.input_layernorm.weight' % layer_idx].data.numpy(),
            'bias': base.state_dict()['model.layers.%d.input_layernorm.bias' % layer_idx].data.numpy()
            },
        'post_layer_norm': {
            'scale': base.state_dict()['model.layers.%d.post_attention_layernorm.weight' % layer_idx].data.numpy(),
            'bias': base.state_dict()['model.layers.%d.post_attention_layernorm.bias' % layer_idx].data.numpy()
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
