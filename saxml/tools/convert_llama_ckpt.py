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

# Get LLaMA pytorch_vars from Meta,or follow the steps  

# Example cmd:
python3 -m convert_llama_ckpt --base llama_7b --pax pax_7b
"""
# pylint: disable=g-line-too-long
import argparse
import pathlib

import jax
from jax.experimental import pjit
import numpy as np

from paxml import checkpoints
from paxml import train_states
from praxis import py_utils

import torch

num_layers = 32
num_heads = 32
dims_per_head = 128
vocab = 32000
num_gpus = 1


def convert(base_model_path, pax_model_path):
  """Convert from vicuna to pax."""
  print(f'Loading the base model from {base_model_path}')
  ckpt_paths = sorted(pathlib.Path(base_model_path).glob('*.pth'))
  pytorch_vars = {}
  for i, ckpt_path in enumerate(ckpt_paths):
    print(f'Loading checkpoint {i+1} of {len(ckpt_paths)} ...')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    pytorch_vars[int(ckpt_path.name.split('.', maxsplit=2)[1])] = checkpoint
  pytorch_vars = [pytorch_vars[i] for i in sorted(list(pytorch_vars.keys()))]

  jax_weights = {
      'lm': {
          'embedding_lookup': {
              'emb_var': np.concatenate([var['tok_embeddings.weight'].numpy() for var in pytorch_vars], axis=1)[:vocab,:]
              },
          'softmax': {
              'logits_ffn': {
                  'linear': {
                      'w': np.concatenate([var['output.weight'].numpy() for var in pytorch_vars], axis=0).transpose()[:, :vocab]
                      }
                  }
              },
          'final_ln': {
              'scale': pytorch_vars[0]['norm.weight'].numpy()
              },
          'transformer': {}
          }
      }
  for layer_idx in range(num_layers):
    wq = np.concatenate([var['layers.%d.attention.wq.weight' % (layer_idx)].numpy() for var in pytorch_vars], axis=0).transpose()
    wk = np.concatenate([var['layers.%d.attention.wk.weight' % (layer_idx)].numpy() for var in pytorch_vars], axis=0).transpose()
    wv = np.concatenate([var['layers.%d.attention.wv.weight' % (layer_idx)].numpy() for var in pytorch_vars], axis=0).transpose()
    wc = np.stack([wq, wk, wv], axis=0)
    wc = np.reshape(wc, [3, num_heads * dims_per_head, num_heads, dims_per_head])

    w_post = np.concatenate(
        [
            var['layers.%d.attention.wo.weight' % (layer_idx)].numpy()
            for var in pytorch_vars
        ],
        axis=1,
    )
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
                    'w': np.concatenate([var['layers.%d.feed_forward.w1.weight' % (layer_idx)].numpy() for var in pytorch_vars], axis=0).transpose()
                    }
                },
            'ffn_layer1': {
                'linear': {
                    'w': np.concatenate([var['layers.%d.feed_forward.w3.weight' % (layer_idx)].numpy() for var in pytorch_vars], axis=0).transpose()
                    }
                },
            'ffn_layer2': {
                'linear': {
                    'w': np.concatenate([var['layers.%d.feed_forward.w2.weight' % (layer_idx)].numpy() for var in pytorch_vars], axis=1).transpose()
                    }
                },
            'layer_norm': {
                'scale': pytorch_vars[0]['layers.%d.ffn_norm.weight' % (layer_idx)].numpy()
                }
            },
        'layer_norm': {
            'scale': pytorch_vars[0]['layers.%d.attention_norm.weight' % (layer_idx)].numpy()
            }
        }
    jax_weights['lm']['transformer']['x_layers_%d' % layer_idx] = layer_weight

  print(f'Saving the pax model to {pax_model_path}')
  jax_states = train_states.TrainState(
      step=np.zeros(1),
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
