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
"""Unstack and quantize a model."""

import argparse
import pathlib

from absl import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from jax import numpy as jnp
import quant_fn as quant_fn_lib
import quantization_actions
import quantization_provider


def parse_known_args(argv):
  """Parses args for the workflow."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input_dir', default=None, help='Directory for input data.'
  )
  parser.add_argument(
      '--prune_params_regex',
      default=None,
      help=(
          'Regex pattern to prune params from input checkpoint(s), matched'
          ' params will not be included in the quantized checkpoint.'
      ),
  )
  parser.add_argument(
      '--skip_params_regex',
      default=None,
      help=(
          'Regex pattern to skip params from input checkpoint(s), matched'
          ' params will stay unquantized in the quantized checkpoint.'
      ),
  )
  parser.add_argument(
      '--output_dir', default=None, help='Directory for output data.'
  )
  parser.add_argument(
      '--quantization_configs',
      default='gptj',
      choices=[
          'gptj',
          'gemma2b',
          'gemma7b',
          'llama2-70b-weight-linear-only-int8',
      ],
      help='Quantization Config.',
  )
  parser.add_argument(
      '--symmetric',
      default=True,
      type=lambda x: bool(str(x).lower() == 'true'),
      help='Symmetric quantize weight.',
  )
  parser.add_argument(
      '--transpose_embedding',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help='Transpose embedding to reduce latency.',
  )
  parser.add_argument(
      '--quantize_embedding',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help='Deprecated. Use quantization_configs. Quantize embedding weights.',
  )
  parser.add_argument(
      '--number_bits',
      type=int,
      default=8,
      help='The default number of bits to quantize.',
  )
  parser.add_argument(
      '--use_optimization',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help='If True use efficient automl optimization.',
  )
  parser.add_argument(
      '--per_channel_clipping_optimization',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help=(
          'If True use per channel clipping optimization.'
          'When True, only supports linear-only quantization'
      ),
  )
  parser.add_argument(
      '--optimization_p_value',
      type=float,
      default=4,
      help='The p value for efficient automl optimization.',
  )
  parser.add_argument(
      '--use_int4_packed_weights',
      default=True,
      type=lambda x: bool(str(x).lower() == 'true'),
      help='If True use int4 packing into int32.',
  )
  parser.add_argument(
      '--int4_packed_dtype',
      default='int32',
      choices=['int32', 'int8'],
      help='Container type to pack int4 weights to, either int32 or int8.',
  )
  parser.add_argument(
      '--stacked',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help=(
          'If True, quantized but still stacked, otherwise we will get'
          ' unstacked version of the checkpoint which is compatible with'
          ' REPEAT=False for the model config.'
      ),
  )
  parser.add_argument(
      '--use_fp',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help='If True, quantized into fp8 (jnp.float8_e4m3fn).',
  )
  parser.add_argument(
      '--add_scale_eps',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help=(
          'If True, add epsilon to scale to avoid division by zero, else it'
          ' will replace zero scale by 1.'
      ),
  )
  parser.add_argument(
      '--quantize_ngrammer_embedding',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help='Quantize ngrammer embedding weights.',
  )
  parser.add_argument(
      '--model_vars_tag',
      default='mdl_vars.params',
      help=(
          'Only the vars (+ `step` vars) that includes this tag string in its'
          ' source name is quantized/copied to resulting ckpt.'
      ),
  )
  parser.add_argument(
      '--accelerator_type',
      default='cpu',
      choices=['cpu'],
      help=(
          'Accelerator for running offline_quantizer. Acclerators have numeric'
          ' differences which might lead to different quantized weights.'
      ),
  )
  parser.add_argument(
      '--preserve_shardings',
      default=False,
      type=lambda x: bool(str(x).lower() == 'true'),
      help=(
          'If True, the output checkpoint will have the same sharding format'
          ' with the input checkpoint.'
      ),
  )
  return parser.parse_known_args(argv)


NAME_TO_CONFIG = quantization_provider.get_name_to_config(stacked=False)

NAME_TO_CONFIG_STACKED = quantization_provider.get_name_to_config(stacked=True)


def run(argv=None, save_main_session=True):
  """Runs the offline quantization script."""
  known_args, pipeline_args = parse_known_args(argv)
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

  output_dir = known_args.output_dir
  default_num_bits = known_args.number_bits
  use_optimization = (
      known_args.use_optimization
  )
  optimization_p_value = known_args.optimization_p_value
  use_int4_packed_weights = known_args.use_int4_packed_weights
  int4_packed_dtype = (
      jnp.int8 if known_args.int4_packed_dtype == 'int8' else jnp.int32
  )
  if known_args.stacked:
    quant_config = NAME_TO_CONFIG_STACKED.get(known_args.quantization_configs)
  else:
    quant_config = NAME_TO_CONFIG.get(known_args.quantization_configs)

  actions = quantization_actions.create_actions(
      known_args.input_dir,
      quant_config,
      transpose=known_args.transpose_embedding,
      quantize_embedding=known_args.quantize_embedding,
      number_bit=default_num_bits,
      prune_params_regex=known_args.prune_params_regex,
      skip_params_regex=known_args.skip_params_regex,
      use_optimization=use_optimization,
      optimization_p_value=optimization_p_value,
      per_channel_clipping=known_args.per_channel_clipping_optimization,
      use_int4_packed_weights=use_int4_packed_weights,
      int4_packed_dtype=int4_packed_dtype,
      stacked=known_args.stacked,
      use_fp=known_args.use_fp,
      add_scale_eps=known_args.add_scale_eps,
      quantize_ngrammer_embedding=known_args.quantize_ngrammer_embedding,
      model_var_tags=known_args.model_vars_tag,
      preserve_shardings=known_args.preserve_shardings,
  )

  logging.info(
      '\n'.join(
          f'{index}th action {action}' for index, action in enumerate(actions)
      )
  )

  input_dirs = [known_args.input_dir]

  if pathlib.Path(output_dir).is_dir():
    logging.info('Removing existing files in %s', output_dir)
    pathlib.Path(output_dir).rmdir()
  pathlib.Path(output_dir).mkdir(parents=True)

  with beam.Pipeline(options=pipeline_options) as p:
    quant_fn = beam.ParDo(
        quant_fn_lib.QuantFn(
            input_dirs,
            output_dir,
            known_args.symmetric,
        )
    )

    if known_args.accelerator_type != 'cpu':
      quant_fn = quant_fn.with_resource_hints(
          allowed_accelerator_types=[(known_args.accelerator_typee, 1, 1)]
      )
    _ = p | beam.Create(actions) | quant_fn


if __name__ == '__main__':
  run()
