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
"""Weight only quantization transformations on HParams of various layers.

Those functions help creating quantized models/params for sax system.

Example usage:

class QuantizedXYZModel(XYZModel):
  MODE = quantization_hparams.QuantizationMode.INFERENCE
  TYPE = quantization_hparams.QuantizationType.PTQ

  def task(self):
    task_p = super().task()
    quantize_transformer_layer_weights(
        task_p.model.lm_tpl.stacked_transformer_tpl.block
        .transformer_layer_params_tpl, self.TYPE, self.MODE)
    return task_p

This creates a quantized model for the original XYZModel by quantizing all
transformer blocks.


TODO(jianlijianli): extend this part to include end-to-end workflow when it's
ready.
"""

import functools

from jax import numpy as jnp
from praxis import layers
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantize


# TODO(jianlijianli): Merge this with the decorator in pax.
# Ready-to-use quantization decorators for quantizing transformer.
def for_transformer(
    quantize_on_the_fly=True,
    num_bits: int = 8,
    linear_only: bool = False,
    use_symmetric: bool = True,
    quantize_embedding_softmax: bool = False,
    transposed_embedding_softmax: bool = False,
    quantize_ngrammer_embedding: bool = False,
    dtype: jnp.dtype = jnp.int8,
):
  """Find and quantize transformer.

  If there are transformers that shouldn't be quantized, use the quantize_*
  functions and manually/selectively quantize the model.

  If there are no transformers in the model, it's a no-op.

  Note that this decorator is only for weight-only quantization.

  Args:
    quantize_on_the_fly: If the model is to be quantized on the fly. - Defaults
      to True, and the input model is float, and quantization happen on the fly.
      - When set to False, the input model is already quantized.
    num_bits: number of bits for quantized weights. Currently supports 8 and 4
      but any integer [1, 8] works.
    linear_only: Quantize only the linear layers.
    use_symmetric: use symmetric weight quantization.
    quantize_embedding_softmax: If true, Quantize embedding table of embedding
      softmax layer. Regardless of `target` argument, this results in rewriting
      TransformerLm.softmax_tpl in `config`.
    transposed_embedding_softmax: If the model is using transposed embedding for
      embedding softmax layer.
    quantize_ngrammer_embedding: If true, Quantize embedding table of each
      embedding in Ngrammer/VQNgrammer layer. Regardless of `target` argument,
      this results in rewriting TransformerLm.ngrammer_tpl in `config`.
    dtype: Dtype of the quantized variables.

  Returns:
    a modifier that quantizes transformers when applied to a config.
  """

  def decorator(cls):
    """decorator that quantize transformers."""

    @functools.wraps(cls, updated=())  # to keep original class name.
    class Wrapper(cls):
      """Wrapper class for cls with Quantization enabled."""

      def task(self):
        config = super(Wrapper, self)
        if quantize_on_the_fly:
          mode = quantization_hparams.QuantizationMode.MATERIALIZE
        else:
          mode = quantization_hparams.QuantizationMode.INFERENCE
        config.set_quant_mode(mode)
        task_p = config.task()

        quantization_type_str, _ = config.get_quant_configs()
        quantization_type = quantization_hparams.QuantizationType(
            quantization_type_str
        )
        assert num_bits == 8 or num_bits == 4
        quantize.set_quantization(
            task_p.model,
            layers.transformers.Transformer,
            quantization_type,
            mode=mode,
            num_bits=num_bits,
            linear_only=linear_only,
            use_symmetric=use_symmetric,
            quantize_embedding_softmax=quantize_embedding_softmax,
            transposed_embedding_softmax=transposed_embedding_softmax,
            quantize_ngrammer_embedding=quantize_ngrammer_embedding,
            dtype=dtype,
        )
        return task_p

    return Wrapper

  return decorator
