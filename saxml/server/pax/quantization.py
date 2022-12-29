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

from praxis import layers
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantize


# TODO(jianlijianli): Merge this with the decorator in pax.
# Ready-to-use quantization decorators for quantizing transformer.
def for_transformer(quantize_on_the_fly=True):
  """Find and quantize transformer.

  If there are transformers that shouldn't be quantized, use the quantize_*
  functions and manually/selectively quantize the model.

  If there are no transformers in the model, it's a no-op.

  Args:
    quantize_on_the_fly: If the model is to be quantized on the fly.
      - Defaults to True, and the input model is float, and quantization happen
        on the fly.
      - When set to False, the input model is already quantized.

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
        task_p = config.task()

        quantization_type_str, _ = config.get_quant_configs()
        quantization_type = quantization_hparams.QuantizationType(
            quantization_type_str)
        quantize.set_quantization(
            task_p.model,
            layers.transformers.Transformer,
            quantization_type,
            mode=mode)
        return task_p

    return Wrapper

  return decorator
