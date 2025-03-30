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
"""Gemma LM models."""

import abc
import io
from typing import Any, Dict, List, Union

from PIL import Image
from saxml.protobuf import multimodal_pb2
from saxml.server import servable_model_registry
from saxml.server.services import lm_service
from saxml.server.services import multimodal_service
from saxml.server.torch import servable_model
import torch

from gemma import config
from gemma import gemma3_model

DeviceTensors = servable_model.DeviceTensors
InputShapeInfo = servable_model.InputShapeInfo


class TextSampleMethod(servable_model.ServableMethod):
  """lm.generate method on a pytorch model."""

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  def pre_processing(self, raw_inputs: List[str]) -> List[str]:
    return raw_inputs  # list of prompts

  def post_processing(self, outputs: List[str]) -> List[str]:
    return outputs  # list of strings


class MultiModalSampleMethod(servable_model.ServableMethod):
  """mm.generate method on a pytorch model."""

  @classmethod
  def service_id(cls) -> str:
    return multimodal_service.SERVICE_ID

  @abc.abstractmethod
  def pre_processing(
      self, raw_inputs: List[multimodal_pb2.GenerateRequest]
  ) -> List[Any]:
    """Preprocesses an unpadded batch of data into host arrays."""

  def post_processing(self, outputs: List[str]) -> List[str]:
    return outputs  # list of strings


class GemmaTextSampleMethod(TextSampleMethod):
  """lm.generate method on a Gemma model."""

  max_decoded_steps: int = 128
  temperature: float = 1.0
  top_p: float = 0.95
  top_k: int = 64

  def device_compute(
      self, input_batch: DeviceTensors, padded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes the device computation."""
    device = torch.device(self._device)
    return self._model.generate(
        input_batch,
        device=device,
        output_len=self.max_decoded_steps,
        temperature=self.temperature,
        top_p=self.top_p,
        top_k=self.top_k,
    )


class GammaMMSampleMethod(MultiModalSampleMethod):
  """mm.generate method on a Gemma model."""

  max_decoded_steps: int = 128
  temperature: float = 1.0
  top_p: float = 0.95
  top_k: int = 64

  def device_compute(
      self, input_batch: DeviceTensors, padded_shape: InputShapeInfo
  ) -> DeviceTensors:
    """Executes the device computation."""
    device = torch.device(self._device)
    return self._model.generate(
        input_batch,
        device=device,
        output_len=self.max_decoded_steps,
        temperature=self.temperature,
        top_p=self.top_p,
        top_k=self.top_k,
    )

  def pre_processing(
      self, raw_inputs: List[multimodal_pb2.GenerateRequest]
  ) -> List[List[Union[str, Image.Image]]]:
    batched_inputs = []
    for raw_input in raw_inputs:
      input_itmes = []
      for item in raw_input.items:
        if item.WhichOneof('item') == 'text':
          input_itmes.append(item.text)
        elif item.WhichOneof('item') == 'image_bytes':
          image = Image.open(io.BytesIO(item.image_bytes))
          input_itmes.append(image)
      batched_inputs.append(input_itmes)
    return batched_inputs


def _get_gemma_config(
    model_variant: str, test_tokenizer_path: str
) -> config.GemmaConfig:
  """Returns the Gemma config for the given model variant."""
  if model_variant == 'test':
    test_config = config.get_config_for_4b('float32')
    test_config.max_position_embeddings = 256
    test_config.num_hidden_layers = 2
    test_config.num_attention_heads = 4
    test_config.num_key_value_heads = 1
    test_config.hidden_size = 16
    test_config.intermediate_size = 64
    test_config.head_dim = 4
    test_config.tokenizer = test_tokenizer_path
    test_config.final_logit_softcapping = 1.0
    test_config.attn_types = (config.AttentionType.GLOBAL,)
    test_config.sliding_window_size = 256
    test_config.rope_wave_length = {
        config.AttentionType.GLOBAL: 1_000_000,
    }
    return test_config
  return config.get_model_config(model_variant, dtype='float32')


class GemmaServableModel(servable_model.ServableModelParams):
  """A generic ServableModel for pytorch models."""

  MODEL_VARIANT: str = '4b'  # 4b, 12b, 27b_v3
  TEST_TOKENIZER_PATH: str = None
  DEVICE: str = 'cuda'  # 'cpu' or 'cuda'

  MAX_DECODED_STEPS: int = 128
  TEMPERATURE: float = 1.0
  TOP_P: float = 0.95
  TOP_K: int = 64

  def load(
      self,
      model_key: str,
      checkpoint_path: str,
      primary_process_id: int,
      prng_key: int,
  ) -> Any:
    model_config = _get_gemma_config(
        self.MODEL_VARIANT, self.TEST_TOKENIZER_PATH
    )
    torch.set_default_dtype(model_config.get_dtype())
    device = torch.device(self.DEVICE)
    model = gemma3_model.Gemma3ForMultimodalLM(model_config)
    if checkpoint_path != 'None':
      model.load_weights(checkpoint_path)
    else:
      with torch.no_grad():
        for param in model.parameters():
          param.data.fill_(1.0)
    model = model.to(device).eval()
    methods = self.methods()
    return servable_model.ServableModel(model, methods, device=self.DEVICE)

  def methods(self) -> Dict[str, servable_model.ServableMethodParams]:
    return {
        lm_service.LMMethodName.GENERATE: servable_model.ServableMethodParams(
            method_cls=GemmaTextSampleMethod,
            method_attrs={
                'max_decoded_steps': self.MAX_DECODED_STEPS,
                'temperature': self.TEMPERATURE,
                'top_p': self.TOP_P,
                'top_k': self.TOP_K,
            },
        ),
        multimodal_service.MultimodalMethodName.GENERATE: (
            servable_model.ServableMethodParams(
                method_cls=GammaMMSampleMethod,
                method_attrs={
                    'max_decoded_steps': self.MAX_DECODED_STEPS,
                    'temperature': self.TEMPERATURE,
                    'top_p': self.TOP_P,
                    'top_k': self.TOP_K,
                },
            )
        ),
    }


@servable_model_registry.register
class Gemma4b(GemmaServableModel):
  MODEL_VARIANT = '4b'
