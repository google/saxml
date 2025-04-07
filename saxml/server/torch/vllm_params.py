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
"""vLLM (https://github.com/vllm-project/vllm) integration with SAX."""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from absl import logging
import numpy as np
from saxml.server import servable_model
from saxml.server import servable_model_params
from saxml.server import servable_model_registry
from saxml.server import utils
from saxml.server.services import lm_service
import vllm

DeviceTensors = Any  # More flexible for vLLM
HostTensors = Any
InputShapeInfo = servable_model.InputShapeInfo


@dataclasses.dataclass
class ServableMethodParams(servable_model_params.ServableMethodParams):
  """A base param class for a torch method.

  Attributes:
    method_cls: the class name to construct the ServableMethod instance.
    method_attrs: extra attributes to assign to the ServableMethod.
  """

  method_cls: Type[servable_model.ServableMethod]
  method_attrs: Dict[str, Any] = dataclasses.field(default_factory=dict)

  batch_size: Union[int, List[int]] = 1
  max_live_batches: int = 4
  extra_inputs: Optional[Dict[str, float]] = None
  extra_inputs_dtypes: Optional[Dict[str, np.dtype]] = None

  def get_batch_size(self) -> Union[int, List[int]]:
    return self.batch_size

  def get_max_live_batches(self) -> int:
    return self.max_live_batches

  def get_default_extra_inputs(self) -> Optional[Dict[str, float]]:
    return self.extra_inputs

  def get_extra_inputs_dtypes(self) -> Optional[Dict[str, np.dtype]]:
    return self.extra_inputs_dtypes

  def get_batching_wait_secs(self) -> Optional[float]:
    return None


# Base class might be simplified as vLLM handles more internally
class TextSampleMethod(servable_model.ServableMethod):
  """Base class for lm.generate method using a vLLM backend."""

  def __init__(
      self,
      llm: Any,  # Pass the vLLM llm instance
      method_params: ServableMethodParams,
      device: str,  # Keep device info if needed, though vLLM manages it
  ):
    """Constructor.

    Args:
        llm: The initialized vLLM LLM instance.
        method_params: Parameters of this method, including default sampling
          params.
        device: The primary device (less critical for vLLM usage here).
    """
    super().__init__(method_params)
    self._model = llm
    self._device = device  # Store if needed, maybe for compatibility checks
    self._sampling_params = vllm.SamplingParams(**method_params.method_attrs)

  @classmethod
  def service_id(cls) -> str:
    return lm_service.SERVICE_ID

  def pre_processing(self, raw_inputs: List[str]) -> List[str]:
    """Input is already a list of prompts."""
    return raw_inputs

  def input_to_device(
      self,
      one_core_inputs: List[str],  # Input is list of strings
      unpadded_shape: InputShapeInfo,  # Shape info might be less relevant
      padded_shape: InputShapeInfo,
  ) -> List[str]:  # Output is just the list of strings for vLLM
    """vLLM handles internal device placement and batching."""
    # No explicit tensor creation or device movement needed here
    return one_core_inputs

  def device_compute(
      self,
      input_batch: List[str],
      padded_shape: InputShapeInfo,
  ) -> List[Tuple[List[str], List[float]]]:
    """Executes the generation using the vLLM engine."""

    # vLLM handles batching, inference, and decoding internally
    request_outputs: List[vllm.RequestOutput] = self._model.generate(
        prompts=input_batch,
        sampling_params=self._sampling_params,
    )

    # Extract the generated text from each output
    responses = []
    for output in request_outputs:
      generated_texts = []
      scores = []
      for generated_output in output.outputs:
        score = 0.0
        if generated_output.cumulative_logprob is not None:
          score = output.outputs[0].cumulative_logprob
        generated_texts.append(generated_output.text)
        scores.append(score)
      else:
        # Handle cases where generation might fail or produce no output
        generated_texts.append("")
        scores.append(0.0)
      responses.append((generated_texts, scores))
    return responses

  def output_to_host(
      self,
      output_tensors: List[Tuple[List[str], List[float]]],
      unpadded_batch_size: int,  # output_tensors is actually List[str]
  ) -> List[Tuple[List[str], List[float]]]:
    """vLLM generate already returns host data (strings, scores)."""
    # No explicit device-to-host transfer needed
    return output_tensors

  def post_processing(self,
                      outputs: List[Tuple[List[str], List[float]]]
                      ) -> List[Tuple[List[str], List[float]]]:
    """Outputs are already the final generated strings."""
    return outputs

  def unload(self) -> None:
    """Clears references held by this model."""
    del self._model

  def remove_batch_padding(
      self, host_tensors: HostTensors, unpadded_batch_size: int
  ) -> HostTensors:
    """Removes batch padding."""
    return host_tensors

  def update_extra_inputs(
      self,
      input_batch: HostTensors,
      batch_size: int,
      extra_inputs: Optional[List[Any]] = None,
  ) -> HostTensors:
    if extra_inputs and extra_inputs[0]:
      raise ValueError(
          f"extra_inputs is not supported by this method! Got {extra_inputs}."
      )
    return input_batch

  @property
  def streamable_output(self) -> bool:
    return False

  def device_compute_with_dummy_data(
      self, padded_shape: InputShapeInfo
  ) -> DeviceTensors:
    raise NotImplementedError


# --- vLLM SAX Model Wrapper ---
class VLLMSaxServableModel(servable_model.ServableModel):
  """A ServableModel specifically for vLLM instances."""

  def __init__(
      self,
      llm: Any,  # Pass the llm instance
      method_params: Dict[str, ServableMethodParams],
      device: str,
  ):
    super().__init__()  # Call grand-parent init if necessary
    self._model = llm
    self._device = device
    # vLLM model is already in 'eval' mode conceptually

    for name, params in method_params.items():
      # Instantiate the method, passing the vLLM engine
      method_instance = params.method_cls(
          llm=self._model,  # Pass the engine here
          method_params=params,
          device=self._device,
      )
      self.add_method(name, method_instance)

  def supports_dummy_compute_on_primary(self) -> bool:
    return False


# --- Servable Model Parameter Definition ---
class VLLMServableModel(servable_model_params.ServableModelParams):
  """ServableModelParams for models served via vLLM."""

  # --- Configuration Flags ---
  # The name or path of a HuggingFace Transformers model.
  HF_MODEL_PATH: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
  DEVICE: str = "cuda"  # vLLM primarily targets CUDA

  TENSOR_PARALLEL_SIZE: int = 1  # Default for single SAX process managing GPUs
  GPU_MEMORY_UTILIZATION: float = 0.90  # Default vLLM memory usage
  DTYPE: str = "auto"  # Or "bfloat16", "float16"

  # Default vLLM Sampling Parameters (passed to ServableMethodParams)
  TEMPERATURE: float = 0.7
  TOP_P: float = 1.0  # vLLM default is 1.0
  TOP_K: int = -1  # vLLM default is -1 (disabled)
  MAX_TOKENS: int = 1024  # Max *new* tokens to generate

  # --- SAX Interface Methods ---

  def load(
      self,
      model_key: str,
      hf_model_path: str,  # HF identifier or local path
      primary_process_id: int,
      prng_key: int,
  ) -> VLLMSaxServableModel:  # Return our custom wrapper
    """Loads the model using the vLLM engine."""

    # Use provided checkpoint_path if valid, otherwise use class default
    model_load_path = (
        hf_model_path
        if hf_model_path and hf_model_path != "None"
        else self.HF_MODEL_PATH
    )

    logging.info(
        "Loading model via vLLM from: %s on device: %s",
        model_load_path,
        self.DEVICE,
    )
    logging.info(
        "vLLM Config: TP=%s, GPU Mem=%s, dtype=%s",
        self.TENSOR_PARALLEL_SIZE,
        self.GPU_MEMORY_UTILIZATION,
        self.DTYPE,
    )

    try:
      llm = vllm.LLM(
          model=model_load_path,
          tensor_parallel_size=self.TENSOR_PARALLEL_SIZE,
          gpu_memory_utilization=self.GPU_MEMORY_UTILIZATION,
          dtype=self.DTYPE,
      )
    except Exception as e:
      logging.exception(
          "Error initializing vLLM engine for %s: %s", model_load_path, e)
      raise

    method_params = self.methods()

    # Instantiate the custom ServableModel wrapper
    return VLLMSaxServableModel(
        llm=llm,
        method_params=method_params,
        device=self.DEVICE,
    )

  def methods(self) -> Dict[str, ServableMethodParams]:
    """Defines the servable methods for this model."""
    # Define default sampling parameters for vLLM via method_attrs
    vllm_sampling_attrs = {
        "temperature": self.TEMPERATURE,
        "top_p": self.TOP_P,
        "top_k": self.TOP_K,
        "max_tokens": self.MAX_TOKENS,
        "stop": [],  # Example: default stop sequences if any
    }

    return {
        lm_service.LMMethodName.GENERATE: ServableMethodParams(
            method_cls=TextSampleMethod,  # Use the vLLM method class
            method_attrs=vllm_sampling_attrs,
            batch_size=1,
            max_live_batches=4,  # SAX queue control
        ),
    }

  @classmethod
  def get_supported_device_mesh(
      cls,
  ) -> Tuple[utils.Status, Optional[np.ndarray]]:
    """vLLM primarily supports CUDA GPUs."""
    return utils.ok(), None

  @classmethod
  def check_serving_platform(cls) -> utils.Status:
    return utils.ok()


# --- Registration ---
@servable_model_registry.register
class QWen1P5B(VLLMServableModel):
  HF_MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@servable_model_registry.register
class Gemma4B(VLLMServableModel):
  HF_MODEL_PATH = "google/gemma-3-4b-it"
