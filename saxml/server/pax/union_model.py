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
"""A single model with multiple service APIs."""

import abc
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import logging
import jax
from paxml import base_task
from paxml import checkpoints
from praxis import base_input
from praxis import base_model
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params

PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap


class UnionModelParams(
    servable_model_params.ServableModelParams, metaclass=abc.ABCMeta
):
  """Params for grouping multiple service interfaces.

  It specifies a list of children model params, which must be able to share the
  same model PAX model config (the `task()` method in BaseExperiment). The first
  model in `children()` will be used to create the PAX model and states.
  """

  overrides: Dict[str, Any] = {}

  @classmethod
  @abc.abstractmethod
  def children(cls) -> Sequence[servable_model_params.ServableModelParamsT]:
    """A list of servable params classes to be grouped."""

  @classmethod
  def serving_mesh_shape(cls) -> List[int]:
    return cls.children()[0].serving_mesh_shape()

  def methods(self) -> Dict[str, servable_model_params.ServableMethodParams]:
    raise NotImplementedError('should not be called')

  def create_model(self, primary_process_id: int) -> 'UnionModel':
    """Loads and initializes the model."""
    return UnionModel(self, primary_process_id, self.get_checkpoint_type())

  def input_for_model_init(self) -> NestedMap:
    raise NotImplementedError('should not be called')

  def task(self) -> pax_fiddle.Config[base_task.BaseTask]:
    raise NotImplementedError('should not be called')

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    raise NotImplementedError('should not be called')

  def apply_model_overrides(self, overrides: Dict[str, Any]) -> None:
    """Delays the model overrides until child creation."""
    self.overrides = overrides


class UnionModel(servable_model.ServableModel):
  """A model that implements multiple interfaces."""

  def __init__(
      self,
      model_config: servable_model_params.ServableModelParams,
      primary_process_id: int,
      ckpt_type: checkpoints.CheckpointType,
      test_mode: bool = False,
  ):
    super().__init__(model_config, primary_process_id, ckpt_type, test_mode)
    self._models: List[servable_model.ServableModel] = []

  def load_state(
      self,
      checkpoint_path: Optional[str],
      prng_key: PRNGKey,
      precompile: bool = True,
  ) -> Tuple[base_model.BaseModel, servable_model.ServableModelState]:
    union_config = self.model_config
    assert isinstance(union_config, UnionModelParams)
    children = union_config.children()
    if not children:
      raise ValueError('No children in UnionModelParams')
    self._models = []
    for child in children:
      child_inst = child()
      child_inst.apply_model_overrides(union_config.overrides)
      self._models.append(child_inst.create_model(self.primary_process_id))
    return self._models[0].load_state(checkpoint_path, prng_key, precompile)

  def load_methods(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
  ) -> None:
    for child_model in self._models:
      prng_key, my_key = jax.random.split(prng_key)
      child_model.load_methods(model, model_state, my_key)
      for k, m in child_model.methods.items():
        logging.info(
            'Initialized method %s from %s',
            k,
            child_model.model_config.__class__.__name__,
        )
        if k in self.methods:
          raise ValueError(f'Duplicate method name: {k}')
        self.methods[k] = m

  def unload(self) -> None:
    self.methods.clear()
    super().unload()
    for model in self._models:
      model.unload()
    del self._models
