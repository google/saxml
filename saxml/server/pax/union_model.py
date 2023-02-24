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
from typing import Dict, List, Optional, Sequence, Tuple

from absl import logging
import jax
from paxml import base_task
from paxml import checkpoints
from praxis import base_input
from praxis import base_model
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

  def task(self) -> base_task.BaseTask.HParams:
    raise NotImplementedError('should not be called')

  def datasets(self) -> List[base_input.BaseInput.HParams]:
    raise NotImplementedError('should not be called')


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

  def load(
      self,
      checkpoint_path: Optional[str],
      prng_key: PRNGKey,
      precompile: bool = True,
  ) -> None:
    union_config = self.model_config
    assert isinstance(union_config, UnionModelParams)
    children = union_config.children()
    if not children:
      raise ValueError('No children in UnionModelParams')
    prng_key, init_key = jax.random.split(prng_key)
    # pytype: disable=not-instantiable
    self._models = [
        child().create_model(self.primary_process_id) for child in children
    ]
    # pytype: enable=not-instantiable
    prng_key, init_key = jax.random.split(prng_key)
    pax_model, model_state = self._models[0].load_state(
        checkpoint_path, init_key, precompile
    )
    for model in self._models:
      prng_key, my_key = jax.random.split(prng_key)
      model.load_methods(pax_model, model_state, my_key)
      for k, m in model.methods.items():
        logging.info(
            'Initialized method %s from %s',
            k,
            model.model_config.__class__.__name__,
        )
        if k in self.methods:
          raise ValueError(f'Duplicate method name: {k}')
        self.methods[k] = m

  def load_state(
      self,
      checkpoint_path: Optional[str],
      prng_key: PRNGKey,
      precompile: bool = True,
  ) -> Tuple[base_model.BaseModel, servable_model.ServableModelState]:
    raise NotImplementedError('should not be called')

  def load_methods(
      self,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      prng_key: PRNGKey,
  ) -> None:
    raise NotImplementedError('should not be called')

  def unload(self) -> None:
    self.methods.clear()
    super().unload()
    for model in self._models:
      model.unload()
    del self._models
