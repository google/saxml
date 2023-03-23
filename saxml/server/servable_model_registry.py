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
"""Global registry for servable models (experiments).

Servable models are subclasses of servable_model_params.ServableModelParams.
Each servable model is named with a key, which is module_path.class_name, with
an alias which is the relative path to REGISTRY_ROOT.

E.g., in /path/to/root/lm/params/foo.py

import ....servable_model_registry

@servable_model_registry.register
class FooExpABC(servable_lm_model.ServableLMModelParams):
  ...

Then this servable model will be named `path.to.root.lm.params.foo.FooExpABC`;
if
  REGISTRY_ROOT == 'path.to.root.lm.params',
it will have an alias `foo.FooExpABC`.
"""

from typing import List, Mapping, Optional, Pattern

from saxml.server.servable_model_params import ServableModelParams
from saxml.server.servable_model_params import ServableModelParamsT

# Root prefix path for the modules of servable params. Can be overwritten before
# server starts.
REGISTRY_ROOT = None
# Global registry. name -> params
_registry = {}

# A regex to filter (full match) models by their names.
MODEL_FILTER_REGEX: Optional[Pattern[str]] = None


def get_aliases(full_model_name: str) -> List[str]:
  """Gets a list of aliases for a model name."""
  if not REGISTRY_ROOT:
    return []
  paths = []
  prefix = REGISTRY_ROOT + '.'
  if full_model_name.startswith(prefix):
    full_model_name = full_model_name[len(prefix) :]
    paths.append(full_model_name)
  return paths


def full_registration_name(model_class) -> str:
  """Returns the full registration name for a class."""
  if issubclass(model_class, ServableModelParams):
    custom_name = model_class.sax_registration_name()
    if custom_name is not None:
      return custom_name
  return model_class.__module__ + '.' + model_class.__name__


def register(model_class):
  """Registers a model."""
  full_name = full_registration_name(model_class)
  _registry[full_name] = model_class
  return model_class


def _get_full_model_name_from_alias(alias: str) -> Optional[str]:
  if not REGISTRY_ROOT:
    return None
  maybe_full_name = REGISTRY_ROOT + '.' + alias
  if maybe_full_name in _registry:
    return maybe_full_name
  return None


def get(model_name: str) -> Optional[ServableModelParamsT]:
  """Returns a model with the name."""
  if (
      MODEL_FILTER_REGEX is not None
      and MODEL_FILTER_REGEX.fullmatch(model_name) is None
  ):
    # Filtered.
    return None
  maybe_params = _registry.get(model_name)
  if maybe_params:
    return maybe_params
  maybe_full_name = _get_full_model_name_from_alias(model_name)
  if maybe_full_name:
    return _registry.get(maybe_full_name)
  return None


def get_all() -> Mapping[str, ServableModelParamsT]:
  """Returns all models. Full model names only."""
  if MODEL_FILTER_REGEX is None:
    return _registry
  models = {}
  for k, m in _registry.items():
    if MODEL_FILTER_REGEX.fullmatch(k) is not None:
      models[k] = m
  return models
