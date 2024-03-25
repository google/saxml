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
"""Base classes for servable model and method config classes."""

import abc
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from absl import logging
import numpy as np
from saxml.server import utils


ExtraInputs = Dict[str, Union[float, List[float], str]]  # from common.proto


class ServableMethodParams(abc.ABC):
  """A base class for a servable method config class."""

  @abc.abstractmethod
  def get_batch_size(self) -> Union[int, List[int]]:
    """Returns a fixed batch size or a list of allowed batch sizes."""
    # TODO(changlan): Refactor to always return a list.

  @abc.abstractmethod
  def get_max_live_batches(self) -> int:
    """Returns the (approximate) maximum number of in-flight batches.

    The batching queue keeps roughly (max batch size) * (max live batches)
    requests in-flight. Additional requests are dropped on arrival with a
    "resource exhausted" error.
    """

  @abc.abstractmethod
  def get_batching_wait_secs(self) -> Optional[float]:
    """Returns an optional batch formation time window.

    When the batching queue is drained to form a batch of requests, this value
    indicates the maximum time interval to wait for, from when the first request
    is dequeued to when the batch is formed. If this is None, a batch is formed
    immediately when the batching queue is drained empty.
    """

  @abc.abstractmethod
  def get_extra_inputs_dtypes(self) -> Optional[Dict[str, np.dtype]]:
    """Returns the (device) dtypes for optional extra inputs.

    Extra inputs are per-request key-value pairs. They can be used to override
    default model params on a per-request basis. This function returns the
    dtype each extra input value should be cast to, in the form of a
    key-to-dtype dictionary. If the dtype for a key is not defined in the return
    value, it defaults to np.float32.
    """

  @abc.abstractmethod
  def get_default_extra_inputs(self) -> Optional[ExtraInputs]:
    """Returns the default (host) values for optional extra inputs.

    Extra inputs are per-request key-value pairs. They can be used to override
    default model params on a per-request basis. This function returns the
    default value for each extra input key, which is used when the key is not
    found in the per-request "extra_inputs" proto field.
    """


class ServableModelParams(abc.ABC):
  """A base class for a servable model config class."""

  @classmethod
  @abc.abstractmethod
  def get_supported_device_mesh(
      cls,
  ) -> Tuple[utils.Status, Optional[np.ndarray]]:
    """Returns OK and a supported device mesh, or non-OK and None.

    A model config can declare a single device mesh shape or a list of device
    mesh shapes it supports. This function returns a JAX device mesh (i.e., the
    jax.sharding.Mesh.devices attribute) successfully created on the current
    platform using one of the supported device mesh shapes.
    """

  @classmethod
  def check_serving_platform(cls) -> utils.Status:
    """Returns OK if the current platform supports this model."""
    status, _ = cls.get_supported_device_mesh()
    return status

  @abc.abstractmethod
  def load(
      self,
      model_key: str,
      checkpoint_path: str,
      primary_process_id: int,
      prng_key: int,
  ) -> Any:
    """Loads a checkpoint and returns a ServableModel instance."""
    # TODO(jiawenhao): Break ServableMethodParams into its own file so we can
    # type the return value as ServableModel.

  @abc.abstractmethod
  def methods(self) -> Dict[str, ServableMethodParams]:
    """Returns a dictionary of supported method names and their configs."""

  @classmethod
  def sax_registration_name(cls) -> Optional[str]:
    """Returns an optional custom registration name for the model."""
    return None

  def apply_model_overrides(self, overrides: Dict[str, Any]) -> None:
    """Applies model config overrides received from Publish.

    The default handling of overrides is as follows:

      - Warning if the provided key is not found on this model config.
      - Fail if the types of the original and provided values mismatch.
      - Replace the original value with the provided value.

    This method may be overridden by subclasses for more customized behavior.

    Args:
        overrides: Model config key-value pairs supplied by the Publish command.
    """
    for k, v_raw in overrides.items():
      if not hasattr(self, k):
        logging.warning("Can't override %s because it's not set on %s", k, self)
        continue
      try:
        v = json.loads(v_raw)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('Not a valid json value: %s %s', v_raw, e)
        continue
      cur_v = getattr(self, k)
      if v is not None and cur_v is not None and type(v) != type(cur_v):  # pylint: disable=unidiomatic-typecheck
        raise ValueError(
            'Mismatched type of override: original: %s; override: %s'
            % (cur_v, v)
        )
      setattr(self, k, v)
      logging.info('Set override %s to %s on %s', k, v, self)
