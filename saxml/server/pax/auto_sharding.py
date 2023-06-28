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
"""Auto sharding: automatically and smartly choose model shardings.

The functions in this file enables XLA auto sharding pass for a given SAX model.

Example usage:

@auto_sharding.enable()
Class ModelAutoSharding(Model):
  # Model definitions

When compiling ModelAutoSharding, the auto sharding pass in XLA is invoked which
computes optimal shardings for this model. With auto sharding enabled, no SPMD
sharding annotation is needed when building this model. Auto sharding can be
used with or without existing sharding annotations. To let auto sharding
re-compute all shardings, add
--xla_tpu_auto_spmd_keep_all_user_shardings=false to the launch command.
"""

import functools


def enable():
  """Enable auto sharding."""

  def decorator(cls):
    @functools.wraps(cls, updated=())
    class Wrapper(cls):
      """Wrapper class for cls with auto sharding."""

      @property
      def enable_auto_sharding(self) -> bool:
        return True

    return Wrapper

  return decorator


def disable():
  """Disable auto sharding."""

  def decorator(cls):
    @functools.wraps(cls, updated=())
    class Wrapper(cls):
      """Wrapper class for cls with auto sharding."""

      @property
      def enable_auto_sharding(self) -> bool:
        return False

    return Wrapper

  return decorator
