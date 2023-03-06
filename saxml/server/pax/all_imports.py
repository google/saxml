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
"""Imports the registered services and Cloud LM servable model params."""

# pylint: disable=unused-import,g-bad-import-order

# Import the servables.
from saxml.server.pax.lm import all_imports as _
from saxml.server.pax.vision import all_imports as _

# Specify the registry root.
from saxml.server import servable_model_registry

servable_model_registry.REGISTRY_ROOT = 'saxml.server.pax'
# pylint: enable=unused-import,g-bad-import-order
