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
"""Serving configs ImageNet ResNet models."""

import numpy as np
from paxml.tasks.vision.params import imagenet_resnets
from praxis import py_utils
from saxml.server import servable_model_registry
from saxml.server.pax.vision import imagenet_metadata
from saxml.server.pax.vision import servable_vision_model


@servable_model_registry.register
class ImageNetResNet50(
    imagenet_resnets.ResNet50Pjit, servable_vision_model.VisionModelParams
):
  """ImageNet ResNet50 base model for classification tasks."""

  IMAGE_SIZE = 224
  BATCH_SIZE = 1
  TOP_K = 5

  @classmethod
  def serving_mesh_shape(cls):
    return [
        [1, 1, 1],  # A single device or just on CPU
        [1, 1, 4],  # 4 accelerators.
        [1, 1, 8],  # 8 accelerators.
    ]

  def serving_dataset(self):
    """Dataset used to define serving preprocessing by the model."""
    return self._dataset_test()

  def id_to_string(self, idx: int) -> str:
    return imagenet_metadata.CLASS_LIST[idx]

  def classify(self):
    return servable_vision_model.ClassifyHParams(
        top_k=self.TOP_K, batch_size=self.BATCH_SIZE
    )

  def input_for_model_init(self):
    # Batch-2 is sufficient for model init. Imagenet num_classes=1000
    batch_size, image_size, num_classes = self.BATCH_SIZE, self.IMAGE_SIZE, 1000
    img_shape = (batch_size, image_size, image_size, 3)
    label_shape = (batch_size, num_classes)

    img_inputs = np.ones(img_shape, dtype=np.float32)
    label_inputs = np.random.uniform(low=0.0, high=1.0, size=label_shape)
    label_inputs = label_inputs / np.sum(label_inputs, axis=1, keepdims=True)
    mdl_inputs = py_utils.NestedMap(image=img_inputs, label_probs=label_inputs)
    return mdl_inputs
