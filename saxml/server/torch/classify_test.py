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
"""Tests for classification models."""
from absl.testing import absltest
from praxis import py_utils
from saxml.server.torch import classify
from saxml.server.torch import servable_model
import tensorflow as tf
from torchvision.models import resnet


class ClassifyMethodTest(absltest.TestCase):

  def test_load_model_and_classify(self):
    # Use a model with no checkpoint.
    model = resnet.resnet50()
    model = servable_model.ServableModel(
        model, classify.ResNetModel().methods(), "cpu"
    )
    fake_image = tf.image.encode_jpeg(tf.ones((224, 224, 3), dtype=tf.uint8))
    test_input = py_utils.NestedMap(image_bytes=fake_image)
    result = model.method(classify.VisionMethodName.CLASSIFY).compute(
        [test_input]
    )
    # Test properties of result.
    self.assertLen(result, 1)
    labels, scores = result[0]
    self.assertLen(labels, 10)
    self.assertLen(scores, 10)


if __name__ == "__main__":
  absltest.main()
