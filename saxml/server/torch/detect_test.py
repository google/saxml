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
"""Tests for detection models."""
from absl.testing import absltest
from detectron2 import model_zoo
from detectron2 import modeling
from praxis import py_utils
from saxml.server.torch import detect
from saxml.server.torch import servable_model
import tensorflow as tf


class DetectMethodTest(absltest.TestCase):

  def test_load_model_and_detect(self):
    cfg = model_zoo.get_config(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml"
    )
    cfg.MODEL.DEVICE = "cpu"
    model = modeling.build_model(cfg)
    model = servable_model.ServableModel(
        model, detect.Detectron2Model().methods(), "cpu"
    )
    fake_image = tf.image.encode_jpeg(tf.ones((224, 224, 3), dtype=tf.uint8))
    test_input = py_utils.NestedMap(image_bytes=fake_image)
    result = model.method(detect.VisionMethodName.DETECT).compute([test_input])
    # Test properties of result.
    self.assertLen(result, 1)
    self.assertLen(result[0], 3)


if __name__ == "__main__":
  absltest.main()
