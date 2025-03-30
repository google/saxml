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
import os

from absl.testing import absltest
from saxml.protobuf import multimodal_pb2
from saxml.server.services import lm_service
from saxml.server.services import multimodal_service
from saxml.server.torch import lm
import tensorflow as tf


def test_src_dir_path(relative_path: str) -> str:
  path = os.path.join(
      absltest.get_default_test_srcdir(),
      '__main__/saxml',
      relative_path,
  )
  return path


class GemmaTestConfig(lm.GemmaServableModel):
  MAX_DECODED_STEPS = 4
  DEVICE = 'cpu'
  MODEL_VARIANT = 'test'
  TEST_TOKENIZER_PATH = test_src_dir_path(
      'server/torch/testdata/spiece.model.1000.model'
  )


class LmTest(absltest.TestCase):

  def testGemma(self):
    model = GemmaTestConfig()
    llm = model.load('/sax/test/gemma', 'None', 0, 32)
    test_input = 'hello world\n'
    result = llm.method(lm_service.LMMethodName.GENERATE).compute([test_input])
    self.assertLen(result, 1)
    print('result: %s' % result[0])
    # Test multimodal method.
    image_path = test_src_dir_path('server/torch/testdata/test_image.jpg')
    with tf.io.gfile.GFile(image_path, 'rb') as f:
      image_bytes = f.read()
    items = [
        multimodal_pb2.DataItem(text='hello world\n'),
        multimodal_pb2.DataItem(image_bytes=image_bytes),
    ]
    test_input = multimodal_pb2.GenerateRequest(items=items)
    result = llm.method(
        multimodal_service.MultimodalMethodName.GENERATE
    ).compute([test_input])
    self.assertLen(result, 1)
    print('result: %s' % result[0])


if __name__ == '__main__':
  absltest.main()
