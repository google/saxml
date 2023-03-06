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
"""Tests for validate."""

from absl.testing import absltest
import grpc
from saxml.protobuf import test_pb2
from saxml.server import validate


class ValidateTest(absltest.TestCase):

  def test_none_req_and_empty_extra_inputs(self):
    req = None
    extra_inputs = None
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_none_req_and_defined_extra_inputs(self):
    req = None
    extra_inputs = {'a': 0.1, 'b': 0.2}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_no_extra_input_req_and_empty_extra_inputs(self):
    req = test_pb2.TestRequest()
    extra_inputs = None
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_no_extra_input_req_and_defined_extra_inputs(self):
    req = test_pb2.TestRequest()
    extra_inputs = {'a': 0.1, 'b': 0.2}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_no_defined_extra_input_req_and_defined_extra_inputs(self):
    req = test_pb2.TestRequestWithExtraInput()
    extra_inputs = {'a': 0.1, 'b': 0.2}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_no_defined_extra_input_req_and_empty_extra_inputs(self):
    req = test_pb2.TestRequestWithExtraInput()
    extra_inputs = None
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_defined_extra_input_req_and_empty_extra_inputs(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.items['a'] = 0.1
    extra_inputs = None
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertEqual(validate_status.code, grpc.StatusCode.INVALID_ARGUMENT)

  def test_defined_extra_input_req_and_extra_inputs_mismatch(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.items['a'] = 0.1
    extra_inputs = {'b': 0.3}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertEqual(validate_status.code, grpc.StatusCode.INVALID_ARGUMENT)

  def test_defined_extra_input_req_and_extra_inputs_partial_mismatch(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.items['a'] = 0.1
    req.extra_inputs.items['c'] = 0.2
    extra_inputs = {'a': 0.6, 'b': 0.3}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertEqual(validate_status.code, grpc.StatusCode.INVALID_ARGUMENT)

  def test_defined_extra_input_req_and_extra_inputs_partial_match(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.items['a'] = 0.1
    extra_inputs = {'a': 0.6, 'b': 0.3}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_defined_extra_input_req_and_extra_inputs_match(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.items['a'] = 0.1
    req.extra_inputs.items['b'] = 0.5
    extra_inputs = {'a': 0.6, 'b': 0.3}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_defined_extra_input_req_and_extra_inputs_tensor_match(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.tensors['a'].values.extend([0.1, 0.2])
    extra_inputs = {'a': [0.3, 0.4]}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertTrue(validate_status.ok())

  def test_defined_extra_input_req_and_extra_inputs_tensor_mismatch(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.tensors['a'].values.extend([0.1, 0.2])
    extra_inputs = {'b': [0.1, 0.2]}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertEqual(validate_status.code, grpc.StatusCode.INVALID_ARGUMENT)

  def test_defined_extra_input_req_and_extra_inputs_tensor_size_mismatch(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.tensors['a'].values.extend([0.1, 0.2, 0.3])
    extra_inputs = {'b': [0.1, 0.2]}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertEqual(validate_status.code, grpc.StatusCode.INVALID_ARGUMENT)

  def test_extra_inputs_items_values_key_collision(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.tensors['a'].values.extend([0.1, 0.2])
    req.extra_inputs.items['a'] = 0.3
    extra_inputs = {'a': [0.1, 0.2]}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertEqual(validate_status.code, grpc.StatusCode.INVALID_ARGUMENT)

  def test_defined_extra_input_non_list_and_extra_inputs_tensor(self):
    req = test_pb2.TestRequestWithExtraInput()
    req.extra_inputs.tensors['a'].values.extend([0.1, 0.2])
    extra_inputs = {'a': 0.1}
    validate_status = validate.ValidateRequestForExtraInputs(req, extra_inputs)
    self.assertEqual(validate_status.code, grpc.StatusCode.INVALID_ARGUMENT)


if __name__ == '__main__':
  absltest.main()
