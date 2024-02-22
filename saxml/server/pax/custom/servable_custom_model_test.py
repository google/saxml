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
"""Tests for servable_custom_model."""

from typing import List

from absl import logging
from absl.testing import absltest
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
from paxml import base_experiment
from paxml import checkpoints
from paxml import learners
from paxml import tasks_lib
from praxis import base_input
from praxis import base_layer
from praxis import base_model
from praxis import optimizers
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from praxis import schedules
from praxis import test_utils
from saxml.server.pax.custom import servable_custom_model

MN = servable_custom_model.CustomMethodName
NestedMap = py_utils.NestedMap
JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor


class TestModel(base_model.BaseModel):
  """Test model."""

  def test_method(self, input_batch: NestedMap) -> NestedMap:
    plus_one = input_batch.nums + self.get_variable(base_layer.PARAMS, 'b')
    return NestedMap(plus_one=plus_one)

  @nn.compact
  def __call__(self, input_batch: NestedMap) -> NestedMap:
    var_p = base_layer.WeightHParams(
        shape=input_batch.nums.shape,
        init=base_layer.WeightInit.Constant(1),
        dtype=jnp.int32,
        mesh_shape=[1, 1, 1],
        tensor_split_dims_mapping=(-1,),
    )
    self.create_variable('b', var_hparams=var_p, trainable=True)
    return self.test_method(input_batch)


class TestExpt(base_experiment.BaseExperiment):
  """Test experitment."""

  def _optimizer(self) -> pax_fiddle.Config[optimizers.BaseOptimizer]:
    return pax_fiddle.Config(
        optimizers.ShardedSgd,
        momentum=0.9,
        nesterov=True)

  def _learner(self) -> pax_fiddle.Config[learners.Learner]:
    lp = pax_fiddle.Config(learners.Learner)
    lp.optimizer = self._optimizer()
    lp.loss_name = 'loss'
    op = lp.optimizer
    op.lr_schedule = self._lr_schedule()
    op.learning_rate = 0.1
    return lp

  def _lr_schedule(self) -> pax_fiddle.Config[schedules.BaseSchedule]:
    lrs = [1, 0.1]
    boundaries = [10, 100]
    return pax_fiddle.Config(
        schedules.LinearRampupPiecewiseConstant,
        boundaries=boundaries,
        values=lrs
    )

  def datasets(self) -> List[pax_fiddle.Config[base_input.BaseInput]]:
    """Returns a list of dataset parameters."""
    pass

  def get_input_specs_provider_params(
      self,
  ) -> pax_fiddle.Config[base_input.BaseInputSpecsProvider]:
    pass

  def task(self) -> pax_fiddle.Config[tasks_lib.SingleTask]:
    """Returns the task parameters."""
    task_p = pax_fiddle.Config(tasks_lib.SingleTask, name='test_task')

    model_p = pax_fiddle.Config(TestModel, name='test_model')
    task_p.model = model_p
    task_p.model.mesh_axis_names = ('replica', 'data', 'model')
    task_p.model.ici_mesh_shape = (1, 1, 1)
    task_p.train.learner = self._learner()
    return task_p


class TestCustomCall:
  """Test custom call wrapper."""

  def get_fetch_output_fn(self) -> servable_custom_model.FetchOutputFn:
    """Gets fetch_output_fn."""

    def fetch_output(
        model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
    ) -> NestedJTensor:
      del model_fn_inputs
      outs, custom_state = model_fn_outputs[0]
      return outs.plus_one, custom_state

    return fetch_output

  def get_pre_process_fn(self) -> servable_custom_model.PreProcessingFn:
    """Gets pre_process_fn."""

    def pre_process(
        raw_inputs: List[str], method_state: List[List[int]]
    ) -> NestedNpTensor:
      nums = [int(raw_input) for raw_input in raw_inputs]
      nums = np.array(nums, dtype=np.int32)
      method_state.append([len(method_state)] * len(raw_inputs))
      return py_utils.NestedMap(nums=nums)

    return pre_process

  def get_post_process_fn(self) -> servable_custom_model.PostProcessingFn:
    """Gets post_process_fn."""

    def post_process(
        compute_outputs: NestedNpTensor, method_state: List[List[int]]
    ) -> List[str]:
      logging.info('compute_outputs: %s', compute_outputs)
      method_state.pop()
      return [str(compute_output) for compute_output in compute_outputs]

    return post_process

  def get_create_init_state_fn(self) -> servable_custom_model.CreateInitStateFn:
    """Gets create_init_state_fn."""

    def create_init_state_fn(
        method: servable_custom_model.ServableCustomMethod,
    ) -> list[int]:
      del method
      return []

    return create_init_state_fn

  def get_call_model_fn(self) -> servable_custom_model.CallModelFn:
    def call_model_fn(model, inputs, mdl_vars, prng_key, method_state):
      k1, k2 = prng_key
      outputs, updated_vars = model.apply(
          mdl_vars,
          inputs,
          method=model.test_method,
          mutable=[
              base_layer.NON_TRAINABLE,
              base_layer.DECODE_CACHE,
              base_layer.PREFIX_DECODE_CACHE,
          ],
          rngs={
              base_layer.PARAMS: k1,
              base_layer.RANDOM: k2,
          },
      )
      return (outputs, jnp.asarray(method_state[-1])), updated_vars

    return call_model_fn


class TestServableModel(
    TestExpt, servable_custom_model.ServableCustomModelParams
):
  """SPMD model with small params."""

  ICI_MESH_SHAPE = [1, 1, 1]
  BATCH_SIZE = 4

  @classmethod
  def serving_mesh_shape(cls):
    return cls.ICI_MESH_SHAPE

  def methods(self):
    custom_call_wrapper = TestCustomCall()
    custom_call_hparams = servable_custom_model.CustomCallHParams(
        batch_size=self.BATCH_SIZE,
        dummy_input_sample='0',
        model_fn_name='test_method',
        fetch_output_fn=custom_call_wrapper.get_fetch_output_fn(),
        pre_process_fn=custom_call_wrapper.get_pre_process_fn(),
        post_process_fn=custom_call_wrapper.get_post_process_fn(),
        create_init_state_fn=custom_call_wrapper.get_create_init_state_fn(),
        call_model_fn=custom_call_wrapper.get_call_model_fn(),
    )
    return {
        'test_call': custom_call_hparams,
        'test_another_call': custom_call_hparams,
    }

  def input_for_model_init(self):
    input_batch = py_utils.NestedMap(
        nums=jnp.zeros((self.BATCH_SIZE,), jnp.int32)
    )
    return input_batch


class ServableCustomModelTest(test_utils.TestCase):

  def setUp(self):
    super().setUp()
    self._prng_key = jax.random.PRNGKey(1234)

  def test_load_model_custom(self):
    model = servable_custom_model.ServableCustomModel(
        TestServableModel(),
        primary_process_id=0,
        ckpt_type=checkpoints.CheckpointType.GDA,
        test_mode=True,
    )
    model.load(checkpoint_path=None, prng_key=self._prng_key)
    custom_call_result = model.method('test_call').compute(['100', '12'])
    logging.info('custom_call_result: %s', custom_call_result)
    self.assertLen(custom_call_result, 2)

    custom_call_result = model.method('test_another_call').compute(
        ['10', '112']
    )
    logging.info('custom_call_result: %s', custom_call_result)
    self.assertLen(custom_call_result, 2)


if __name__ == '__main__':
  absltest.main()
