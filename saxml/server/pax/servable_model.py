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
"""Wraps a model with service APIs."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from absl import logging
from etils import epath
import jax
from jax import numpy as jnp
from jax.experimental import pjit
import numpy as np
from paxml import checkpoints
from paxml import tasks_lib
from paxml import trainer_lib
from praxis import base_layer
from praxis import base_model
from praxis import py_utils
from praxis import pytypes
from praxis.layers.quantization import quantization_hparams
from praxis.layers.quantization import quantize
from saxml.server.jax import servable_model
from saxml.server.pax import branch_selection
from saxml.server.pax import servable_model_params

# pytype: disable=attribute-error

ServableModelState = servable_model.ServableModelState
StepCounter = servable_model.StepCounter
HostTensors = servable_model.HostTensors
InputShapeInfo = servable_model.InputShapeInfo
MethodInputInfo = servable_model.MethodInputInfo
ShapesAndDtypes = servable_model.ShapesAndDtypes
CheckpointType = checkpoints.CheckpointType
JTensor = pytypes.JTensor
NpTensor = pytypes.NpTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap
NestedPartitionSpec = pytypes.NestedPartitionSpec

CKPT_MODULE = checkpoints


class ServableMethod(servable_model.ServableMethod):
  """Base class for method implementation and its pre- and post-processing.

  This class initializes the method based on a model function specified by
  `model_fn_name`, and separates device computation from pre- and
  post-processing so that a service can pipeline them. It also provides
  device-only computation with dummy data for secondary hosts in a
  multi-Jax-client setup.

  In addition to providing init arguments, subclasses need to override methods:
    - fetch_output()
    - pre_processing()
    - post_processing()
  """

  def __init__(
      self,
      model: base_model.BaseModel,
      model_fn_name: str,
      model_state: ServableModelState,
      method_params: servable_model_params.ServableMethodParams,
      prng_key: PRNGKey,
      dummy_input_sample: Any,
      exportable: bool = False,
      load: bool = True,
  ):
    """Initializes the method.

    Args:
      model: A PAX model.
      model_fn_name: The method name of `model`.
      model_state: Model state created by ServableModel.
      method_params: The params for the method.
      prng_key: PRNG key used for this method.
      dummy_input_sample: Dummy inputs for an example in the batch. It is used
        for 1) initializing the method (pre-compilation with input shapes) and
        2) feeding devices in secondary hosts during actual computation, which
        will be discarded by device computation but GSPMD still requires them to
        be provided. Because of 1), make sure the dummy data will not cause
        problems like infinite loop in the device computation (but silent
        problems like NaNs are fine).
      exportable: whether this method is exportable to a SavedModel.
      load: Whether to load this method during this __init__ call.
    """
    super().__init__(method_params, model_state, prng_key, dummy_input_sample)
    self._model = model
    self._model_fn_name = model_fn_name
    self._dummy_bucket_key = -1
    self._exportable = exportable
    self._bucket_keys = method_params.bucket_keys
    self._method_params = method_params

    # TODO(b/261075587): remove conditional based input prefix bucketization.
    self._branch_selector = branch_selection.BranchSelector(
        keys=[self._dummy_bucket_key]
    )
    if load:
      self.load()

  @property
  def pax_model(self) -> base_model.BaseModel:
    return self._model

  @property
  def exportable(self) -> bool:
    return self._exportable

  @property
  def method_params(self) -> servable_model_params.ServableMethodParams:
    return self._method_params

  @property
  def streamable(self) -> bool:
    return False

  def get_unpadded_branch_key(self, inputs: NestedNpTensor) -> int:
    """Returns the bucket key (before padding) used for inputs."""
    del inputs
    return self._dummy_bucket_key

  def _assign_branch_index(self, inputs: NestedNpTensor) -> Optional[NpTensor]:
    """Assigns branch index to the inputs."""
    # Do nothing by default.
    if not self._branch_selector.has_multiple_branches():
      return None
    bucket_key = self.get_unpadded_branch_key(inputs)
    # Supports multiple batch size in base class.
    branch_index = self._branch_selector.get_branch_index(bucket_key)
    return np.asarray(branch_index, dtype=np.int32)

  def get_branch_inputs(
      self, inputs: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Returns the inputs for a branch key."""
    del branch_key
    return inputs

  def post_process_branch_outputs(
      self, outputs: NestedJTensor, branch_key: int
  ) -> NestedJTensor:
    """Post processes branch outputs."""
    del branch_key
    return outputs

  def get_nonbatch_inputs(
      self, one_core_inputs: NestedNpTensor
  ) -> NestedNpTensor:
    branch_index = self._assign_branch_index(one_core_inputs)
    if branch_index is not None:
      return np.array(branch_index, dtype=np.int32)
    return ()

  def add_extra_inputs(
      self,
      input_batch: NestedNpTensor,
      extra_input_tensors: Dict[str, np.ndarray],
  ) -> NestedNpTensor:
    assert isinstance(
        input_batch, (NestedMap, dict)
    ), 'extra_inputs unsupported on non-dict'
    for k, v in extra_input_tensors.items():
      if isinstance(input_batch, NestedMap):
        input_batch.Set(k, v)
      else:
        input_batch[k] = v
    return input_batch

  def call_model_function(
      self, inputs: NestedJTensor, mdl_vars: NestedJTensor, prng_key: PRNGKey
  ) -> NestedJTensor:
    k1, k2 = prng_key
    outputs = self._model.apply(
        mdl_vars,
        inputs,
        method=getattr(self._model, self._model_fn_name),
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
    return outputs

  def jax_func(
      self,
      mdl_vars: NestedJTensor,
      prng_key: PRNGKey,
      batched_inputs: NestedJTensor,
      non_batched_inputs: NestedJTensor,
  ) -> NestedJTensor:
    if self._model.fprop_dtype == jnp.bfloat16:
      # Convert float inputs/vars if fprop dtype is bfloat16.
      batched_inputs, mdl_vars = jax.tree_map(
          (lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x),
          (batched_inputs, mdl_vars),
      )

    context_p = base_layer.JaxContext.HParams(do_eval=True)
    k1, k2 = jax.random.split(prng_key)
    with base_layer.JaxContext.new_context(hparams=context_p):

      def _model_fn(inputs):
        outputs = self.call_model_function(inputs, mdl_vars, [k1, k2])  # pytype: disable=wrong-arg-types  # jax-ndarray
        # DECODE_CACHE are not read by caller. But they can be large. Tell XLA
        # to remove it from output. Note MLP decoder don't have DECODE_CACHE.
        updated_vars = outputs[1]
        if base_layer.DECODE_CACHE in updated_vars:
          del updated_vars[base_layer.DECODE_CACHE]
        if base_layer.PREFIX_DECODE_CACHE in updated_vars:
          del updated_vars[base_layer.PREFIX_DECODE_CACHE]
        return outputs

      if isinstance(non_batched_inputs, tuple) and not non_batched_inputs:
        outputs = _model_fn(batched_inputs)
      else:

        def _build_branch_model_fn(branch_key):
          """Gets model_fn for each branch."""

          def _branch_model_fn(inputs):
            branch_inputs = self.get_branch_inputs(inputs, branch_key)
            branch_outputs = _model_fn(branch_inputs)
            return self.post_process_branch_outputs(branch_outputs, branch_key)

          return _branch_model_fn

        branch_fns = [
            _build_branch_model_fn(branch_key)
            for branch_key in self._branch_selector.branch_keys
        ]
        branch_index = non_batched_inputs
        outputs = jax.lax.switch(branch_index, branch_fns, batched_inputs)

      if (
          self.method_params.cast_bfloat16_outputs
          and self._model.fprop_dtype == jnp.bfloat16
      ):
        # Convert bfloat16 back to float32.
        def maybe_to_float32(x):
          if x.dtype == jnp.bfloat16:
            return x.astype(jnp.float32)
          return x

        outputs = jax.tree_map(maybe_to_float32, outputs)
      return self.fetch_output(outputs, batched_inputs)

  def unload(self) -> None:
    super().unload()
    self._model = None

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    raise NotImplementedError('fetch_output not implemented')

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    raise NotImplementedError('pre_processing not implemented')

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    raise NotImplementedError('post_processing not implemented')

  @property
  def exportable_model_fn(
      self,
  ) -> Callable[[NestedJTensor, Tuple[NestedJTensor, JTensor]], NestedJTensor]:
    """Exportable model function for `ExportableToSavedModel` protocol."""

    def _wrapped_fn(
        mdl_vars: NestedJTensor,
        inputs_with_rng_seed: Tuple[NestedJTensor, JTensor],
    ) -> NestedJTensor:
      # Remove padding on the vars.
      mdl_vars = jax.tree_util.tree_map(
          servable_model.remove_padding,
          mdl_vars,
          self.model_state.mdl_var_unpadded_shapes,
      )
      mdl_vars = jax.tree_util.tree_map(
          pjit.with_sharding_constraint,
          mdl_vars,
          self.model_state.mdl_var_pspecs,
      )
      inputs, seed = inputs_with_rng_seed
      prng_key = jax.random.PRNGKey(seed)  # pytype: disable=wrong-arg-types  # jax-ndarray
      return self.jax_func(mdl_vars, prng_key, inputs, ())

    # pjit-ed function.
    return pjit.pjit(
        _wrapped_fn,
        in_axis_resources=(self.model_state.mdl_var_pspecs, None),
        out_axis_resources=None,
        donate_argnums=(1,),
    )

  @property
  def model_fn_input_polymorphic_shape(self) -> pytypes.Nested[str]:
    """Returns a batch polymorphic shape for jax2tf."""
    batched_host_dummy = self.get_dummy_inputs(InputShapeInfo(self.batch_size))
    batched_host_dummy = self.update_extra_inputs(
        batched_host_dummy,
        self.batch_size,
        [self.default_extra_inputs] * self.batch_size,
    )
    batch_pattern = 'b, ...' if len(self.sorted_batch_sizes) > 1 else None
    return jax.tree_util.tree_map(lambda _: batch_pattern, batched_host_dummy)


class ServableModel(servable_model.ServableModel):
  """Base class for service implementation, backed by a model.

  This class loads model checkpoints and initializes its methods. It uses a
  `primary_process_id` to identify primary and secondary hosts in a
  multi-Jax-client setup.

  Subclasses need to override init_method() to create model-specific
  ServableMethod.
  """

  def __init__(
      self,
      model_config: servable_model_params.ServableModelParams,
      primary_process_id: int,
      ckpt_type: CheckpointType,
      test_mode: bool = False,
  ):
    super().__init__()
    self._test_mode = test_mode
    self._primary_process_id = primary_process_id
    assert ckpt_type in (CheckpointType.GDA, CheckpointType.PERSISTENCE)
    self._ckpt_type = ckpt_type
    self._model_config = model_config

  @property
  def primary_process_id(self) -> int:
    return self._primary_process_id

  @property
  def model_config(self) -> servable_model_params.ServableModelParams:
    return self._model_config

  def load(
      self,
      checkpoint_path: Optional[str],
      prng_key: PRNGKey,
      precompile: bool = True,
  ) -> None:
    if self._test_mode:
      logging.info('Ignoring checkpoint_path %s in test mode.', checkpoint_path)
      checkpoint_path = None
    prng_key, init_key = jax.random.split(prng_key)
    model, model_state = self.load_state(checkpoint_path, init_key, precompile)
    self.load_methods(model, model_state, prng_key)

  def load_state(
      self,
      checkpoint_path: Optional[str],
      prng_key: PRNGKey,
      precompile: bool = True,
  ) -> Tuple[base_model.BaseModel, ServableModelState]:
    """Initializes the model state."""
    task_p = self._model_config.task()
    jax_task = task_p.Instantiate()
    model_p = task_p.model  # pytype: disable=attribute-error  # enable-nested-classes

    prng_key, init_key = jax.random.split(prng_key)

    status, device_mesh = self._model_config.get_supported_device_mesh()
    if not status.ok():
      raise ValueError(status.details)

    logging.info('device_mesh: %s', device_mesh)
    global_mesh = jax.sharding.Mesh(device_mesh, model_p.mesh_axis_names)
    self._global_mesh = global_mesh

    # TODO(zhangqiaorjc, yuanzx): Retrieve unpadded var shapes from checkpoint.
    sample_input_for_init = self.model_config.input_for_model_init()
    with global_mesh:
      vars_weight_params = jax_task.model.abstract_init_with_metadata(
          sample_input_for_init
      )
      discard_opt_states = not self._model_config.load_ema()
      train_state_global_shapes = jax_task.create_train_state_padded_shapes(
          vars_weight_params, discard_opt_states=discard_opt_states
      )

      if model_p.fprop_dtype == jnp.bfloat16:

        def maybe_to_bfloat16_dtype(x):
          if x.dtype in (jnp.float32, np.float32):
            return jax.ShapeDtypeStruct(x.shape, jnp.bfloat16)
          return x

        train_state_global_shapes = jax.tree_map(
            maybe_to_bfloat16_dtype, train_state_global_shapes
        )

      partition_specs = jax_task.create_train_state_partition_specs(
          vars_weight_params, discard_opt_states=discard_opt_states
      )
      if checkpoint_path is not None:
        checkpoint_path = epath.Path(checkpoint_path)
        if not checkpoint_path.is_dir():
          raise ValueError(
              f'Invalid checkpoint path {checkpoint_path}. Must be a directory.'
          )
        try:
          step = CKPT_MODULE.get_step_from_checkpoint_asset(checkpoint_path)
          maybe_updated_format = CKPT_MODULE.maybe_update_checkpoint_type(
              self._ckpt_type, checkpoint_path
          )
          checkpoint_path = checkpoint_path.parent
        except Exception as e:
          raise ValueError(
              f'Invalid checkpoint path {checkpoint_path}. Expected a step '
              'directory, e.g., /some/path/checkpoint_00100000'
          ) from e
        partitioned_train_state = CKPT_MODULE.restore_checkpoint(
            train_state_global_shapes,
            checkpoint_path,
            global_mesh=global_mesh,
            checkpoint_type=maybe_updated_format,
            state_specs=partition_specs,
            step=step,
        )
      else:
        assert self._test_mode, 'Must provide checkpoint unless in test mode'
        partitioned_train_state, _ = (
            trainer_lib.initialize_partitioned_model_states(
                jax_task,
                init_key,
                sample_input_for_init,
                partition_specs,
                discard_opt_states=discard_opt_states,
                global_mesh=global_mesh,
            )
        )
        step = 0
      assert partitioned_train_state is not None
      if self._model_config.load_ema():
        logging.info('loading ema from checkpoint')
        partitioned_train_state = tasks_lib.extract_ema(partitioned_train_state)
      mdl_vars = partitioned_train_state.mdl_vars
      del partitioned_train_state
      mdl_var_pspecs = partition_specs.mdl_vars
      mdl_var_unpadded_shapes = (
          jax_task.create_train_state_unpadded_shapes(
              vars_weight_params, discard_opt_states=discard_opt_states
          )
      ).mdl_vars
      mdl_var_unpadded_shapes = jax.tree_map(
          lambda x: x.shape, mdl_var_unpadded_shapes
      )

      model = jax_task.model
      logging.info('quant_mode: %s', self.model_config.quant_mode)
      if (
          self.model_config.quant_mode
          == quantization_hparams.QuantizationMode.MATERIALIZE
      ):
        # quantize model.
        def convert(x):
          # writing this as lamda triggers "g-long-lambda" warning.
          if x.dtype == jnp.float32:
            return x.astype(jnp.bfloat16)
          return x

        # get PartitionSpec for output.
        def quant_pspec_fn(mdl_vars_to_quant, prng_keys):
          mdl_vars_to_quant = py_utils.maybe_slice_uneven_sharding(
              mdl_vars_to_quant,
              mdl_var_pspecs,
              mdl_var_unpadded_shapes,
          )
          if (
              model_p.fprop_dtype == jnp.bfloat16
              or model_p.dtype == jnp.bfloat16
          ):
            # Convert float inputs/vars if fprop dtype is bfloat16.
            mdl_vars_to_quant = jax.tree_map(convert, mdl_vars_to_quant)
          k1, k2, prng_keys = jax.random.split(prng_keys, num=3)
          return jax_task.model.apply(
              mdl_vars_to_quant,
              mutable=[],
              method=jax_task.model.quantized_partition_specs,
              rngs={
                  base_layer.PARAMS: k1,
                  base_layer.RANDOM: k2,
              },
          )

        pjit_quant_pspec_fn = pjit.pjit(
            quant_pspec_fn,
            in_shardings=(mdl_var_pspecs, None),
            out_shardings=None,
        )
        new_pspec, _ = pjit_quant_pspec_fn(mdl_vars, prng_key)
        # pylint: disable=g-long-lambda
        new_pspec = jax.tree_map(
            lambda x: x.meta
            if isinstance(x, base_layer.BoxedPartitionSpec)
            else x,
            new_pspec,
            is_leaf=lambda x: isinstance(x, base_layer.BoxedPartitionSpec),
        )

        # pylint: enable=g-long-lambda

        def quant_fn(mdl_vars_to_quant, prng_keys):
          mdl_vars_to_quant = py_utils.maybe_slice_uneven_sharding(
              mdl_vars_to_quant,
              mdl_var_pspecs,
              mdl_var_unpadded_shapes,
          )
          if (
              model_p.fprop_dtype == jnp.bfloat16
              or model_p.dtype == jnp.bfloat16
          ):
            # Convert float inputs/vars if fprop dtype is bfloat16.
            mdl_vars_to_quant = jax.tree_map(convert, mdl_vars_to_quant)
          k1, k2, prng_keys = jax.random.split(prng_keys, num=3)
          return jax_task.model.apply(
              mdl_vars_to_quant,
              mutable=[],
              method=jax_task.model.quantize_weight,
              rngs={
                  base_layer.PARAMS: k1,
                  base_layer.RANDOM: k2,
              },
          )

        pjit_quant_fn = pjit.pjit(
            quant_fn,
            in_shardings=(mdl_var_pspecs, None),
            out_shardings=(new_pspec, None),
        )
        mdl_vars, _ = pjit_quant_fn(mdl_vars, prng_key)
        new_task_p = self._model_config.task()
        quantize.set_inference_mode(new_task_p.model)
        new_jax_task = new_task_p.Instantiate()
        model = new_jax_task.model
        task_p = new_task_p
        # TODO(jianlijianli): Get unpadded_shapes properly.
        mdl_var_unpadded_shapes = jax.tree_map(lambda x: x.shape, mdl_vars)
        mdl_var_unpadded_types = jax.tree_map(lambda x: x.dtype, mdl_vars)
        logging.info('quantized vars pspec %s', new_pspec)
        logging.info('quantized vars shapes %s', mdl_var_unpadded_shapes)
        logging.info('quantized vars types %s', mdl_var_unpadded_types)
        logging.info('quantized model %s', new_task_p.to_text())
        mdl_var_pspecs = new_pspec

      # load model.
      model_state = ServableModelState(
          is_primary_host=jax.process_index() == self._primary_process_id,
          primary_process_id=self._primary_process_id,
          global_mesh=self._global_mesh,
          mdl_vars=mdl_vars,
          mdl_var_pspecs=mdl_var_pspecs,
          mdl_var_unpadded_shapes=mdl_var_unpadded_shapes,
          input_prefetch=self._ckpt_type == CheckpointType.GDA,
          precompile=precompile,
          step=step,
      )
      return model, model_state

  def load_methods(
      self,
      model: base_model.BaseModel,
      model_state: ServableModelState,
      prng_key: PRNGKey,
  ) -> None:
    try:
      method_params = self.model_config.methods()
      for method in sorted(method_params.keys()):
        prng_key, method_prng_key = jax.random.split(prng_key)
        params = method_params[method]
        assert isinstance(params, servable_model_params.ServableMethodParams)
        self.add_method(
            method,
            self.init_method(
                method, model, model_state, params, method_prng_key
            ),
        )
    except Exception as e:
      self.unload()
      raise e
    logging.info('loading completed.')

  def init_method(
      self,
      method: str,
      model: base_model.BaseModel,
      model_state: ServableModelState,
      method_params: servable_model_params.ServableMethodParams,
      prng_key: PRNGKey,
  ) -> ServableMethod:
    raise NotImplementedError(f'method {method} not implemented')
