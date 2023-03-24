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
"""Wraps a model with VisionService APIs."""

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from jax import numpy as jnp
from lingvo.core import cluster_factory
import numpy as np
from paxml import base_task
from praxis import base_layer
from praxis import base_model
from praxis import pax_fiddle
from praxis import py_utils
from praxis import pytypes
from saxml.server.pax import servable_model
from saxml.server.pax import servable_model_params
from saxml.server.services import vision_service
import tensorflow as tf

JTensor = pytypes.JTensor
NestedJTensor = pytypes.NestedJTensor
NestedNpTensor = pytypes.NestedNpTensor
PRNGKey = pytypes.PRNGKey
NestedMap = py_utils.NestedMap
NestedPartitionSpec = pytypes.NestedPartitionSpec
NestedTfTensor = pytypes.Nested[tf.Tensor]
NestedTfTensorSpec = pytypes.Nested[tf.TensorSpec]
NestedTfTrackable = pytypes.Nested[
    tf.saved_model.experimental.TrackableResource
]
NpTensor = pytypes.NpTensor
VisionMethodName = vision_service.VisionMethodName


class ClassifyHParams(servable_model_params.ServableMethodParams):
  """HParameters for Classification method.

  Attributes:
    top_k: top k results from classification.
  """

  top_k: int = 0


class TextToImageHParams(servable_model_params.ServableMethodParams):
  """HParameters for TextToImage method.

  Attributes:
    max_seq_length: max sequence length for tokenizer.
    num_samples: number of samples.
    top_samples_to_keep: if not None, it is the number of top samples to return
      to the client based on their scores.
    text_preprocessor: optional text pre-processing function before
      tokenization.
    image_postprocessor: optional image post-processor function. Input and
      output are both encoded a list of lists of PNG bytes, which has the shape
      (batch, samples).
    tf_text_preprocessor: optional text pre-processing function in TF context.
      Used by SavedModel exporting only.
    tf_image_postprocessor: optional image post-processing function in TF
      context. Used by SavedModel exporting only.
    tf_input_tokenized: whether the exported SavedModel's signature accepts
      tokenized inputs.
  """

  max_seq_length: int = 0
  num_samples: int = 0
  top_samples_to_keep: Optional[int] = None
  text_preprocessor: Optional[Callable[[str], str]] = None
  image_postprocessor: Optional[
      Callable[[List[List[bytes]]], List[List[bytes]]]
  ] = None
  tf_text_preprocessor: Optional[Callable[[tf.Tensor], tf.Tensor]] = None
  tf_image_postprocessor: Optional[Callable[[tf.Tensor], tf.Tensor]] = None
  tf_input_tokenized: bool = False


class EmbedHParams(servable_model_params.ServableMethodParams):
  """HParameters for Embed method.

  Attributes:
    image_preprocessor: Pre-processing function to convert image_bytes into
      image Tensor.  Required.
    model_method_name: The name of the method to call to extract embeddings from
      an input image.  Required.
    output_embedding_name: The name of the embedding to use from the model's
      outputs.  Required.
  """

  image_preprocessor: Optional[Callable[[str], tf.Tensor]] = None
  model_method_name: Optional[str] = None
  output_embedding_name: Optional[str] = None


class DetectHParams(servable_model_params.ServableMethodParams):
  """HParameters for Detect method.

  Attributes:
    is_open_set: Indicates whether the model supports open set detection by
      passing text arguments to the Detect() API.
    model_method_name: The name of the method to call to extract embeddings from
      an input image.  Required.
  """

  is_open_set: bool = False
  model_method_name: Optional[str] = None


class ImageToTextHParams(servable_model_params.ServableMethodParams):
  """HParameters for ImageToText method.

  Attributes:
    model_method_name: The name of the method to call to extract embeddings from
      an input image.  Required.
  """

  model_method_name: Optional[str] = None


class VideoToTextHParams(servable_model_params.ServableMethodParams):
  """HParameters for VideoToText method.

  Attributes:
    model_method_name: The name of the method to call to extract embeddings from
      an input image.  Required.
  """

  model_method_name: Optional[str] = None


class VisionModelParamsBase(servable_model_params.ServableModelParams):
  """Base Vision Model params.

  Subclasses should define the serving mesh shapes supported,
  and the serving_dataset from which the input preprocessing for the
  model is defined.
  """

  @classmethod
  def serving_mesh_shape(cls):
    # Concrete subclasses should override this as needed.
    raise NotImplementedError()

  @property
  def serving_batch_size(self):
    """The static batch size to use for serving."""
    raise NotImplementedError()

  def methods(self) -> Dict[str, servable_model_params.ServableMethodParams]:
    methods = {}

    # pylint: disable=assignment-from-none
    classify_params = self.classify()
    if classify_params is not None:
      methods[VisionMethodName.CLASSIFY] = classify_params
    text_to_image_params = self.text_to_image()
    if text_to_image_params is not None:
      methods[VisionMethodName.TEXT_TO_IMAGE] = text_to_image_params
    embed_params = self.embed()
    if embed_params is not None:
      methods[VisionMethodName.EMBED] = embed_params
    detect_params = self.detect()
    if detect_params is not None:
      methods[VisionMethodName.DETECT] = detect_params
    image_to_text_params = self.image_to_text()
    if image_to_text_params is not None:
      methods[VisionMethodName.IMAGE_TO_TEXT] = image_to_text_params
    video_to_text_params = self.video_to_text()
    if video_to_text_params is not None:
      methods[VisionMethodName.VIDEO_TO_TEXT] = video_to_text_params
    # pylint: enable=assignment-from-none
    return methods

  def task(self) -> base_task.BaseTask.HParams:
    p = super().task()
    # We do this because p looks like a BaseParameterizable, not a BaseModel
    # that has a model attribute.
    model = getattr(p, 'model', None)
    if model:
      model.ici_mesh_shape = VisionModelParams.serving_mesh_shape()
    return p

  def classify(self) -> Optional[ClassifyHParams]:
    return None

  def text_to_image(self) -> Optional[TextToImageHParams]:
    return None

  def embed(self) -> Optional[EmbedHParams]:
    return None

  def detect(self) -> Optional[DetectHParams]:
    return None

  def image_to_text(self) -> Optional[ImageToTextHParams]:
    return None

  def video_to_text(self) -> Optional[VideoToTextHParams]:
    return None

  def create_model(self, primary_process_id: int) -> 'VisionModel':
    return VisionModel(
        self,
        primary_process_id,
        self.get_checkpoint_type(),
        test_mode=self.test_mode,
    )


class VisionModelParams(VisionModelParamsBase):
  """Model params for image classification task."""

  def id_to_string(self, idx: int) -> str:
    """Converts from the index of the softmax to a string label name."""
    raise NotImplementedError()

  def serving_dataset(self):
    """Dataset used to define serving preprocessing by the model."""
    raise NotImplementedError()


class TextToImageModelParams(VisionModelParamsBase):
  """Model params for text-to-image task."""

  def serving_tokenizer(self) -> pax_fiddle.Config[base_layer.BaseLayer]:
    """Specifies the tokenizer."""
    raise NotImplementedError()


class ImageBytesToLabelScorePairs(servable_model.ServableMethod):
  """Method for implementing image_bytes -> (label,score) extraction."""

  def __init__(
      self,
      model,
      model_fn_name: str,
      model_state,
      method_hparams: ClassifyHParams,
      prng_key,
      dummy_input_sample: Any,
      model_config: Any,
  ):
    self._model_config = model_config
    self._dataset = model_config.serving_dataset()
    self._cluster = copy.deepcopy(cluster_factory.Current())
    self._cluster.params.do_eval = True
    with self._cluster:
      self._input_processor = self._dataset.input.Instantiate()

    super().__init__(
        model,
        model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
    )

  @classmethod
  def service_id(cls) -> str:
    return vision_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    return NestedMap(
        logp=model_fn_outputs[0]['logp'],
    )

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    images = []
    with self._cluster:
      for inp in raw_inputs:
        image_bytes = inp['image_bytes']
        image_data = self._input_processor.ImageBytesToBatch(image_bytes)
        images.append(image_data.image[0])
    images = np.stack(images)
    processed_input_batch = NestedMap(image=images)
    return processed_input_batch

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    # Convert scores to top_k class_list.
    top_k = self._model_config.classify().top_k
    scores = compute_outputs['logp']

    # Find indices of topK scores.
    #
    # We can also use argpartition to speed this up.
    idx = np.argsort(-scores, axis=-1)[:, :top_k]

    # Convert each index to a string.
    vfn = np.vectorize(self._model_config.id_to_string)
    labels = np.reshape(vfn(np.reshape(idx, [-1])), idx.shape)

    # Fetch the associated scores for each top index.
    top_scores = np.take_along_axis(scores, idx, axis=-1)

    # Combine together for each sample.
    output = list(zip(labels, top_scores))
    return output


@tf.function
def _sort_and_encode_images(images, scores, samples_to_keep):
  """Sorts images by score and encode them to png."""
  # Sort and slice.
  ids = tf.argsort(scores, axis=-1, direction='DESCENDING')
  images = tf.gather(images, ids, batch_dims=1)[:, :samples_to_keep]
  scores = tf.gather(scores, ids, batch_dims=1)[:, :samples_to_keep]
  # Encode.
  shape = tf.shape(images)
  b = shape[0]
  n = shape[1]
  images = tf.reshape(images, tf.concat([[b * n], shape[2:]], axis=0))
  encoded = tf.vectorized_map(tf.image.encode_png, images)
  return tf.reshape(encoded, (b, n)), scores


class TextToImageMethod(servable_model.ServableMethod):
  """Method for implementing text -> [(image_bytes,score)] extraction."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_fn_name: str,
      model_state: servable_model.ServableModelState,
      method_hparams: TextToImageHParams,
      prng_key: PRNGKey,
      dummy_input_sample: Any,
      model_config: Any,
  ):
    self._tokenizer = model_config.serving_tokenizer().Instantiate()
    self._max_length = method_hparams.max_seq_length
    self._text_preprocessor = method_hparams.text_preprocessor
    self._image_postprocessor = method_hparams.image_postprocessor
    self._tf_text_preprocessor = method_hparams.tf_text_preprocessor
    self._tf_image_postprocessor = method_hparams.tf_image_postprocessor
    self._tf_input_tokenized = method_hparams.tf_input_tokenized
    self._top_samples_to_keep = method_hparams.top_samples_to_keep
    exportable = (
        self._tf_text_preprocessor is not None
        and self._tf_image_postprocessor is not None
    )
    super().__init__(
        model,
        model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
        exportable,
    )

  @classmethod
  def service_id(cls) -> str:
    return vision_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    # Fetech useful information from output.
    images = model_fn_outputs[0]['generated_images']
    # TODO(jianlijianli): check model output contract.
    assert images.dtype == jnp.uint8, images.dtype
    return NestedMap(images=images, scores=model_fn_outputs[0]['scores'])

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    if self._text_preprocessor is not None:
      raw_inputs = [self._text_preprocessor(inp) for inp in raw_inputs]
    ids, _, paddings = self._tokenizer.StringsToIds(
        raw_inputs, max_length=self._max_length
    )
    ids = np.array(ids)
    paddings = np.array(paddings)
    return NestedMap(ids=ids, paddings=paddings)

  def _sort_and_encode(
      self,
      compute_outputs: Union[NestedNpTensor, NestedTfTensor],
      tf_mode=False,
  ) -> Tuple[Any, Any]:
    images = compute_outputs['images']
    scores = compute_outputs['scores']

    # Assumes images are [b, n, h, w, c] and scores are [b, n]
    # TODO(jianlijianli): consider supporting other input shapes, if needed.
    assert 5 == len(images.shape), images.shape
    b, n, _, _, _ = images.shape
    # scores are already summed in model.text_to_image().
    assert scores.shape == (b, n), scores.shape

    samples_to_keep = self._top_samples_to_keep or n
    samples_to_keep = min(samples_to_keep, n)
    image_bytes, scores = _sort_and_encode_images(
        images, scores, samples_to_keep
    )
    return image_bytes, scores

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    image_bytes, scores = self._sort_and_encode(compute_outputs)
    image_bytes = image_bytes.numpy()
    scores = scores.numpy()
    if self._image_postprocessor is not None:
      image_bytes = [list(imgs) for imgs in image_bytes]
      image_bytes = self._image_postprocessor(image_bytes)
    return [(list(i), list(s)) for i, s in zip(image_bytes, scores)]

  def tf_pre_processing(self, inputs: NestedTfTensorSpec) -> NestedTfTensorSpec:
    # TODO(dinghua): Consider moving this to the prediction method.
    if self._tf_input_tokenized:
      ids = inputs['token_ids']
      paddings = inputs['paddings']
    else:
      if self._text_preprocessor is not None:
        inputs = self._tf_text_preprocessor(inputs)
      ids, _, paddings = self._tokenizer.StringsToIds(
          inputs, max_length=self._max_length
      )
    ids = tf.ensure_shape(ids, [self.batch_size, self._max_length])
    paddings = tf.ensure_shape(paddings, [self.batch_size, self._max_length])
    return NestedMap(ids=ids, paddings=paddings)

  def tf_post_processing(
      self, compute_outputs: NestedTfTensor
  ) -> NestedTfTensor:
    image_bytes, scores = self._sort_and_encode(compute_outputs)
    if self._image_postprocessor is not None:
      image_bytes = self._tf_image_postprocessor(image_bytes)
    return {'images': image_bytes, 'scores': scores}

  def input_signature(
      self, batch_size: Optional[int]
  ) -> List[NestedTfTensorSpec]:
    if self._tf_input_tokenized:
      return [{
          'token_ids': tf.TensorSpec(
              shape=[batch_size, self._max_length],
              dtype=tf.int32,
              name='token_ids',
          ),
          'paddings': tf.TensorSpec(
              shape=[batch_size, self._max_length],
              dtype=tf.float32,
              name='paddings',
          ),
      }]
    else:
      return [
          tf.TensorSpec(shape=[batch_size], dtype=tf.string, name='text_batch')
      ]

  @property
  def extra_trackables(self) -> Optional[NestedTfTrackable]:
    return None


class ImageBytesToEmbedding(servable_model.ServableMethod):
  """Method to go from image_bytes to image embedding."""

  def __init__(
      self,
      model: base_model.BaseModel,
      model_fn_name: str,
      model_state: servable_model.ServableModelState,
      method_hparams: EmbedHParams,
      prng_key: PRNGKey,
      dummy_input_sample: Any,
      model_config: Any,
  ):
    self._model_config = model_config
    if method_hparams.image_preprocessor is None:
      raise ValueError(
          'image_preprocessor method must be defined in EmbedHParams'
      )
    self._image_preprocessor = method_hparams.image_preprocessor
    self._embedding_name = method_hparams.output_embedding_name
    super().__init__(
        model,
        model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
    )

  @classmethod
  def service_id(cls) -> str:
    return vision_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    image_embedding = model_fn_outputs[0].GetItem(self._embedding_name)  # pytype: disable=attribute-error  # jax-ndarray
    return NestedMap(image_embedding=image_embedding)

  @tf.function
  def _preprocess_batch(self, image_bytes_batch):
    return tf.map_fn(
        self._image_preprocessor, image_bytes_batch, dtype=tf.float32
    )

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    image_bytes_batch = tf.convert_to_tensor(
        [inp['image_bytes'] for inp in raw_inputs]
    )
    images = self._preprocess_batch(image_bytes_batch).numpy()
    processed_input_batch = NestedMap(image=images)
    return processed_input_batch

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    image_embedding = compute_outputs['image_embedding']
    if image_embedding.dtype not in [np.float32, np.float64]:
      image_embedding = image_embedding.astype(np.float32)
    return list(image_embedding)


class ImageBytesToDetect(servable_model.ServableMethod):
  """Method for implementing detection."""

  def __init__(
      self,
      model,
      model_fn_name: str,
      model_state,
      method_hparams: DetectHParams,
      prng_key,
      dummy_input_sample: Any,
      model_config: Any,
  ):
    self._model_config = model_config
    model_config.init_for_serving()
    super().__init__(
        model,
        model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
    )

  @classmethod
  def service_id(cls) -> str:
    return vision_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    return self._model_config.fetch_output(model_fn_outputs, model_fn_inputs)

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    return self._model_config.pre_processing(raw_inputs)

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    return self._model_config.post_processing(compute_outputs)


class ImageBytesToText(servable_model.ServableMethod):
  """Method for implementing image->text."""

  def __init__(
      self,
      model,
      model_fn_name: str,
      model_state,
      method_hparams: Union[ImageToTextHParams, VideoToTextHParams],
      prng_key,
      dummy_input_sample: Any,
      model_config: Any,
  ):
    self._model_config = model_config

    self._dataset = model_config.serving_dataset()
    self._tokenizer = model_config.serving_tokenizer().Instantiate()

    self._cluster = copy.deepcopy(cluster_factory.Current())
    self._cluster.params.do_eval = True
    with self._cluster:
      self._input_processor = self._dataset.input.Instantiate()

    super().__init__(
        model,
        model_fn_name,
        model_state,
        method_hparams,
        prng_key,
        dummy_input_sample,
    )

  @classmethod
  def service_id(cls) -> str:
    return vision_service.SERVICE_ID

  def fetch_output(
      self, model_fn_outputs: NestedJTensor, model_fn_inputs: NestedJTensor
  ) -> NestedJTensor:
    """Fetches useful output tensors from the model function outputs."""
    _, results = model_fn_outputs[0]
    logprobs = results['logprobs']  # [batch, num_hyps, max_len]
    # Sum valid log probs for each hyp to produce a score.
    valid = (logprobs != 1.0) * 1.0
    logprobs = jnp.sum(logprobs * valid, axis=-1)
    return NestedMap(
        hyps=results['hyp'], hyplen=results['hyplen'], logprobs=logprobs
    )

  def _preprocess_images(self, raw_input: Any) -> NestedNpTensor:
    """Preprocesses images on one unpadded data."""
    image_bytes = raw_input['image_bytes']
    image_data = self._input_processor.ImageBytesToBatch(image_bytes)
    image_data = image_data.Transform(lambda x: x[0])
    return image_data

  def pre_processing(self, raw_inputs: List[Any]) -> NestedNpTensor:
    """Preprocesses an unpadded batch of data into host numpy arrays."""
    images = []
    image_info = []
    texts = []
    with self._cluster:
      # TODO(weihan,sax-dev): wrap this loop with tf.function.
      for inp in raw_inputs:
        image_data = self._preprocess_images(inp)
        images.append(image_data['image'])
        if 'image_info' in image_data:
          image_info.append(image_data['image_info'])
        texts += [inp['text']]

    images = np.stack(images)

    # Tokenizer text prefixes.
    #
    # Potential optimization: use _empty_ids / paddings if all texts are empty.
    ids, _, paddings = self._tokenizer.StringsToIds(
        texts, self._model_config.TEXT_MAX_LEN
    )

    # Add dimension to make [B, N, T] (N=1), which is better supported in the
    # pax codebase.
    ids = ids[:, tf.newaxis, :]
    paddings = paddings[:, tf.newaxis, :]

    ids = ids.numpy()
    paddings = paddings.numpy()

    # For now, target_ids and target_paddings must be set
    # for prefix decoding.  Since inference doesn't have targets, we just
    # set it to the same value as ids, paddings.
    processed_input_batch = NestedMap(
        image=images,
        ids=ids,
        paddings=paddings,
        target_ids=ids,
        target_paddings=paddings,
    )

    if image_info:
      processed_input_batch.image_info = np.stack(image_info)

    return processed_input_batch

  def post_processing(self, compute_outputs: NestedNpTensor) -> List[Any]:
    """Postprocesses the output numpy arrays to final host output."""
    # Take output ids and convert back to strings using tokenizer.
    hyps = compute_outputs['hyps']  # [batch, num_hyps, max_len]
    hyplen = compute_outputs['hyplen']  # [batch, num_hyps]
    logprobs = compute_outputs['logprobs']  # [batch, num_hyps]
    batch_size = len(hyps)
    output_text = []
    for i in range(batch_size):
      text = self._tokenizer.IdsToStrings(hyps[i], hyplen[i]).numpy()
      output_text.append(text)
    return [
        (list(text), list(score))
        for text, score in zip(output_text, list(logprobs))
    ]


class VideoBytesToText(ImageBytesToText):
  """Method for implementing video->text."""

  def _preprocess_images(self, raw_input: Any) -> NestedNpTensor:
    """Preprocesses images on one unpadded data."""
    image_frames = tf.convert_to_tensor(raw_input['image_frames'])
    image_data = self._input_processor.ImageBytesToBatch(image_frames)
    image_data = image_data.Transform(lambda x: x[0])
    return image_data


class VisionModel(servable_model.ServableModel):
  """Model for vision tasks."""

  def init_method(
      self,
      method: str,
      model: base_model.BaseModel,
      model_state: servable_model.ServableModelState,
      method_params: servable_model_params.ServableMethodParams,
      prng_key: PRNGKey,
  ) -> servable_model.ServableMethod:
    if method == VisionMethodName.CLASSIFY:
      assert isinstance(method_params, ClassifyHParams)
      # Create dummy encoded jpeg of all ones.
      image_bytes = tf.image.encode_jpeg(np.ones((256, 256, 3), dtype=np.uint8))
      dummy_input = {'image_bytes': image_bytes}
      return ImageBytesToLabelScorePairs(
          model,
          'predict',
          model_state,
          method_params,
          prng_key=prng_key,
          dummy_input_sample=dummy_input,
          model_config=self.model_config,
      )
    elif method == VisionMethodName.TEXT_TO_IMAGE:
      assert isinstance(method_params, TextToImageHParams)
      return TextToImageMethod(
          model,
          'text_to_image',
          model_state,
          method_params,
          prng_key=prng_key,
          dummy_input_sample='',
          model_config=self.model_config,
      )
    elif method == VisionMethodName.EMBED:
      assert isinstance(method_params, EmbedHParams)
      if method_params.model_method_name is None:
        raise ValueError('Must specify `model_method_name` in EmbedHParams.')
      if method_params.output_embedding_name is None:
        raise ValueError(
            'Must specify `output_embedding_name` in EmbedHParams.'
        )
      image_bytes = tf.image.encode_jpeg(np.ones((256, 256, 3), dtype=np.uint8))
      dummy_input = {'image_bytes': image_bytes}
      return ImageBytesToEmbedding(
          model,
          method_params.model_method_name,
          model_state,
          method_params,
          prng_key=prng_key,
          dummy_input_sample=dummy_input,
          model_config=self.model_config,
      )
    elif method == VisionMethodName.DETECT:
      assert isinstance(method_params, DetectHParams)
      if method_params.model_method_name is None:
        raise ValueError('Must specify `model_method_name` in DetectHParams.')
      image_bytes = tf.image.encode_jpeg(np.ones((256, 256, 3), dtype=np.uint8))
      dummy_input = {'image_bytes': image_bytes}
      if method_params.is_open_set:
        dummy_input['text'] = ['dummy']
      return ImageBytesToDetect(
          model,
          method_params.model_method_name,
          model_state,
          method_params,
          prng_key=prng_key,
          dummy_input_sample=dummy_input,
          model_config=self.model_config,
      )
    elif method == VisionMethodName.IMAGE_TO_TEXT:
      assert isinstance(method_params, ImageToTextHParams)
      if method_params.model_method_name is None:
        raise ValueError(
            'Must specify `model_method_name` in ImageToTextHParams.'
        )
      image_bytes = tf.image.encode_jpeg(np.ones((256, 256, 3), dtype=np.uint8))
      dummy_input = {'image_bytes': image_bytes, 'text': ''}
      return ImageBytesToText(
          model,
          method_params.model_method_name,
          model_state,
          method_params,
          prng_key=prng_key,
          dummy_input_sample=dummy_input,
          model_config=self.model_config,
      )
    elif method == VisionMethodName.VIDEO_TO_TEXT:
      assert isinstance(method_params, VideoToTextHParams)
      if method_params.model_method_name is None:
        raise ValueError(
            'Must specify `model_method_name` in VideoToTextHParams.'
        )
      image_bytes = tf.image.encode_jpeg(np.ones((256, 256, 3), dtype=np.uint8))
      dummy_input = {
          'image_frames': [image_bytes, image_bytes],
          'text': 'dummy',
      }
      return VideoBytesToText(
          model,
          method_params.model_method_name,
          model_state,
          method_params,
          prng_key=prng_key,
          dummy_input_sample=dummy_input,
          model_config=self.model_config,
      )
    else:
      raise NotImplementedError(f'method {method} not implemented.')
