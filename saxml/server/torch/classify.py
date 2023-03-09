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
"""Image classification models."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from paxml.tasks.vision import resnet_preprocessing
from saxml.server import servable_model_params
from saxml.server import servable_model_registry
from saxml.server import utils
from saxml.server.pax.vision import imagenet_metadata
from saxml.server.services import vision_service
from saxml.server.torch import servable_model
import torch
from torchvision.models import resnet

DeviceTensors = Any
HostTensors = Any
ServableMethod = servable_model.ServableMethod
ServableMethodParams = servable_model.ServableMethodParams
ServableModel = servable_model.ServableModel
VisionMethodName = vision_service.VisionMethodName


@servable_model_registry.register
class ResNetModel(servable_model_params.ServableModelParams):
  """Model params for a torchvision resnet50.

  Available checkpoints:
    IMAGENET1K_V2 from
    https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
    is available at:
    /placer/prod/home/brain-babelfish/yuxinw/torch_home/hub/checkpoints/resnet50-11ad3fa6.pth
  """

  def load(
      self,
      model_key: str,
      checkpoint_path: str,
      primary_process_id: int,
      prng_key: int,
  ) -> Any:
    model = resnet.resnet50().cuda()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(checkpoint_path, progress=True)
    )
    return ServableModel(model, self.methods(), device="cuda")

  @classmethod
  def get_supported_device_mesh(
      cls,
  ) -> Tuple[utils.Status, Optional[np.ndarray]]:
    """Returns OK status and the supported device mesh, non-OK if this model is not supported."""
    if torch.cuda.is_available():
      return utils.ok(), None
    else:
      return utils.unimplemented("CUDA is not available!"), None

  @classmethod
  def check_serving_platform(cls) -> utils.Status:
    if torch.cuda.is_available():
      return utils.ok()
    else:
      return utils.unimplemented("CUDA is not available!")

  def methods(self) -> Dict[str, servable_model.ServableMethodParams]:
    return {
        VisionMethodName.CLASSIFY: ServableMethodParams(
            method_cls=ClassifyMethod,
            method_attrs={
                # Taken from
                # https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50
                "pixel_mean": torch.Tensor([0.485, 0.456, 0.406]) * 255,
                "pixel_std": torch.Tensor([0.229, 0.224, 0.225]) * 255,
                "image_size": 224,
            },
        )
    }


class ClassifyMethod(ServableMethod):
  """Classification method."""

  pixel_mean: torch.Tensor
  pixel_std: torch.Tensor
  image_size: int = 224
  top_k: int = 10

  @classmethod
  def service_id(cls) -> str:
    """Unique ID for the model service that supports this model."""
    return vision_service.SERVICE_ID

  def pre_processing(self, raw_inputs: List[Any]) -> HostTensors:
    """Preprocesses inputs into RGB images of range [0, 255], then normalize."""
    images = []
    for inp in raw_inputs:
      image_bytes = inp["image_bytes"]
      # This is slightly different from torchvision's IN1K_V2 preprocessing.
      image = resnet_preprocessing.preprocess_image(
          image_bytes, image_size=self.image_size
      )
      images.append(image)
    batch = torch.from_numpy(np.stack(images, axis=0))
    batch = (batch - self.pixel_mean) / self.pixel_std
    return batch.permute(0, 3, 1, 2)

  def post_processing(self, scores: HostTensors) -> List[Any]:
    """Processes output logits into labels and scores."""
    # Return top predictions.
    idx = np.argsort(-scores, axis=-1)[:, : self.top_k]

    # Convert each index to a string.
    vfn = np.vectorize(self._id_to_string)
    labels = np.reshape(vfn(np.reshape(idx, [-1])), idx.shape)

    # Fetch the associated scores for each top index.
    top_scores = np.take_along_axis(scores, idx, axis=-1)
    return list(zip(labels, top_scores))

  def _id_to_string(self, idx: int) -> str:
    """Maps class id in [0, 1000) to label name."""
    return imagenet_metadata.CLASS_LIST[idx]
