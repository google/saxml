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
"""Object detection models from detectron2."""
from typing import Any, Dict, List, Optional, Tuple
from detectron2 import checkpoint
from detectron2 import config
from detectron2 import modeling
import detectron2.data
from detectron2.utils import logger
import numpy as np
from saxml.server import servable_model_params
from saxml.server import servable_model_registry
from saxml.server import utils
from saxml.server.services import vision_service
from saxml.server.torch import servable_model
import tensorflow as tf
import torch

ServableMethod = servable_model.ServableMethod
ServableMethodParams = servable_model.ServableMethodParams
ServableModel = servable_model.ServableModel
VisionMethodName = vision_service.VisionMethodName


@servable_model_registry.register
class Detectron2Model(servable_model_params.ServableModelParams):
  """A detection model in detectron2.

  Available model paths:
    A small ResNet MaskRCNN:
    /cns/me-d/home/yuxinw/detectron2_sax_configs/COCO_mask_rcnn_R_50_FPN_3x.py

    A large VitDet MaskRCNN:
    /cns/me-d/home/yuxinw/detectron2_sax_configs/COCO_cascade_mask_rcnn_vitdet_h_75ep.py
  """

  def load(
      self,
      model_key: str,
      config_path: str,
      primary_process_id: int,
      prng_key: int,
  ) -> Any:
    logger.setup_logger()
    if config_path.endswith(".yaml"):
      cfg = config.get_cfg()
      cfg.merge_from_file(config_path)
      cfg.MODEL.DEVICE = "cuda"
      model = modeling.build_model(cfg)
      checkpoint.DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
      input_format = cfg.INPUT.FORMAT
    elif config_path.endswith(".py"):
      cfg = config.LazyConfig.load(config_path)
      model = config.instantiate(cfg.model)
      model.cuda()
      checkpoint.DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
      input_format = cfg.model.input_format
    if input_format not in ["RGB", "BGR"]:
      raise ValueError("Expects config to have either RGB or BGR input format.")
    methods = self.methods()
    for _, method_params in methods.items():
      method_params.method_attrs["input_format"] = input_format
    return ServableModel(model, methods, device="cuda")

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
        VisionMethodName.DETECT: ServableMethodParams(method_cls=DetectMethod)
    }


class DetectMethod(ServableMethod):
  """Detection method.

  Attributes:
      input_format: one of "BGR" or "RGB", the input image format used by the
        underlying model.
  """

  input_format: str = "RGB"

  @classmethod
  def service_id(cls) -> str:
    """Unique ID for the model service that supports this model."""
    return vision_service.SERVICE_ID

  def pre_processing(self, raw_inputs: List[Any]) -> Any:
    """Preprocesses inputs into RGB/BGR images of range [0, 255]."""
    inputs = []
    for inp in raw_inputs:
      image_bytes = inp["image_bytes"]
      if "texts" in inp:
        raise ValueError(
            "This detection model does not support text input! Got "
            + str(inp["texts"])
        )
      image = tf.io.decode_image(image_bytes, channels=3).numpy()
      height, width = image.shape[:2]
      image = torch.from_numpy(image).permute(2, 0, 1)
      if self.input_format == "BGR":
        image = torch.flip(image, [0])  # Model expects BGR format.
      inputs.append({"image": image, "height": height, "width": width})
    return inputs

  def post_processing(self, outputs: List[Any]) -> List[Any]:
    """Processes outputs into boxes, scores and text labels."""
    results = []
    for output in outputs:
      inst = output["instances"].to(device="cpu")
      # Convert boxes from XYXY to CxCyWH format.
      cxcy = inst.pred_boxes.get_centers().numpy()
      pred_boxes = inst.pred_boxes.tensor
      wh = (pred_boxes[:, 2:] - pred_boxes[:, :2]).numpy()
      pred_boxes = np.concatenate([cxcy, wh], axis=1)

      scores = inst.scores.numpy()
      pred_cls = inst.pred_classes.numpy()
      texts = [self._id_to_string(i) for i in pred_cls]
      results.append((pred_boxes, scores, texts))
    return results

  def _id_to_string(self, idx: int) -> str:
    """Maps class id in [0, 80) to label name."""
    # Note that it's different from the 90 classes in COCO metadata:
    # 10 invalid classes are removed in detectron2 models.
    return detectron2.data.MetadataCatalog.get("coco_2017_train").thing_classes[
        idx
    ]
