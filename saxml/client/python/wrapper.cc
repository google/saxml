// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "saxml/client/python/wrapper.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "saxml/client/cc/sax.h"
#include "saxml/protobuf/multimodal.pb.h"
#include "pybind11/gil.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace sax {
namespace client {
namespace pybind {

namespace {
#define RETURN_IF_ERROR(s) \
  {                        \
    auto c = (s);          \
    if (!c.ok()) return c; \
  }
}  // namespace

// Construct AudioModel with sax::client::Model. AudioModel does not take
// ownership of Model.
AudioModel::AudioModel(::sax::client::Model* base, const absl::Status& status) {
  status_ = status;
  base_ = base;
  if (base_) model_ = base_->AM();
}

AudioModel::AudioModel(const AudioModel& obj) {
  status_ = obj.status_;
  base_ = obj.base_;
  if (base_) model_ = base_->AM();
}

AudioModel::~AudioModel() {
  if (model_) delete model_;
}

absl::StatusOr<std::vector<std::pair<std::string, double>>>
AudioModel::Recognize(absl::string_view audio_bytes,
                      const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  std::vector<::sax::client::AudioModel::AsrHyp> hyps;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Recognize(audio_bytes, &hyps));
  } else {
    RETURN_IF_ERROR(model_->Recognize(*options, audio_bytes, &hyps));
  }
  std::vector<std::pair<std::string, double>> result;
  for (size_t i = 0; i < hyps.size(); i++) {
    auto& item = hyps[i];
    result.push_back(std::make_pair(std::move(item.text), item.score));
  }
  return result;
}

// Construct CustomModel with sax::client::Model. CustomModel does not take
// ownership of Model.
CustomModel::CustomModel(::sax::client::Model* base,
                         const absl::Status& status) {
  status_ = status;
  base_ = base;
  if (base_) model_ = base_->CM();
}

CustomModel::CustomModel(const CustomModel& obj) {
  status_ = obj.status_;
  base_ = obj.base_;
  if (base_) model_ = base_->CM();
}

CustomModel::~CustomModel() {
  if (model_) delete model_;
}

absl::StatusOr<pybind11::bytes> CustomModel::Custom(
    pybind11::bytes request, absl::string_view method_name,
    const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  std::string result;
  {
    pybind11::gil_scoped_release release;
    if (options == nullptr) {
      RETURN_IF_ERROR(model_->Custom(request, method_name, &result));
    } else {
      RETURN_IF_ERROR(model_->Custom(*options, request, method_name, &result));
    }
  }
  // NOTE: pybind11::bytes must be called within GIL.
  // TODO(changlan): Avoid memcpy here.
  return pybind11::bytes(result);
}

// Construct LanguageModel with sax::client::Model. LanguageModel does not take
// ownership of Model.
LanguageModel::LanguageModel(::sax::client::Model* base,
                             const absl::Status& status) {
  status_ = status;
  base_ = base;
  if (base_) model_ = base_->LM();
}

LanguageModel::LanguageModel(const LanguageModel& obj) {
  status_ = obj.status_;
  base_ = obj.base_;
  if (base_) model_ = base_->LM();
}

LanguageModel::~LanguageModel() {
  if (model_) delete model_;
}

absl::StatusOr<std::vector<double>> LanguageModel::Score(
    absl::string_view prefix, std::vector<absl::string_view> suffix,
    const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  std::vector<double> logp;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Score(prefix, suffix, &logp));
  } else {
    RETURN_IF_ERROR(model_->Score(*options, prefix, suffix, &logp));
  }
  return logp;
}

absl::StatusOr<std::vector<std::pair<std::string, double>>>
LanguageModel::Generate(absl::string_view prefix,
                        const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  std::vector<::sax::client::LanguageModel::ScoredText> samples;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Generate(prefix, &samples));
  } else {
    RETURN_IF_ERROR(model_->Generate(*options, prefix, &samples));
  }
  std::vector<std::pair<std::string, double>> result;
  for (size_t i = 0; i < samples.size(); i++) {
    auto& item = samples[i];
    result.emplace_back(std::make_pair(std::move(item.suffix), item.score));
  }
  return result;
}

absl::Status LanguageModel::GenerateStream(absl::string_view prefix,
                                           GenerateCallback callback,
                                           const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  // While holding the GIL, create a shared_ptr around the Python callback.
  auto fn = std::make_shared<GenerateCallback>(std::move(callback));
  // Now we can release the GIL.
  pybind11::gil_scoped_release release;
  // Capturing the shared_ptr won't trigger the Python callback's pybind11
  // std::function constructor, avoiding a "GIL not held" error.
  auto callback_wrapper =
      [fn](bool last,
           const std::vector<::sax::client::LanguageModel::GenerateItem>&
               items) {
        // Reacquire the GIL because the callback contains Python code.
        pybind11::gil_scoped_acquire acquire;
        std::vector<std::tuple<std::string, int, std::vector<double>>> r;
        r.reserve(items.size());
        if (last) return (*fn)(true, r);
        for (size_t i = 0; i < items.size(); i++) {
          auto& item = items[i];
          r.emplace_back(std::make_tuple(std::move(item.text), item.prefix_len,
                                         item.scores));
        }
        (*fn)(false, std::move(r));
      };

  // The shared_ptr is passed to additional owners, so the Python callback's
  // pybind11 std::function destructor won't be called either when this function
  // returns and the local ownership ends.
  if (options == nullptr) {
    return model_->GenerateStream(prefix, std::move(callback_wrapper));
  }
  return model_->GenerateStream(*options, prefix, std::move(callback_wrapper));
}

absl::StatusOr<std::vector<double>> LanguageModel::Embed(
    absl::string_view text, const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  std::vector<double> embedding;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Embed(text, &embedding));
  } else {
    RETURN_IF_ERROR(model_->Embed(*options, text, &embedding));
  }
  return embedding;
}

absl::StatusOr<std::pair<std::vector<double>,
                         absl::flat_hash_map<std::string, std::vector<double>>>>
LanguageModel::Gradient(absl::string_view prefix, absl::string_view suffix,
                        const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  std::vector<double> score;
  absl::flat_hash_map<std::string, std::vector<double>> gradients;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Gradient(prefix, suffix, &score, &gradients));
  } else {
    RETURN_IF_ERROR(
        model_->Gradient(*options, prefix, suffix, &score, &gradients));
  }
  return std::make_pair(score, gradients);
}

// Construct MultimodalModel with sax::client::Model. MultimodalModel does not
// take ownership of Model.
MultimodalModel::MultimodalModel(::sax::client::Model* base,
                                 const absl::Status& status) {
  status_ = status;
  base_ = base;
  if (base_) model_ = base_->MM();
}

MultimodalModel::MultimodalModel(const MultimodalModel& obj) {
  status_ = obj.status_;
  base_ = obj.base_;
  if (base_) model_ = base_->MM();
}

MultimodalModel::~MultimodalModel() {
  if (model_) delete model_;
}

absl::StatusOr<::sax::server::multimodal::GenerateResponse>
MultimodalModel::Generate(
    const ::sax::server::multimodal::GenerateRequest& request,
    const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  ::sax::server::multimodal::GenerateResponse response;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Generate(request, &response));
  } else {
    RETURN_IF_ERROR(model_->Generate(*options, request, &response));
  }
  return response;
}

absl::StatusOr<::sax::server::multimodal::ScoreResponse> MultimodalModel::Score(
    const ::sax::server::multimodal::ScoreRequest& request,
    const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  ::sax::server::multimodal::ScoreResponse response;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Score(request, &response));
  } else {
    RETURN_IF_ERROR(model_->Score(*options, request, &response));
  }
  return response;
}

// Construct VisionModel with sax::client::Model. VisionModel does not take
// ownership of Model.
VisionModel::VisionModel(::sax::client::Model* base,
                         const absl::Status& status) {
  status_ = status;
  base_ = base;
  if (base_) model_ = base_->VM();
}

VisionModel::VisionModel(const VisionModel& obj) {
  status_ = obj.status_;
  base_ = obj.base_;
  if (base_) model_ = base_->VM();
}
VisionModel::~VisionModel() {
  if (model_) delete model_;
}

absl::StatusOr<std::vector<std::pair<std::string, double>>>
VisionModel::Classify(absl::string_view image_bytes,
                      const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  std::vector<::sax::client::VisionModel::ScoredText> samples;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Classify(image_bytes, &samples));
  } else {
    RETURN_IF_ERROR(model_->Classify(*options, image_bytes, &samples));
  }
  std::vector<std::pair<std::string, double>> result;
  for (size_t i = 0; i < samples.size(); i++) {
    auto& item = samples[i];
    result.push_back(std::make_pair(std::move(item.text), item.score));
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<pybind11::bytes, double>>>
VisionModel::TextToImage(absl::string_view text,
                         const ModelOptions* options) const {
  if (!status_.ok()) return status_;
  std::vector<::sax::client::VisionModel::GeneratedImage> images;
  {
    pybind11::gil_scoped_release release;
    if (options == nullptr) {
      RETURN_IF_ERROR(model_->TextToImage(text, &images));
    } else {
      RETURN_IF_ERROR(model_->TextToImage(*options, text, &images));
    }
  }
  // NOTE: pybind11::bytes must be called within GIL.
  std::vector<std::pair<pybind11::bytes, double>> result;
  for (size_t i = 0; i < images.size(); i++) {
    auto& item = images[i];
    result.push_back(
        std::make_pair(pybind11::bytes(std::move(item.image)), item.score));
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<pybind11::bytes, double>>>
VisionModel::TextAndImageToImage(absl::string_view text,
                                 absl::string_view image_bytes,
                                 const ModelOptions* options) const {
  if (!status_.ok()) return status_;
  std::vector<::sax::client::VisionModel::GeneratedImage> generated_images;
  {
    pybind11::gil_scoped_release release;
    if (options == nullptr) {
      RETURN_IF_ERROR(
          model_->TextAndImageToImage(text, image_bytes, &generated_images));
    } else {
      RETURN_IF_ERROR(model_->TextAndImageToImage(*options, text, image_bytes,
                                                  &generated_images));
    }
  }
  // NOTE: pybind11::bytes must be called within GIL.
  std::vector<std::pair<pybind11::bytes, double>> result;
  for (size_t i = 0; i < generated_images.size(); i++) {
    auto& item = generated_images[i];
    result.push_back(
        std::make_pair(pybind11::bytes(std::move(item.image)), item.score));
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<pybind11::bytes, double>>>
VisionModel::ImageToImage(absl::string_view image,
                          const ModelOptions* options) const {
  if (!status_.ok()) return status_;
  std::vector<::sax::client::VisionModel::GeneratedImage> images;
  {
    pybind11::gil_scoped_release release;
    RETURN_IF_ERROR(model_->ImageToImage(image, &images));
  }
  // NOTE: pybind11::bytes must be called within GIL.
  std::vector<std::pair<pybind11::bytes, double>> result;
  for (size_t i = 0; i < images.size(); i++) {
    auto& item = images[i];
    result.push_back(
        std::make_pair(pybind11::bytes(std::move(item.image)), item.score));
  }
  return result;
}

absl::StatusOr<std::vector<double>> VisionModel::Embed(
    absl::string_view image, const ModelOptions* options) const {
  if (!status_.ok()) return status_;

  pybind11::gil_scoped_release release;
  std::vector<double> embedding;
  if (options == nullptr) {
    RETURN_IF_ERROR(model_->Embed(image, &embedding));
  } else {
    RETURN_IF_ERROR(model_->Embed(*options, image, &embedding));
  }
  return embedding;
}

absl::StatusOr<std::vector<VisionModel::PyDetectResult>> VisionModel::Detect(
    absl::string_view image, std::vector<std::string> text,
    std::vector<std::tuple<double, double, double, double>> boxes,
    const ModelOptions* options) const {
  if (!status_.ok()) return status_;
  std::vector<PyDetectResult> ret;

  std::vector<::sax::client::VisionModel::DetectResult> result;
  {
    pybind11::gil_scoped_release release;
    std::vector<::sax::client::VisionModel::BoundingBox> bounding_boxes;
    bounding_boxes.reserve(boxes.size());
    for (const auto& box : boxes) {
      bounding_boxes.push_back({std::get<0>(box), std::get<1>(box),
                                std::get<2>(box), std::get<3>(box)});
    }
    if (options == nullptr) {
      RETURN_IF_ERROR(model_->Detect(image, text, bounding_boxes, &result));
    } else {
      RETURN_IF_ERROR(
          model_->Detect(*options, image, text, bounding_boxes, &result));
    }
  }

  // NOTE: pybind11::bytes must be called within GIL.
  for (size_t i = 0; i < result.size(); i++) {
    auto& item = result[i];
    PyDetectResult res = std::make_tuple(
        item.cx, item.cy, item.w, item.h, pybind11::bytes(std::move(item.text)),
        item.score,
        std::make_tuple(item.mask.mask_height, item.mask.mask_width,
                        pybind11::bytes(std::move(item.mask.mask_values))));
    ret.push_back(res);
  }
  return ret;
}

absl::StatusOr<std::vector<std::pair<pybind11::bytes, double>>>
VisionModel::ImageToText(absl::string_view image_bytes, absl::string_view text,
                         const ModelOptions* options) const {
  if (!status_.ok()) return status_;
  std::vector<::sax::client::VisionModel::ScoredText> samples;
  {
    pybind11::gil_scoped_release release;
    if (options == nullptr) {
      RETURN_IF_ERROR(model_->ImageToText(image_bytes, text, &samples));
    } else {
      RETURN_IF_ERROR(
          model_->ImageToText(*options, image_bytes, text, &samples));
    }
  }

  // NOTE: pybind11::bytes must be called within GIL.
  std::vector<std::pair<pybind11::bytes, double>> result;
  for (size_t i = 0; i < samples.size(); i++) {
    auto& item = samples[i];
    result.push_back(
        std::make_pair(pybind11::bytes(std::move(item.text)), item.score));
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<pybind11::bytes, double>>>
VisionModel::VideoToText(const std::vector<absl::string_view>& image_frames,
                         absl::string_view text,
                         const ModelOptions* options) const {
  if (!status_.ok()) return status_;
  std::vector<::sax::client::VisionModel::ScoredText> samples;
  {
    pybind11::gil_scoped_release release;
    if (options == nullptr) {
      RETURN_IF_ERROR(model_->VideoToText(image_frames, text, &samples));
    } else {
      RETURN_IF_ERROR(
          model_->VideoToText(*options, image_frames, text, &samples));
    }
  }

  // NOTE: pybind11::bytes must be called within GIL.
  std::vector<std::pair<pybind11::bytes, double>> result;
  for (size_t i = 0; i < samples.size(); i++) {
    auto& item = samples[i];
    result.push_back(
        std::make_pair(pybind11::bytes(std::move(item.text)), item.score));
  }
  return result;
}

Model::Model(absl::string_view id, const Options* options) {
  status_ = ::sax::client::Model::Open(id, options, &base_);
}

Model::Model(absl::string_view id) {
  status_ = ::sax::client::Model::Open(id, nullptr, &base_);
}

Model::~Model() {
  if (base_) delete base_;
}

AudioModel Model::AM() { return AudioModel(base_, status_); }

LanguageModel Model::LM() { return LanguageModel(base_, status_); }

VisionModel Model::VM() { return VisionModel(base_, status_); }

CustomModel Model::CM() { return CustomModel(base_, status_); }

MultimodalModel Model::MM() { return MultimodalModel(base_, status_); }

void StartDebugPort(int port) { ::sax::client::StartDebugPort(port); }

absl::Status Publish(
    absl::string_view id, absl::string_view model_path,
    absl::string_view checkpoint_path, int num_replicas,
    std::optional<std::map<std::string, std::string>> overrides,
    const AdminOptions* options) {
  pybind11::gil_scoped_release release;
  AdminOptions default_options;
  if (options == nullptr) {
    options = &default_options;
  }
  if (!overrides) {
    overrides.emplace();
  }
  return ::sax::client::Publish(*options, id, model_path, checkpoint_path,
                                num_replicas, *overrides);
}

absl::Status Unpublish(absl::string_view id, const AdminOptions* options) {
  pybind11::gil_scoped_release release;
  if (options == nullptr) {
    return ::sax::client::Unpublish(id);
  } else {
    return ::sax::client::Unpublish(*options, id);
  }
}

absl::Status Update(absl::string_view id, absl::string_view model_path,
                    absl::string_view checkpoint_path, int num_replicas,
                    const AdminOptions* options) {
  pybind11::gil_scoped_release release;
  if (options == nullptr) {
    return ::sax::client::Update(id, model_path, checkpoint_path, num_replicas);
  } else {
    return ::sax::client::Update(*options, id, model_path, checkpoint_path,
                                 num_replicas);
  }
}

absl::StatusOr<PyListResult> List(absl::string_view id,
                                  const AdminOptions* options) {
  pybind11::gil_scoped_release release;
  ::sax::client::ModelDetail published_model;
  if (options == nullptr) {
    RETURN_IF_ERROR(::sax::client::List(id, &published_model));
  } else {
    RETURN_IF_ERROR(::sax::client::List(*options, id, &published_model));
  }
  return std::make_tuple(published_model.model, published_model.ckpt,
                         published_model.active_replicas);
}

absl::StatusOr<::sax::client::ModelDetail> ListDetail(
    absl::string_view id, const AdminOptions* options) {
  pybind11::gil_scoped_release release;
  ::sax::client::ModelDetail published_model;
  if (options == nullptr) {
    RETURN_IF_ERROR(::sax::client::List(id, &published_model));
  } else {
    RETURN_IF_ERROR(::sax::client::List(*options, id, &published_model));
  }
  return published_model;
}

absl::StatusOr<std::vector<std::string>> ListAll(absl::string_view id,
                                                 const AdminOptions* options) {
  pybind11::gil_scoped_release release;
  std::vector<std::string> published_models;
  if (options == nullptr) {
    RETURN_IF_ERROR(::sax::client::ListAll(id, &published_models));
  } else {
    RETURN_IF_ERROR(::sax::client::ListAll(*options, id, &published_models));
  }
  return published_models;
}

absl::Status WaitForReady(absl::string_view id, int num_replicas,
                          const AdminOptions* options) {
  pybind11::gil_scoped_release release;
  if (options == nullptr) {
    return ::sax::client::WaitForReady(id, num_replicas);
  } else {
    return ::sax::client::WaitForReady(*options, id, num_replicas);
  }
}

absl::StatusOr<std::vector<::sax::client::ModelServerTypeStat>> Stats(
    absl::string_view id, const AdminOptions* options) {
  pybind11::gil_scoped_release release;
  std::vector<::sax::client::ModelServerTypeStat> stats;
  if (options == nullptr) {
    RETURN_IF_ERROR(::sax::client::Stats(id, &stats));
  } else {
    RETURN_IF_ERROR(::sax::client::Stats(*options, id, &stats));
  }
  return stats;
}

}  // namespace pybind
}  // namespace client
}  // namespace sax
