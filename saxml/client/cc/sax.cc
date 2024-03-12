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

#include "saxml/client/cc/sax.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "saxml/client/cc/saxwrapper.h"
#include "saxml/protobuf/admin.pb.h"
#include "saxml/protobuf/audio.pb.h"
#include "saxml/protobuf/common.pb.h"
#include "saxml/protobuf/custom.pb.h"
#include "saxml/protobuf/lm.pb.h"
#include "saxml/protobuf/multimodal.pb.h"
#include "saxml/protobuf/vision.pb.h"

namespace sax {
namespace client {

using ::sax::ExtraInputs;
using ::sax::Tensor;
using ::sax::server::lm::GenerateResponse;
using ::sax::server::lm::GenerateStreamResponse;
using ::sax::server::lm::GradientRequest;
using ::sax::server::lm::GradientResponse;
using ::sax::server::lm::ScoreRequest;
using ::sax::server::lm::ScoreResponse;
using LmEmbedResponse = ::sax::server::lm::EmbedResponse;
using ::sax::server::custom::CustomResponse;
using ::sax::server::vision::ClassifyResponse;
using ::sax::server::vision::DetectRequest;
using ::sax::server::vision::DetectResponse;
using ::sax::server::vision::ImageToImageResponse;
using ::sax::server::vision::ImageToTextResponse;
using ::sax::server::vision::TextAndImageToImageResponse;
using ::sax::server::vision::TextToImageResponse;
using ::sax::server::vision::VideoToTextResponse;
using VmEmbedResponse = ::sax::server::vision::EmbedResponse;
using AsrResponse = ::sax::server::audio::AsrResponse;

namespace {

absl::Status CreateErrorAndFree(int error_code, char* errMsgStr) {
  auto status = absl::Status(absl::StatusCode(error_code),
                             errMsgStr ? std::string(errMsgStr) : "");
  if (errMsgStr != nullptr) {
    free(errMsgStr);
  }
  return status;
}

int* GetQueryCostTpuMs(const ModelOptions& opts) {
  return opts.GetQueryCost() ? &opts.GetQueryCost()->tpu_ms : nullptr;
}

}  // namespace

void ModelOptions::SetTimeout(float value) { timeout_ = value; }

float ModelOptions::GetTimeout() const { return timeout_; }

void ModelOptions::SetExtraInput(absl::string_view key, float value) {
  kv_[std::string(key)] = value;
}

void ModelOptions::SetExtraInputTensor(absl::string_view key,
                                       const std::vector<float>& value) {
  kv_t_[std::string(key)] = value;
}

void ModelOptions::SetExtraInputString(absl::string_view key,
                                       std::string value) {
  kv_s_[std::string(key)] = value;
}

void ModelOptions::ToProto(ExtraInputs* proto) const {
  for (auto const& option : kv_) {
    (*proto->mutable_items())[option.first] = option.second;
  }
  for (auto const& option : kv_t_) {
    Tensor tensor;
    tensor.mutable_values()->Assign(option.second.begin(), option.second.end());
    (*proto->mutable_tensors())[option.first] = tensor;
  }
  for (auto const& option : kv_s_) {
    (*proto->mutable_strings())[option.first] = option.second;
  }
}

float ModelOptions::GetExtraInput(absl::string_view key) const {
  return kv_.at(std::string(key));
}

std::vector<float> ModelOptions::GetExtraInputTensor(
    absl::string_view key) const {
  return kv_t_.at(std::string(key));
}

std::string ModelOptions::GetExtraInputString(absl::string_view key) const {
  return kv_s_.at(std::string(key));
}

void ModelOptions::FromProto(const ExtraInputs& proto) {
  for (auto const& option : proto.items()) {
    kv_[option.first] = option.second;
  }
  for (auto const& option : proto.tensors()) {
    kv_t_[option.first].assign(option.second.values().begin(),
                               option.second.values().end());
  }
  for (auto const& option : proto.strings()) {
    kv_s_[option.first] = option.second;
  }
}

void ModelOptions::SetQueryCost(QueryCost* query_cost) {
  query_cost_ = query_cost;
}

QueryCost* ModelOptions::GetQueryCost() const { return query_cost_; }

absl::Status AudioModel::Recognize(absl::string_view audio_bytes,
                                   std::vector<AsrHyp>* result) const {
  return AudioModel::Recognize(ModelOptions(), audio_bytes, result);
}

absl::Status AudioModel::Recognize(const ModelOptions& options,
                                   absl::string_view audio_bytes,
                                   std::vector<AsrHyp>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);
  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_recognize(model_handle_, options.GetTimeout(),
               const_cast<char*>(audio_bytes.data()), audio_bytes.size(),
               const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
               &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  AsrResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.hyps()) {
    result->push_back(AsrHyp{res.text(), res.score()});
  }
  return absl::OkStatus();
}

AudioModel::~AudioModel() { go_release_model(model_handle_); }

absl::Status CustomModel::Custom(absl::string_view request,
                                 absl::string_view method_name,
                                 std::string* result) const {
  return CustomModel::Custom(ModelOptions(), request, method_name, result);
}

absl::Status CustomModel::Custom(const ModelOptions& options,
                                 absl::string_view request,
                                 absl::string_view method_name,
                                 std::string* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_custom(model_handle_, options.GetTimeout(),
            const_cast<char*>(request.data()), request.size(),
            const_cast<char*>(method_name.data()), method_name.size(),
            const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
            &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  CustomResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }

  *result = output.response();
  return absl::OkStatus();
}

CustomModel::~CustomModel() { go_release_model(model_handle_); }

absl::Status LanguageModel::Score(absl::string_view prefix,
                                  std::vector<absl::string_view> suffix,
                                  std::vector<double>* log_p) const {
  return LanguageModel::Score(ModelOptions(), prefix, suffix, log_p);
}

absl::Status LanguageModel::Score(const ModelOptions& options,
                                  absl::string_view prefix,
                                  std::vector<absl::string_view> suffixes,
                                  std::vector<double>* log_p) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);
  // Fill score request with prefix and suffixes.
  ScoreRequest score_request;
  score_request.set_prefix(std::string(prefix));
  for (const auto& suffix : suffixes) {
    score_request.add_suffix(std::string(suffix));
  }
  std::string scoreReqStr = "";
  score_request.SerializeToString(&scoreReqStr);

  // Call score function.
  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_score(model_handle_, options.GetTimeout(),
           const_cast<char*>(scoreReqStr.data()), scoreReqStr.size(),
           const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
           &outputSize, GetQueryCostTpuMs(options), &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  ScoreResponse result;
  log_p->clear();
  if (outputStr != nullptr) {
    result.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : result.logp()) {
    log_p->push_back(res);
  }
  return absl::OkStatus();
}

absl::Status LanguageModel::Generate(absl::string_view prefix,
                                     std::vector<ScoredText>* result) const {
  return LanguageModel::Generate(ModelOptions(), prefix, result);
}

absl::Status LanguageModel::Generate(const ModelOptions& options,
                                     absl::string_view prefix,
                                     std::vector<ScoredText>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_generate(model_handle_, options.GetTimeout(),
              const_cast<char*>(prefix.data()), prefix.size(),
              const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
              &outputSize, GetQueryCostTpuMs(options), &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  GenerateResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.texts()) {
    result->push_back(ScoredText{res.text(), res.score()});
  }
  return absl::OkStatus();
}

absl::Status LanguageModel::GenerateStream(absl::string_view prefix,
                                           GenerateCallback cb) const {
  return LanguageModel::GenerateStream(ModelOptions(), prefix, cb);
}

namespace {

static void GenerateCallbackWrapper(void* cbCtx, void* outData, int outSize) {
  // Unwrap `cbCtx` to recover the original callback function.
  // Because capturing lambdas and member functions cannot be converted to
  // function pointers, we need to wrap the callback in a free function to use
  // the C API. `cbCtx` allows us to send through the original callback.
  auto cb = reinterpret_cast<LanguageModel::GenerateCallback*>(cbCtx);
  std::vector<LanguageModel::GenerateItem> items;

  // Check if this is the last call.
  if (outData == nullptr) {
    return (*cb)(/*last=*/true, items);
  }

  // For other calls, translate output to the format expected by the callback.
  GenerateStreamResponse out;
  out.ParseFromArray(outData, outSize);
  free(outData);
  for (const auto& item : out.items()) {
    auto generate_item = LanguageModel::GenerateItem{
        item.text(), item.prefix_len(),
        std::vector<double>{item.scores().begin(), item.scores().end()}};
    // Populate the score from deprecated `score` field if `scores` is empty.
    // TODO(b/286079331): Remove this fallback.
    if (generate_item.scores.empty()) {
      generate_item.scores.push_back(item.score());
    }
    items.push_back(generate_item);
  }
  return (*cb)(/*last=*/false, items);
}

}  // namespace

absl::Status LanguageModel::GenerateStream(const ModelOptions& options,
                                           absl::string_view prefix,
                                           GenerateCallback cb) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  // Wrap `cb` in an opaque context and give it to go_generate_stream.
  // go_generate_stream is responsible for calling GenerateCallbackWrapper
  // with the context.
  auto cbCtx = reinterpret_cast<void*>(&cb);
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_generate_stream(model_handle_, options.GetTimeout(),
                     const_cast<char*>(prefix.data()), prefix.size(),
                     const_cast<char*>(extraStr.data()), extraStr.size(),
                     GenerateCallbackWrapper, cbCtx, GetQueryCostTpuMs(options),
                     &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }
  return absl::OkStatus();
}

absl::Status LanguageModel::Embed(absl::string_view text,
                                  std::vector<double>* embedding) const {
  return LanguageModel::Embed(ModelOptions(), text, embedding);
}

absl::Status LanguageModel::Embed(const ModelOptions& options,
                                  absl::string_view text,
                                  std::vector<double>* embedding) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);
  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_lm_embed(model_handle_, options.GetTimeout(),
              const_cast<char*>(text.data()), text.size(),
              const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
              &outputSize, GetQueryCostTpuMs(options), &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  LmEmbedResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& value : output.embedding()) {
    embedding->push_back(value);
  }
  return absl::OkStatus();
}

absl::Status LanguageModel::Gradient(
    absl::string_view prefix, absl::string_view suffix,
    std::vector<double>* score,
    absl::flat_hash_map<std::string, std::vector<double>>* gradients) const {
  return LanguageModel::Gradient(ModelOptions(), prefix, suffix, score,
                                 gradients);
}

absl::Status LanguageModel::Gradient(
    const ModelOptions& options, absl::string_view prefix,
    absl::string_view suffix, std::vector<double>* score,
    absl::flat_hash_map<std::string, std::vector<double>>* gradients) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);
  // Fill gradient request with prefix and suffix.
  GradientRequest req;
  req.set_prefix(std::string(prefix));
  req.set_suffix(std::string(suffix));
  std::string reqStr = "";
  req.SerializeToString(&reqStr);

  // Call gradient function.
  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_gradient(model_handle_, options.GetTimeout(),
              const_cast<char*>(reqStr.data()), reqStr.size(),
              const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
              &outputSize, GetQueryCostTpuMs(options), &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  GradientResponse result;
  score->clear();
  gradients->clear();
  if (outputStr != nullptr) {
    result.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : result.score()) {
    score->push_back(res);
  }
  for (const auto& res : result.gradients()) {
    gradients->emplace(res.first,
                       std::vector<double>(res.second.values().begin(),
                                           res.second.values().end()));
  }
  return absl::OkStatus();
}

LanguageModel::~LanguageModel() { go_release_model(model_handle_); }

absl::Status MultimodalModel::Generate(
    const ::sax::server::multimodal::GenerateRequest& request,
    ::sax::server::multimodal::GenerateResponse* response) const {
  return MultimodalModel::Generate(ModelOptions(), request, response);
}

absl::Status MultimodalModel::Generate(
    const ModelOptions& options,
    const ::sax::server::multimodal::GenerateRequest& request,
    ::sax::server::multimodal::GenerateResponse* response) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  std::string reqStr = "";
  request.SerializeToString(&reqStr);

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_mm_generate(model_handle_, options.GetTimeout(),
                 const_cast<char*>(reqStr.data()), reqStr.size(),
                 const_cast<char*>(extraStr.data()), extraStr.size(),
                 &outputStr, &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  if (outputStr != nullptr) {
    response->ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  return absl::OkStatus();
}

absl::Status MultimodalModel::Score(
    const ::sax::server::multimodal::ScoreRequest& request,
    ::sax::server::multimodal::ScoreResponse* response) const {
  return MultimodalModel::Score(ModelOptions(), request, response);
}

absl::Status MultimodalModel::Score(
    const ModelOptions& options,
    const ::sax::server::multimodal::ScoreRequest& request,
    ::sax::server::multimodal::ScoreResponse* response) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  std::string reqStr = "";
  request.SerializeToString(&reqStr);

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_mm_score(model_handle_, options.GetTimeout(),
              const_cast<char*>(reqStr.data()), reqStr.size(),
              const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
              &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  if (outputStr != nullptr) {
    response->ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  return absl::OkStatus();
}

MultimodalModel::~MultimodalModel() { go_release_model(model_handle_); }

absl::Status VisionModel::Classify(absl::string_view text,
                                   std::vector<ScoredText>* result) const {
  return VisionModel::Classify(ModelOptions(), text, result);
}

absl::Status VisionModel::Classify(const ModelOptions& options,
                                   absl::string_view text,
                                   std::vector<ScoredText>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);
  char* outputStr = nullptr;
  int outputSize = 0;

  char* errMsgStr = nullptr;
  int errCode = 0;
  go_classify(model_handle_, options.GetTimeout(),
              const_cast<char*>(text.data()), text.size(),
              const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
              &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  ClassifyResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.texts()) {
    result->push_back(ScoredText{res.text(), res.score()});
  }
  return absl::OkStatus();
}

absl::Status VisionModel::TextToImage(
    absl::string_view text, std::vector<GeneratedImage>* result) const {
  return VisionModel::TextToImage(ModelOptions(), text, result);
}
absl::Status VisionModel::TextToImage(
    const ModelOptions& options, absl::string_view text,
    std::vector<GeneratedImage>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);
  char* outputStr = nullptr;
  int outputSize = 0;

  char* errMsgStr = nullptr;
  int errCode = 0;
  go_text_to_image(model_handle_, options.GetTimeout(),
                   const_cast<char*>(text.data()), text.size(),
                   const_cast<char*>(extraStr.data()), extraStr.size(),
                   &outputStr, &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  TextToImageResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.images()) {
    result->push_back(GeneratedImage{res.image(), res.score()});
  }
  return absl::OkStatus();
}

absl::Status VisionModel::TextAndImageToImage(
    absl::string_view text, absl::string_view image_bytes,
    std::vector<GeneratedImage>* result) const {
  return VisionModel::TextAndImageToImage(ModelOptions(), text, image_bytes,
                                          result);
}
absl::Status VisionModel::TextAndImageToImage(
    const ModelOptions& options, absl::string_view text,
    absl::string_view image_bytes, std::vector<GeneratedImage>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);
  char* outputStr = nullptr;
  int outputSize = 0;

  char* errMsgStr = nullptr;
  int errCode = 0;
  go_text_and_image_to_image(
      model_handle_, options.GetTimeout(), const_cast<char*>(text.data()),
      text.size(), const_cast<char*>(image_bytes.data()), image_bytes.size(),
      const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
      &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  TextAndImageToImageResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.images()) {
    result->push_back(GeneratedImage{res.image(), res.score()});
  }
  return absl::OkStatus();
}

absl::Status VisionModel::Embed(absl::string_view image_bytes,
                                std::vector<double>* embedding) const {
  return VisionModel::Embed(ModelOptions(), image_bytes, embedding);
}

absl::Status VisionModel::Embed(const ModelOptions& options,
                                absl::string_view image_bytes,
                                std::vector<double>* embedding) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);
  char* outputStr = nullptr;
  int outputSize = 0;

  char* errMsgStr = nullptr;
  int errCode = 0;
  go_vm_embed(model_handle_, options.GetTimeout(),
              const_cast<char*>(image_bytes.data()), image_bytes.size(),
              const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
              &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  VmEmbedResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& value : output.embedding()) {
    embedding->push_back(value);
  }
  return absl::OkStatus();
}

absl::Status VisionModel::Detect(absl::string_view image_bytes,
                                 const std::vector<std::string>& text,
                                 std::vector<DetectResult>* result) const {
  return VisionModel::Detect(ModelOptions(), image_bytes, text, result);
}

absl::Status VisionModel::Detect(absl::string_view image_bytes,
                                 const std::vector<std::string>& text,
                                 const std::vector<BoundingBox>& boxes,
                                 std::vector<DetectResult>* result) const {
  return VisionModel::Detect(ModelOptions(), image_bytes, text, boxes, result);
}

absl::Status VisionModel::Detect(const ModelOptions& options,
                                 absl::string_view image_bytes,
                                 const std::vector<std::string>& text,
                                 std::vector<DetectResult>* result) const {
  return VisionModel::Detect(options, image_bytes, text,
                             std::vector<BoundingBox>(), result);
}

absl::Status VisionModel::Detect(const ModelOptions& options,
                                 absl::string_view image_bytes,
                                 const std::vector<std::string>& text,
                                 const std::vector<BoundingBox>& boxes,
                                 std::vector<DetectResult>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  DetectRequest detect_request;
  detect_request.mutable_text()->Reserve(text.size());
  for (const auto& one_text : text) {
    detect_request.add_text(one_text);
  }
  detect_request.mutable_boxes_of_interest()->Reserve(boxes.size());
  for (const auto& box : boxes) {
    ::sax::server::vision::BoundingBox box_of_interest;
    box_of_interest.set_cx(box.cx);
    box_of_interest.set_cy(box.cy);
    box_of_interest.set_w(box.w);
    box_of_interest.set_h(box.h);
    detect_request.mutable_boxes_of_interest()->Add(std::move(box_of_interest));
  }
  std::string detectReqStr = "";
  detect_request.SerializeToString(&detectReqStr);

  char* outputStr = nullptr;
  int outputSize = 0;

  char* errMsgStr = nullptr;
  int errCode = 0;
  go_vm_detect(model_handle_, options.GetTimeout(),
               const_cast<char*>(image_bytes.data()), image_bytes.size(),
               const_cast<char*>(detectReqStr.data()), detectReqStr.size(),
               const_cast<char*>(extraStr.data()), extraStr.size(), &outputStr,
               &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  // Proto of repeated bounding boxes.
  DetectResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.bounding_boxes()) {
    result->push_back(DetectResult{
        res.cx(), res.cy(), res.w(), res.h(), res.text(), res.score(),
        DetectionMask{res.mask().mask_height(), res.mask().mask_width(),
                      res.mask().mask_values()}});
  }
  return absl::OkStatus();
}

absl::Status VisionModel::ImageToText(absl::string_view image_bytes,
                                      absl::string_view text,
                                      std::vector<ScoredText>* result) const {
  return VisionModel::ImageToText(ModelOptions(), image_bytes, text, result);
}

absl::Status VisionModel::ImageToText(const ModelOptions& options,
                                      absl::string_view image_bytes,
                                      absl::string_view text,
                                      std::vector<ScoredText>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_vm_image_to_text(model_handle_, options.GetTimeout(),
                      const_cast<char*>(image_bytes.data()), image_bytes.size(),
                      const_cast<char*>(text.data()), text.size(),
                      const_cast<char*>(extraStr.data()), extraStr.size(),
                      &outputStr, &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  ImageToTextResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.texts()) {
    result->push_back(ScoredText{res.text(), res.score()});
  }
  return absl::OkStatus();
}

absl::Status VisionModel::ImageToImage(
    absl::string_view image_bytes, std::vector<GeneratedImage>* result) const {
  return VisionModel::ImageToImage(ModelOptions(), image_bytes, result);
}

absl::Status VisionModel::ImageToImage(
    const ModelOptions& options, absl::string_view image_bytes,
    std::vector<GeneratedImage>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_vm_image_to_image(model_handle_, options.GetTimeout(),
                       const_cast<char*>(image_bytes.data()),
                       image_bytes.size(), const_cast<char*>(extraStr.data()),
                       extraStr.size(), &outputStr, &outputSize, &errMsgStr,
                       &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  ImageToImageResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.images()) {
    result->push_back(GeneratedImage{res.image(), res.score()});
  }
  return absl::OkStatus();
}

absl::Status VisionModel::VideoToText(
    const std::vector<absl::string_view>& image_frames, absl::string_view text,
    std::vector<ScoredText>* result) const {
  return VisionModel::VideoToText(ModelOptions(), image_frames, text, result);
}

absl::Status VisionModel::VideoToText(
    const ModelOptions& options,
    const std::vector<absl::string_view>& image_frames, absl::string_view text,
    std::vector<ScoredText>* result) const {
  ExtraInputs extra;
  options.ToProto(&extra);
  std::string extraStr = "";
  extra.SerializeToString(&extraStr);

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  std::vector<int> frame_sizes;
  std::vector<char*> frame_buffers;
  for (const auto& frame : image_frames) {
    frame_sizes.push_back(frame.size());
    frame_buffers.push_back(const_cast<char*>(frame.data()));
  }
  go_vm_video_to_text(model_handle_, options.GetTimeout(),
                      const_cast<char**>(frame_buffers.data()),
                      const_cast<int*>(frame_sizes.data()), image_frames.size(),
                      const_cast<char*>(text.data()), text.size(),
                      const_cast<char*>(extraStr.data()), extraStr.size(),
                      &outputStr, &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  VideoToTextResponse output;
  if (outputStr != nullptr) {
    output.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  for (const auto& res : output.texts()) {
    result->push_back(ScoredText{res.text(), res.score()});
  }
  return absl::OkStatus();
}

VisionModel::~VisionModel() { go_release_model(model_handle_); }

absl::Status Model::Open(absl::string_view id, const Options* options,
                         Model** model) {
  int64_t base_model = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  // Use two call pathes since cgo does not support variadic functions.
  if (options == nullptr) {
    // Use default values when options are not set.
    go_create_model(const_cast<char*>(id.data()), id.size(), &base_model,
                    &errMsgStr, &errCode);
  } else {
    // Otherwise, pass options to Go.
    // Cannot pass sax.Options directly so flatten it here.
    go_create_model_with_config(const_cast<char*>(id.data()), id.size(),
                                options->num_conn,
                                const_cast<char*>(options->proxy_addr.data()),
                                options->proxy_addr.size(), options->fail_fast,
                                &base_model, &errMsgStr, &errCode);
  }
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  *model = new Model(base_model);
  return absl::OkStatus();
}

absl::Status Model::Open(absl::string_view id, Model** model) {
  return Model::Open(id, nullptr, model);
}

Model::~Model() { go_release_model(model_handle_); }

AudioModel* Model::AM() {
  int64_t handle_;
  go_create_am(model_handle_, &handle_);
  return new AudioModel(handle_);
}

CustomModel* Model::CM() {
  int64_t handle_;
  go_create_cm(model_handle_, &handle_);
  return new CustomModel(handle_);
}

VisionModel* Model::VM() {
  int64_t handle_;
  go_create_vm(model_handle_, &handle_);
  return new VisionModel(handle_);
}

LanguageModel* Model::LM() {
  int64_t handle_;
  go_create_lm(model_handle_, &handle_);
  return new LanguageModel(handle_);
}

MultimodalModel* Model::MM() {
  int64_t handle_;
  go_create_mm(model_handle_, &handle_);
  return new MultimodalModel(handle_);
}

void StartDebugPort(int port) { go_start_debug(port); }

absl::Status Publish(absl::string_view id, absl::string_view model_path,
                     absl::string_view checkpoint_path, int num_replicas) {
  return Publish(AdminOptions(), id, model_path, checkpoint_path, num_replicas,
                 {});
}

absl::Status Publish(const AdminOptions& options, absl::string_view id,
                     absl::string_view model_path,
                     absl::string_view checkpoint_path, int num_replicas,
                     const std::map<std::string, std::string>& overrides) {
  char* errMsgStr = nullptr;
  int errCode = 0;

  std::vector<char*> override_keys;
  std::vector<int> override_key_sizes;
  std::vector<char*> override_values;
  std::vector<int> override_value_sizes;

  const int num_overrides = overrides.size();
  for (const auto& [key, value] : overrides) {
    override_keys.push_back(const_cast<char*>(key.data()));
    override_key_sizes.push_back(key.size());
    override_values.push_back(const_cast<char*>(value.data()));
    override_value_sizes.push_back(value.size());
  }

  go_publish(const_cast<char*>(id.data()), id.size(),
             const_cast<char*>(model_path.data()), model_path.size(),
             const_cast<char*>(checkpoint_path.data()), checkpoint_path.size(),
             num_replicas, options.timeout, num_overrides, override_keys.data(),
             override_key_sizes.data(), override_values.data(),
             override_value_sizes.data(), &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  return absl::OkStatus();
}

absl::Status Unpublish(absl::string_view id) {
  return Unpublish(AdminOptions(), id);
}

absl::Status Unpublish(const AdminOptions& options, absl::string_view id) {
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_unpublish(const_cast<char*>(id.data()), id.size(), options.timeout,
               &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  return absl::OkStatus();
}

absl::Status Update(absl::string_view id, absl::string_view model_path,
                    absl::string_view checkpoint_path, int num_replicas) {
  return Update(AdminOptions(), id, model_path, checkpoint_path, num_replicas);
}

absl::Status Update(const AdminOptions& options, absl::string_view id,
                    absl::string_view model_path,
                    absl::string_view checkpoint_path, int num_replicas) {
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_update(const_cast<char*>(id.data()), id.size(),
            const_cast<char*>(model_path.data()), model_path.size(),
            const_cast<char*>(checkpoint_path.data()), checkpoint_path.size(),
            num_replicas, options.timeout, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  return absl::OkStatus();
}

absl::Status List(absl::string_view id, ModelDetail* model) {
  return List(AdminOptions(), id, model);
}

absl::Status List(const AdminOptions& options, absl::string_view id,
                  ModelDetail* model) {
  std::string content;

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_list(const_cast<char*>(id.data()), id.size(), options.timeout, &outputStr,
          &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }
  sax::PublishedModel pub_model;
  if (outputStr != nullptr) {
    pub_model.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }
  auto one_model = pub_model.model();
  model->model = one_model.model_path();
  model->ckpt = one_model.checkpoint_path();
  model->max_replicas = one_model.requested_num_replicas();
  model->active_replicas = pub_model.modelet_addresses_size();
  model->overrides = std::map<std::string, std::string>(
      one_model.overrides().begin(), one_model.overrides().end());

  return absl::OkStatus();
}

absl::Status ListAll(absl::string_view id, std::vector<std::string>* models) {
  return ListAll(AdminOptions(), id, models);
}

absl::Status ListAll(const AdminOptions& options, absl::string_view id,
                     std::vector<std::string>* models) {
  std::string content;

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_list_all(const_cast<char*>(id.data()), id.size(), options.timeout,
              &outputStr, &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  sax::ListResponse resp;
  if (outputStr != nullptr) {
    resp.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }

  for (const auto& res : resp.published_models()) {
    models->push_back(res.model().model_id());
  }

  return absl::OkStatus();
}

absl::Status WaitForReady(absl::string_view id, int num_replicas) {
  return WaitForReady(AdminOptions(), id, num_replicas);
}

absl::Status WaitForReady(const AdminOptions& options, absl::string_view id,
                          int num_replicas) {
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_wait_for_ready(const_cast<char*>(id.data()), id.size(), num_replicas,
                    options.timeout, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  return absl::OkStatus();
}

absl::Status Stats(absl::string_view id,
                   std::vector<ModelServerTypeStat>* stats) {
  return Stats(AdminOptions(), id, stats);
}

absl::Status Stats(const AdminOptions& options, absl::string_view id,
                   std::vector<ModelServerTypeStat>* stats) {
  std::string content;

  char* outputStr = nullptr;
  int outputSize = 0;
  char* errMsgStr = nullptr;
  int errCode = 0;
  go_stats(const_cast<char*>(id.data()), id.size(), options.timeout, &outputStr,
           &outputSize, &errMsgStr, &errCode);
  if (errCode != 0) {
    return CreateErrorAndFree(errCode, errMsgStr);
  }

  sax::StatsResponse resp;
  if (outputStr != nullptr) {
    resp.ParseFromArray(outputStr, outputSize);
    free(outputStr);
  }

  for (const auto& stat : resp.model_server_type_stats()) {
    stats->push_back({sax::ModelServer::ChipType_Name(stat.chip_type()),
                      sax::ModelServer::ChipTopology_Name(stat.chip_topology()),
                      stat.num_replicas()});
  }

  return absl::OkStatus();
}

}  // namespace client
}  // namespace sax
