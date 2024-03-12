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

#ifndef SAXML_CLIENT_CC_SAX_H_
#define SAXML_CLIENT_CC_SAX_H_

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "saxml/protobuf/common.pb.h"
#include "saxml/protobuf/multimodal.pb.h"

namespace sax {
namespace client {

// Options contains options for creating sax client.
struct Options {
  int num_conn = 37;  // Perferred number of connections to sax backend.
  std::string proxy_addr = "";  // Optional proxy address.
  // Whether the model should fail fast instead of waiting for servers to be
  // available.
  bool fail_fast = false;
};

// QueryCost represents the query cost in TPU milliseconds
struct QueryCost {
  int tpu_ms;
};

// Model options.
class ModelOptions {
 public:
  ModelOptions() = default;
  ModelOptions(const ModelOptions&) = default;
  ModelOptions& operator=(const ModelOptions&) = delete;

  // Set time out (in seconds) for the query.
  void SetTimeout(float value);
  // Get timeout value.
  float GetTimeout() const;

  void SetExtraInput(absl::string_view key, float value);
  float GetExtraInput(absl::string_view key) const;

  void SetExtraInputTensor(absl::string_view key,
                           const std::vector<float>& value);
  std::vector<float> GetExtraInputTensor(absl::string_view key) const;

  void SetExtraInputString(absl::string_view key, std::string value);
  std::string GetExtraInputString(absl::string_view key) const;

  void FromProto(const ::sax::ExtraInputs& proto);
  void ToProto(::sax::ExtraInputs* proto) const;

  void SetQueryCost(QueryCost* query_cost);
  QueryCost* GetQueryCost() const;

 private:
  std::map<std::string, float> kv_;  // key-value pair for extra input to model.
  // key-value pair for extra input tensors to model.
  std::map<std::string, std::vector<float>> kv_t_;
  // key-value pair for extra input strings to model.
  std::map<std::string, std::string> kv_s_;
  float timeout_ = -1;  // Query timeout. Negative value means no timeout.

  // Cost of the query. Pointer is not owned.
  QueryCost* query_cost_ = nullptr;
};

// AudioModel provides common audio model API against a given model in
// the sax cluster.
//
// Example usage:
//    Model* m;
//    Status status = Model::Open("/sax/foo/am", &m);
//    if (!status.ok()) {
//       return status; // or crash
//    }
//    AudioModel* am = m->AM();
//    std::vector<AsrHyp> result;
//    status = am->Recognize(fname, &result);
//    if (status.ok()) {
//       // Use result.
//    } else {
//       // Handle error.
//    }
//    delete am;
//    delete m;
//
// Public methods are thread safe.
class AudioModel {
 public:
  AudioModel(const AudioModel&) = delete;
  AudioModel(AudioModel&&) = default;
  AudioModel& operator=(const AudioModel&) = delete;
  AudioModel& operator=(AudioModel&&) = default;
  ~AudioModel();

  // Recognize produces text and scores given the audio input.
  //
  // On success, returns OK and fills in text and their scores computed by
  // the model. Otherwise, returns an error.
  struct AsrHyp {
    std::string text;
    double score;
  };
  absl::Status Recognize(absl::string_view audio_bytes,
                         std::vector<AsrHyp>* result) const;
  absl::Status Recognize(const ModelOptions& options,
                         absl::string_view audio_bytes,
                         std::vector<AsrHyp>* result) const;

 private:
  explicit AudioModel(int64_t model_handle) : model_handle_(model_handle) {}
  friend class Model;
  int64_t model_handle_;
};

// CustomModel provides common custom model API against a given model in
// the sax cluster.
//
// Example usage:
//    Model* m;
//    Status status = Model::Open("/sax/foo/custom", &m);
//    if (!status.ok()) {
//       return status; // or crash
//    }
//    CustomModel* cusom = m->CM();
//    std::string result;
//    status = custom->Custom(request, method_name, &result);
//    if (status.ok()) {
//       // Use result.
//    } else {
//       // Handle error.
//    }
//    delete custom;
//    delete m;
//
// Public methods are thread safe.
class CustomModel {
 public:
  CustomModel(const CustomModel&) = delete;
  CustomModel(CustomModel&&) = default;
  CustomModel& operator=(const CustomModel&) = delete;
  CustomModel& operator=(CustomModel&&) = default;
  ~CustomModel();

  // Custom model with string to string custom call.
  //
  // On success, returns OK and fills in response computed by
  // the model. Otherwise, returns an error.
  absl::Status Custom(absl::string_view request, absl::string_view method_name,
                      std::string* result) const;
  absl::Status Custom(const ModelOptions& options, absl::string_view request,
                      absl::string_view method_name, std::string* result) const;

 private:
  explicit CustomModel(int64_t model_handle) : model_handle_(model_handle) {}
  friend class Model;
  int64_t model_handle_;
};

// LanguageModel provides common language model API against a given model in
// the sax cluster.
//
// Example usage:
//    Model* m;
//    Status status = Model::Open("/sax/foo/lm", &m);
//    if (!status.ok()) {
//       return status; // or crash
//    }
//    LanguageModel* lm = m->LM();
//    double log_p;
//    status = lm->Score("my prefix", "suffix", &log_p);
//    if (status.ok()) {
//       // Use result.
//    } else {
//       // Handle error.
//    }
//    delete lm;
//    delete m;
//
// Public methods are thread safe.
class LanguageModel {
 public:
  LanguageModel(const LanguageModel&) = delete;
  LanguageModel(LanguageModel&&) = default;
  LanguageModel& operator=(const LanguageModel&) = delete;
  LanguageModel& operator=(LanguageModel&&) = default;
  ~LanguageModel();

  // Scores the given 'prefix' and `suffix` using the language model.
  //
  // On success, returns OK and fills in 'score' computed by the model.
  // Otherwise, returns an error.
  absl::Status Score(absl::string_view prefix,
                     std::vector<absl::string_view> suffixes,
                     std::vector<double>* log_p) const;
  absl::Status Score(const ModelOptions& options, absl::string_view prefix,
                     std::vector<absl::string_view> suffixes,
                     std::vector<double>* log_p) const;

  // Samples the model and produces suffixes given the 'prefix'.
  //
  // On success, returns OK and fills in suffixes and their scores computed by
  // the model. Otherwise, returns an error.
  struct ScoredText {
    std::string suffix;
    double score;
  };
  absl::Status Generate(absl::string_view prefix,
                        std::vector<ScoredText>* result) const;
  absl::Status Generate(const ModelOptions& options, absl::string_view prefix,
                        std::vector<ScoredText>* result) const;

  // Samples the model and produces a stream of suffixes given the 'prefix'.
  //
  // This is a blocking call. The calling thread will invoke the given callback
  // multiple times during streaming. This call returns when streaming ends,
  // either successfully or due to error, indicated by the returned status.
  //
  // When the callback is invoked, `last` indicates if this is the last call.
  // `items` is valid iif `last` is false. The callback is called only when
  // no error is encountered.
  //
  // Example:
  //
  // std::vector<std::string> texts;
  // std::vector<double> scores;
  // auto callback = [&texts, &scores](
  //                     bool last,
  //                     const std::vector<GenerateItem>& items) {
  //   if (last) {
  //     return;
  //   }
  //   texts.resize(items.size(), "");
  //   scores.resize(items.size(), 0.0);
  //   for (int i = 0; i < items.size(); i++) {
  //     texts[i] = texts[i].substr(0, items[i].prefix_len) + items[i].text;
  //     scores[i] = items[i].scores[0];
  //   }
  // }
  // absl::Status status = lm.GenerateStream(prefix, callback);
  // EXPECT_OK(status);
  struct GenerateItem {
    std::string text;
    int prefix_len;
    std::vector<double> scores;
  };
  typedef std::function<void(bool last, const std::vector<GenerateItem>& items)>
      GenerateCallback;
  absl::Status GenerateStream(absl::string_view prefix,
                              GenerateCallback cb) const;
  absl::Status GenerateStream(const ModelOptions& options,
                              absl::string_view prefix,
                              GenerateCallback cb) const;

  // Computes the embedding of the given text.
  //
  // On success, returns OK and fills in embedding. Otherwise, returns an error.
  absl::Status Embed(absl::string_view text,
                     std::vector<double>* embedding) const;
  absl::Status Embed(const ModelOptions& options, absl::string_view text,
                     std::vector<double>* embedding) const;

  // Computes scores and gradients for a given 'prefix' and 'suffix' using the
  // language model.
  //
  // On success, returns OK and fills in scores and gradients. Otherwise,
  // returns an error.
  absl::Status Gradient(
      absl::string_view prefix, absl::string_view suffix,
      std::vector<double>* score,
      absl::flat_hash_map<std::string, std::vector<double>>* gradients) const;
  absl::Status Gradient(
      const ModelOptions& options, absl::string_view prefix,
      absl::string_view suffix, std::vector<double>* score,
      absl::flat_hash_map<std::string, std::vector<double>>* gradients) const;

 private:
  explicit LanguageModel(int64_t model_handle) : model_handle_(model_handle) {}
  friend class Model;
  int64_t model_handle_;
};

class MultimodalModel {
 public:
  MultimodalModel(const MultimodalModel&) = delete;
  MultimodalModel(MultimodalModel&&) = default;
  MultimodalModel& operator=(const MultimodalModel&) = delete;
  MultimodalModel& operator=(MultimodalModel&&) = default;
  ~MultimodalModel();

  // Samples the model and produces results given the
  // 'GenerateRequest.data_items'.
  //
  // On success, returns OK and fills in results and their scores computed by
  // the model. Otherwise, returns an error.
  absl::Status Generate(
      const ::sax::server::multimodal::GenerateRequest& request,
      ::sax::server::multimodal::GenerateResponse* response) const;
  absl::Status Generate(
      const ModelOptions& options,
      const ::sax::server::multimodal::GenerateRequest& request,
      ::sax::server::multimodal::GenerateResponse* response) const;

  // Scores the given 'ScoreRequest.prefix_items' and
  // 'ScoreRequest.suffix_items' using the multimodal model.
  //
  // On success, returns OK and fills in their scores computed by the model.
  // Otherwise, returns an error.
  absl::Status Score(const ::sax::server::multimodal::ScoreRequest& request,
                     ::sax::server::multimodal::ScoreResponse* response) const;
  absl::Status Score(const ModelOptions& options,
                     const ::sax::server::multimodal::ScoreRequest& request,
                     ::sax::server::multimodal::ScoreResponse* response) const;

 private:
  explicit MultimodalModel(int64_t model_handle)
      : model_handle_(model_handle) {}
  friend class Model;
  int64_t model_handle_;
};

// VisionModel provides common vision model API against a given model in
// the sax cluster.
//
// Example usage:
//    Model* m;
//    Status status = Model::Open("/sax/foo/vm", &m);
//    if (!status.ok()) {
//       return status; // or crash
//    }
//    VisionModel* vm = m->VM();
//    std::vector<ScoredText> result;
//    status = vm->Classify("my prefix", "suffix", &result);
//    if (status.ok()) {
//       // Use result.
//    } else {
//       // Handle error.
//    }
//    delete vm;
//    delete m;
//
// Public methods are thread safe.
class VisionModel {
 public:
  VisionModel(const VisionModel&) = delete;
  VisionModel(VisionModel&&) = default;
  VisionModel& operator=(const VisionModel&) = delete;
  VisionModel& operator=(VisionModel&&) = default;
  ~VisionModel();

  // Classify produces text and scores given the input image.
  //
  // On success, returns OK and fills in suffixes and their scores computed by
  // the model. Otherwise, returns an error.
  struct ScoredText {
    std::string text;
    double score;
  };
  absl::Status Classify(absl::string_view text,
                        std::vector<ScoredText>* result) const;
  absl::Status Classify(const ModelOptions& options, absl::string_view text,
                        std::vector<ScoredText>* result) const;

  // TextToImage produces a list of images and scores given the 'text'.
  // TextAndImageToImage produces a list of images and scores given the 'text'
  // and the 'image'.
  //
  // On success, returns OK and fills in images and their scores computed by
  // the model. Otherwise, returns an error.
  struct GeneratedImage {
    std::string image;
    double score;
  };
  absl::Status TextToImage(absl::string_view text,
                           std::vector<GeneratedImage>* result) const;
  absl::Status TextToImage(const ModelOptions& options, absl::string_view text,
                           std::vector<GeneratedImage>* result) const;
  absl::Status TextAndImageToImage(absl::string_view text,
                                   absl::string_view image_bytes,
                                   std::vector<GeneratedImage>* result) const;
  absl::Status TextAndImageToImage(const ModelOptions& options,
                                   absl::string_view text,
                                   absl::string_view image_bytes,
                                   std::vector<GeneratedImage>* result) const;

  // Computes the embedding of the given image represented as encoded image
  // bytes.
  //
  // On success, returns OK and fills in embedding. Otherwise, returns an error.
  absl::Status Embed(absl::string_view image_bytes,
                     std::vector<double>* embedding) const;
  absl::Status Embed(const ModelOptions& options, absl::string_view image_bytes,
                     std::vector<double>* embedding) const;

  // The following structs mirror vision.proto's message types. Please refer to
  // comments there for their semantics.
  struct BoundingBox {
    double cx;  // Center of the box in x-axis.
    double cy;  // Center of the box in y-axis.
    double w;   // Width of the box.
    double h;   // Height of the box.
  };

  struct DetectionMask {
    int32_t mask_height;
    int32_t mask_width;
    std::string mask_values;
  };

  struct DetectResult {
    double cx;
    double cy;
    double w;
    double h;
    std::string text;
    double score;
    DetectionMask mask;
  };

  // Detect produces a list of bounding boxes given 'image_bytes'.
  //
  // For open-set detection models, one can pass text lists as the second
  // argument.
  //
  // On success, returns OK and fills in the bounding boxes computed by
  // the model.  Otherwise, returns an error.
  absl::Status Detect(absl::string_view image_bytes,
                      const std::vector<std::string>& text,
                      std::vector<DetectResult>* result) const;
  absl::Status Detect(const ModelOptions& options,
                      absl::string_view image_bytes,
                      const std::vector<std::string>& text,
                      std::vector<DetectResult>* result) const;
  absl::Status Detect(absl::string_view image_bytes,
                      const std::vector<std::string>& text,
                      const std::vector<BoundingBox>& boxes,
                      std::vector<DetectResult>* result) const;
  absl::Status Detect(const ModelOptions& options,
                      absl::string_view image_bytes,
                      const std::vector<std::string>& text,
                      const std::vector<BoundingBox>& boxes,
                      std::vector<DetectResult>* result) const;
  // ImageToText produces a list of captions and scores given 'image_bytes'
  // and an optional prefix 'text'
  //
  // On success, returns OK and fills in text and their scores computed by
  // the model.  Otherwise, returns an error.
  absl::Status ImageToText(absl::string_view image_bytes,
                           absl::string_view text,
                           std::vector<ScoredText>* result) const;
  absl::Status ImageToText(const ModelOptions& options,
                           absl::string_view image_bytes,
                           absl::string_view text,
                           std::vector<ScoredText>* result) const;

  // ImageToImage produces a list of images and scores given 'image_bytes'.
  //
  // On success, returns OK and fills in images and their scores computed by
  // the model.  Otherwise, returns an error.
  absl::Status ImageToImage(absl::string_view image_bytes,
                            std::vector<GeneratedImage>* result) const;
  absl::Status ImageToImage(const ModelOptions& options,
                            absl::string_view image_bytes,
                            std::vector<GeneratedImage>* result) const;

  // VideoToText produces a list of captions and scores given 'image_frames'
  // and an optional prefix 'text'
  //
  // On success, returns OK and fills in text and their scores computed by
  // the model.  Otherwise, returns an error.
  absl::Status VideoToText(const std::vector<absl::string_view>& image_frames,
                           absl::string_view text,
                           std::vector<ScoredText>* result) const;
  absl::Status VideoToText(const ModelOptions& options,
                           const std::vector<absl::string_view>& image_frames,
                           absl::string_view text,
                           std::vector<ScoredText>* result) const;

 private:
  explicit VisionModel(int64_t model_handle) : model_handle_(model_handle) {}
  friend class Model;
  int64_t model_handle_;
};

class Model {
 public:
  // Create a Model instance with options. User takes ownership of the resulting
  // object.
  static absl::Status Open(absl::string_view id, const Options* options,
                           Model** model);
  // Create a Model instance without options. User takes ownership of the
  // resulting object.
  static absl::Status Open(absl::string_view id, Model** model);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model();

  // Create an AudioModel instance. User takes ownership of the resulting
  // object. The lifetime of AudioModel is independent of Model.
  AudioModel* AM();

  // Create a LanguageModel instance. User takes ownership of the resulting
  // object. The lifetime of LanguageModel is independent of Model.
  LanguageModel* LM();

  // Create a VisionModel instance. User takes ownership of the resulting
  // object. The lifetime of VisionModel is independent of Model.
  VisionModel* VM();

  // Create a CustomModel instance. User takes ownership of the resulting
  // object. The lifetime of CUstomModel is independent of Model.
  CustomModel* CM();

  MultimodalModel* MM();

 private:
  explicit Model(int64_t model_handle) : model_handle_(model_handle) {}
  int64_t model_handle_;
};

// Starts a debugging http server at the given `port`. For debugging only.
void StartDebugPort(int port);

struct AdminOptions {
  // Timeout in seconds. Negative values indicate no timeout.
  float timeout = -1;
};

// Publish a model with given parameters.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status Publish(absl::string_view id, absl::string_view model_path,
                     absl::string_view checkpoint_path, int num_replicas);
absl::Status Publish(const AdminOptions& options, absl::string_view id,
                     absl::string_view model_path,
                     absl::string_view checkpoint_path, int num_replicas,
                     const std::map<std::string, std::string>& overrides);

// Unpublish a model for a given id.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status Unpublish(absl::string_view id);
absl::Status Unpublish(const AdminOptions& options, absl::string_view id);

// Update a model with given parameters.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status Update(absl::string_view id, absl::string_view model_path,
                    absl::string_view checkpoint_path, int num_replicas);
absl::Status Update(const AdminOptions& options, absl::string_view id,
                    absl::string_view model_path,
                    absl::string_view checkpoint_path, int num_replicas);

struct ModelDetail {
  std::string model;
  std::string ckpt;
  int max_replicas;
  int active_replicas;
  std::map<std::string, std::string> overrides;
};

// List a model to get details such checkpoint path and model path.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status List(absl::string_view id, ModelDetail* model);
absl::Status List(const AdminOptions& options, absl::string_view id,
                  ModelDetail* model);

// List all model ids in a sax cell.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status ListAll(absl::string_view id, std::vector<std::string>* models);
absl::Status ListAll(const AdminOptions& options, absl::string_view id,
                     std::vector<std::string>* models);

// Wait until at least a certain number of replicas are ready.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status WaitForReady(absl::string_view id, int num_replicas);
absl::Status WaitForReady(const AdminOptions& options, absl::string_view id,
                          int num_replicas);

struct ModelServerTypeStat {
  std::string chip_type;
  std::string chip_topology;
  int num_replicas;
};

// Gets stats of a cell.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status Stats(absl::string_view id,
                   std::vector<ModelServerTypeStat>* stats);
absl::Status Stats(const AdminOptions& options, absl::string_view id,
                   std::vector<ModelServerTypeStat>* stats);

}  // namespace client
}  // namespace sax
#endif  // SAXML_CLIENT_CC_SAX_H_
