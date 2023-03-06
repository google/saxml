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

#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "saxml/protobuf/admin.pb.h"
#include "saxml/protobuf/common.pb.h"

namespace sax {
namespace client {

// Options contains options for creating sax client.
struct Options {
  int num_conn;  // Perferred number of connections to sax backend.
};

// Model options.
class ModelOptions {
 public:
  ModelOptions() = default;
  ModelOptions(const ModelOptions&) = delete;
  ModelOptions& operator=(const ModelOptions&) = delete;

  // Set time out (in seconds) for the query.
  void SetTimeout(float value);
  // Get timeout value.
  float GetTimeout() const;

  void SetExtraInput(absl::string_view key, float value);
  void SetExtraInputTensor(
    absl::string_view key, const std::vector<float>& value);
  void ToProto(::sax::ExtraInputs* proto) const;

 private:
  std::map<std::string, float> kv_;  // key-value pair for extra input to model.
  // key-value pair for extra input tensors to model.
  std::map<std::string, std::vector<float>> kv_t_;
  float timeout_ = -1;  // Query timeout. Negative value means no timeout.
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

  // Custom model with string to string cutom call.
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
  //     scores[i] = items[i].score;
  //   }
  // }
  // absl::Status status = lm.GenerateStream(prefix, callback);
  // EXPECT_OK(status);
  struct GenerateItem {
    std::string text;
    int prefix_len;
    double score;
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

 private:
  explicit LanguageModel(int64_t model_handle) : model_handle_(model_handle) {}
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

  // Computes the embedding of the given image represented as encoded image
  // bytes.
  //
  // On success, returns OK and fills in embedding. Otherwise, returns an error.
  absl::Status Embed(absl::string_view image_bytes,
                     std::vector<double>* embedding) const;
  absl::Status Embed(const ModelOptions& options, absl::string_view image_bytes,
                     std::vector<double>* embedding) const;

  struct DetectResult {
    double cx;
    double cy;
    double w;
    double h;
    std::string text;
    double score;
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

 private:
  explicit Model(int64_t model_handle) : model_handle_(model_handle) {}
  int64_t model_handle_;
};

// Starts a debugging http server at the given `port`. For debugging only.
void StartDebugPort(int port);

// Publish a model with given parameters.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status Publish(absl::string_view id, absl::string_view model_path,
                     absl::string_view checkpoint_path, int num_replicas);

// Unpublish a model for a given id.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status Unpublish(absl::string_view id);

// Update a model with given parameters.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status Update(absl::string_view id, absl::string_view model_path,
                    absl::string_view checkpoint_path, int num_replicas);

struct ModelDetail {
  std::string model;
  std::string ckpt;
  int replicas;
};

// List a model to get details such checkpoint path and model path.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status List(absl::string_view id, ModelDetail* model);

// List all model ids in a sax cell.
//
// On success, returns OK; Otherwise, returns an error.
absl::Status ListAll(absl::string_view id, std::vector<std::string>* models);

}  // namespace client
}  // namespace sax
#endif  // SAXML_CLIENT_CC_SAX_H_
