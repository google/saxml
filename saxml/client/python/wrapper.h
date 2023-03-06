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

// C++ wrapper of sax client for Python.
//
// It modifies the interface of client facing C++ API for pybind11.
// it is NOT meant for direct client use.
#ifndef SAXML_CLIENT_PYTHON_WRAPPER_H_
#define SAXML_CLIENT_PYTHON_WRAPPER_H_

#include <string>
#include <tuple>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "saxml/client/cc/sax.h"
#include "pybind11/pytypes.h"

namespace sax {
namespace client {

class Options;
class ModelOptions;

namespace pybind {

class AudioModel {
 public:
  AudioModel() = delete;
  AudioModel(const AudioModel& obj);
  ~AudioModel();

  // Recognize produces text and scores given the input audio.
  //
  // It uses pair/tuple to avoid another wrapping for return type.
  absl::StatusOr<std::vector<std::pair<std::string, double>>> Recognize(
      absl::string_view audio_bytes,
      const ModelOptions* options = nullptr) const;

 private:
  explicit AudioModel(::sax::client::Model* base, const absl::Status& status);
  ::sax::client::Model* base_ = nullptr;
  ::sax::client::AudioModel* model_ = nullptr;
  absl::Status status_;
  friend class Model;
};

class CustomModel {
 public:
  CustomModel() = delete;
  CustomModel(const CustomModel& obj);
  ~CustomModel();

  // Custom model with string to string cutom call.
  absl::StatusOr<pybind11::bytes> Custom(
      pybind11::bytes request, absl::string_view method_name,
      const ModelOptions* options = nullptr) const;

 private:
  explicit CustomModel(::sax::client::Model* base, const absl::Status& status);
  ::sax::client::Model* base_ = nullptr;
  ::sax::client::CustomModel* model_ = nullptr;
  absl::Status status_;
  friend class Model;
};

class LanguageModel {
 public:
  LanguageModel() = delete;
  LanguageModel(const LanguageModel& obj);
  ~LanguageModel();

  // Scores the given 'prefix' and 'suffix' using the language model.
  absl::StatusOr<std::vector<double>> Score(
      absl::string_view prefix, std::vector<absl::string_view> suffix,
      const ModelOptions* options = nullptr) const;

  // Invokes the model to generate the suffix given the 'prefix'.
  //
  // It uses pair/tuple to avoid another wrapping for return type.
  absl::StatusOr<std::vector<std::pair<std::string, double>>> Generate(
      absl::string_view prefix, const ModelOptions* options = nullptr) const;

  // Invokes the model to generate a stream of suffixes given the 'prefix'.
  //
  // When `last` is true, this is the last callback invocation.
  // When `last` is false, `results` contains all decoding results so far.
  typedef std::function<void(
      bool last, std::vector<std::tuple<std::string, int, double>> results)>
      GenerateCallback;
  absl::Status GenerateStream(absl::string_view prefix,
                              GenerateCallback callback,
                              const ModelOptions* options = nullptr) const;

  // Run embedding on the given text.
  absl::StatusOr<std::vector<double>> Embed(
      absl::string_view text, const ModelOptions* options = nullptr) const;

 private:
  explicit LanguageModel(::sax::client::Model* base,
                         const absl::Status& status);
  ::sax::client::Model* base_ = nullptr;
  ::sax::client::LanguageModel* model_ = nullptr;
  absl::Status status_;
  friend class Model;
};

class VisionModel {
 public:
  VisionModel() = delete;
  VisionModel(const VisionModel& obj);
  ~VisionModel();

  // Classify produces suffixes and scores given the image bytes.
  //
  // It uses pair/tuple to avoid another wrapping for return type.
  absl::StatusOr<std::vector<std::pair<std::string, double>>> Classify(
      absl::string_view image_bytes,
      const ModelOptions* options = nullptr) const;

  // TextToImage produces a list of images and scores given the 'text'.
  //
  // It uses pair/tuple to avoid another wrapping for return type.
  absl::StatusOr<std::vector<std::pair<pybind11::bytes, double>>> TextToImage(
      absl::string_view text, const ModelOptions* options = nullptr) const;

  // Run embedding on the given image.
  absl::StatusOr<std::vector<double>> Embed(
      absl::string_view image, const ModelOptions* options = nullptr) const;

  typedef std::tuple<double, double, double, double, pybind11::bytes, double>
      PyDetectResult;

  // Run detection on the given image.
  //
  // For open-set detection models, one can pass text lists as the second
  // argument.
  //
  // Returns a vector of bounding boxes as a tuple <cx, cy, w, h, text, score>.
  absl::StatusOr<std::vector<PyDetectResult>> Detect(
      absl::string_view image_bytes, std::vector<std::string> text = {},
      const ModelOptions* options = nullptr) const;

  // ImageToText produces captions given the image bytes and prefix text.
  //
  // Returns a vector of tuples <text, score>.
  absl::StatusOr<std::vector<std::pair<pybind11::bytes, double>>> ImageToText(
      absl::string_view image_bytes, absl::string_view text = "",
      const ModelOptions* options = nullptr) const;

  // VideoToText produces a list of captions and scores given 'image_frames'
  // and an optional prefix 'text'
  //
  // Returns a vector of tuples <text, score>.
  absl::StatusOr<std::vector<std::pair<pybind11::bytes, double>>> VideoToText(
      const std::vector<absl::string_view>& image_frames,
      absl::string_view text = "", const ModelOptions* options = nullptr) const;

 private:
  explicit VisionModel(::sax::client::Model* base, const absl::Status& status);
  ::sax::client::Model* base_ = nullptr;
  ::sax::client::VisionModel* model_ = nullptr;
  absl::Status status_;
  friend class Model;
};

class Model {
 public:
  explicit Model(absl::string_view id, const Options* options);
  explicit Model(absl::string_view id);
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;
  ~Model();

  AudioModel AM();
  LanguageModel LM();
  VisionModel VM();
  CustomModel CM();

 private:
  ::sax::client::Model* base_ = nullptr;
  absl::Status status_;
};

void StartDebugPort(int port);

absl::Status Publish(absl::string_view id, absl::string_view model_path,
                     absl::string_view checkpoint_path, int num_replicas);

absl::Status Unpublish(absl::string_view id);

absl::Status Update(absl::string_view id, absl::string_view model_path,
                    absl::string_view checkpoint_path, int num_replicas);

typedef std::tuple<std::string, std::string, int> PyListResult;

absl::StatusOr<PyListResult> List(absl::string_view id);

absl::StatusOr<std::vector<std::string>> ListAll(absl::string_view id);

}  // namespace pybind
}  // namespace client
}  // namespace sax

#endif  // SAXML_CLIENT_PYTHON_WRAPPER_H_
