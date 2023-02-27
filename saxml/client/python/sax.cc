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
#include "saxml/protobuf/common.pb.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11_abseil/absl_casters.h"    // IWYU pragma: keep
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep

namespace py = pybind11;

PYBIND11_MODULE(sax, m) {
  py::google::ImportStatusModule();
  py::class_<sax::client::Options>(m, "Options")
      .def(py::init<>())
      .def_readwrite("num_conn", &sax::client::Options::num_conn);

  py::class_<sax::client::ModelOptions>(m, "ModelOptions")
      .def(py::init<>())
      .def("SetExtraInput", &sax::client::ModelOptions::SetExtraInput)
      .def("SetExtraInputTensor",
           &sax::client::ModelOptions::SetExtraInputTensor)
      .def("SetTimeout", &sax::client::ModelOptions::SetTimeout)
      .def("ToDebugString", [](sax::client::ModelOptions& mo) {
        ::sax::ExtraInputs extra_inputs;
        mo.ToProto(&extra_inputs);
        return extra_inputs.DebugString();
      });

  py::class_<sax::client::pybind::AudioModel>(m, "AudioModel")
      .def("Recognize", &sax::client::pybind::AudioModel::Recognize,
           py::arg("id"), py::arg("options") = nullptr)
      .def(
          "Recognize",
          [](sax::client::pybind::AudioModel& am, absl::string_view audio_bytes,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::vector<std::pair<std::string, double>>> {
            return am.Recognize(audio_bytes, options);
          },
          py::arg("audio_bytes"), py::arg("options") = nullptr);

  py::class_<sax::client::pybind::CustomModel>(m, "CustomModel")
      .def("Custom", &sax::client::pybind::CustomModel::Custom,
           py::arg("request"), py::arg("method_name"),
           py::arg("options") = nullptr)
      .def(
          "Custom",
          [](sax::client::pybind::CustomModel& cm, py::bytes request,
             absl::string_view method_name,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<py::bytes> {
            return cm.Custom(request, method_name, options);
          },
          py::arg("request"), py::arg("method_name"),
          py::arg("options") = nullptr);

  py::class_<sax::client::pybind::LanguageModel>(m, "LanguageModel")
      .def(
          "Score",
          [](sax::client::pybind::LanguageModel& lm, absl::string_view prefix,
             std::vector<absl::string_view> suffix,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::vector<double>> {
            return lm.Score(prefix, suffix, options);
          },
          py::arg("prefix"), py::arg("suffix"), py::arg("options") = nullptr)
      .def(
          "Embed",
          [](sax::client::pybind::LanguageModel& lm, absl::string_view text,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::vector<double>> {
            return lm.Embed(text, options);
          },
          py::arg("text"), py::arg("options") = nullptr)
      .def(
          "Generate",
          [](sax::client::pybind::LanguageModel& lm, absl::string_view text,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::vector<std::pair<std::string, double>>> {
            return lm.Generate(text, options);
          },
          py::arg("text"), py::arg("options") = nullptr)
      .def(
          "GenerateStream",
          [](sax::client::pybind::LanguageModel& lm, absl::string_view text,
             py::function py_callback,
             const sax::client::ModelOptions* options) -> absl::Status {
            // py_callback: Callable[[bool, list[tuple[str, int, float]]], None]
            return lm.GenerateStream(text, py_callback, options);
          },
          py::arg("text"), py::arg("callback"), py::arg("options") = nullptr);

  py::class_<sax::client::pybind::VisionModel>(m, "VisionModel")
      .def(
          "Classify",
          [](sax::client::pybind::VisionModel& vm,
             absl::string_view image_bytes,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::vector<std::pair<std::string, double>>> {
            return vm.Classify(image_bytes, options);
          },
          py::arg("image_bytes"), py::arg("options") = nullptr)
      .def(
          "TextToImage",
          [](sax::client::pybind::VisionModel& vm, absl::string_view text,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<
                  std::vector<std::pair<pybind11::bytes, double>>> {
            return vm.TextToImage(text, options);
          },
          py::arg("text"), py::arg("options") = nullptr)
      .def(
          "Embed",
          [](sax::client::pybind::VisionModel& vm, absl::string_view image,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::vector<double>> {
            return vm.Embed(image, options);
          },
          py::arg("image"), py::arg("options") = nullptr)
      .def(
          "Detect",
          [](sax::client::pybind::VisionModel& vm,
             absl::string_view image_bytes, std::vector<std::string> text,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::vector<std::tuple<
                  double, double, double, double, pybind11::bytes, double>>> {
            return vm.Detect(image_bytes, text, options);
          },
          py::arg("image_bytes"), py::arg("text") = std::vector<std::string>{},
          py::arg("options") = nullptr)
      .def(
          "ImageToText",
          [](sax::client::pybind::VisionModel& vm,
             absl::string_view image_bytes, absl::string_view text,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<
                  std::vector<std::pair<pybind11::bytes, double>>> {
            return vm.ImageToText(image_bytes, text, options);
          },
          py::arg("image_bytes"), py::arg("text") = "",
          py::arg("options") = nullptr)
      .def(
          "VideoToText",
          [](sax::client::pybind::VisionModel& vm,
             const std::vector<absl::string_view>& image_frames,
             absl::string_view text, const sax::client::ModelOptions* options)
              -> absl::StatusOr<
                  std::vector<std::pair<pybind11::bytes, double>>> {
            return vm.VideoToText(image_frames, text, options);
          },
          py::arg("image_frames"), py::arg("text") = "",
          py::arg("options") = nullptr);

  py::class_<sax::client::pybind::Model>(m, "Model")
      .def(py::init<absl::string_view, const sax::client::Options*>())
      .def(py::init<absl::string_view>())
      .def("AM", &sax::client::pybind::Model::AM)
      .def("LM", &sax::client::pybind::Model::LM)
      .def("VM", &sax::client::pybind::Model::VM)
      .def("CM", &sax::client::pybind::Model::CM);

  m.def("StartDebugPort", &sax::client::pybind::StartDebugPort);

  m.def(
      "Publish",
      [](absl::string_view id, absl::string_view model_path,
         absl::string_view checkpoint_path, int num_replicas) -> absl::Status {
        return sax::client::pybind::Publish(id, model_path, checkpoint_path,
                                            num_replicas);
      });

  m.def("Unpublish", [](absl::string_view id) -> absl::Status {
    return sax::client::pybind::Unpublish(id);
  });

  m.def(
      "Update",
      [](absl::string_view id, absl::string_view model_path,
         absl::string_view checkpoint_path, int num_replicas) -> absl::Status {
        return sax::client::pybind::Update(id, model_path, checkpoint_path,
                                           num_replicas);
      });

  m.def("List",
        [](absl::string_view id)
            -> absl::StatusOr<std::tuple<std::string, std::string, int>> {
          return sax::client::pybind::List(id);
        });

  m.def("ListAll",
        [](absl::string_view id) -> absl::StatusOr<std::vector<std::string>> {
          return sax::client::pybind::ListAll(id);
        });
}
