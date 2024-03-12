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

#include <optional>
#include <vector>

#include "saxml/client/python/wrapper.h"
#include "saxml/protobuf/common.pb.h"
#include "saxml/protobuf/multimodal.pb.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11_abseil/absl_casters.h"    // IWYU pragma: keep
#include "pybind11_abseil/status_casters.h"  // IWYU pragma: keep
#include "pybind11_protobuf/native_proto_caster.h"

namespace py = pybind11;

PYBIND11_MODULE(sax, m) {
  py::google::ImportStatusModule();
  pybind11_protobuf::ImportNativeProtoCasters();

  py::class_<sax::client::Options>(m, "Options")
      .def(py::init<>())
      .def("__copy__",
           [](const sax::client::Options& self) {
             return sax::client::Options(self);
           })
      .def("__deepcopy__", [](sax::client::Options& self,
                              py::dict) { return sax::client::Options(self); })
      .def_readwrite("num_conn", &sax::client::Options::num_conn)
      .def_readwrite("proxy_addr", &sax::client::Options::proxy_addr)
      .def_readwrite("fail_fast", &sax::client::Options::fail_fast);

  py::class_<sax::client::ModelOptions>(m, "ModelOptions")
      .def(py::init<>())
      .def("__copy__",
           [](const sax::client::ModelOptions& self) {
             return sax::client::ModelOptions(self);
           })
      .def("__deepcopy__",
           [](sax::client::ModelOptions& self, py::dict) {
             return sax::client::ModelOptions(self);
           })
      .def("SetExtraInput", &sax::client::ModelOptions::SetExtraInput)
      .def("SetExtraInputTensor",
           &sax::client::ModelOptions::SetExtraInputTensor)
      .def("SetExtraInputString",
           &sax::client::ModelOptions::SetExtraInputString)
      .def("GetExtraInput", &sax::client::ModelOptions::GetExtraInput)
      .def("GetTimeout", &sax::client::ModelOptions::GetTimeout)
      .def("SetTimeout", &sax::client::ModelOptions::SetTimeout)
      .def("ToDebugString", [](sax::client::ModelOptions& mo) {
        ::sax::ExtraInputs extra_inputs;
        mo.ToProto(&extra_inputs);
        return extra_inputs.DebugString();
      }).def("ToProto", [](sax::client::ModelOptions& mo) {
        ::sax::ExtraInputs extra_inputs;
        mo.ToProto(&extra_inputs);
        return extra_inputs;
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
            // py_callback:
            // Callable[[bool, list[tuple[str, int, list[float]]]], None]
            return lm.GenerateStream(text, py_callback, options);
          },
          py::arg("text"), py::arg("callback"), py::arg("options") = nullptr)
      .def(
          "Gradient",
          [](sax::client::pybind::LanguageModel& lm, absl::string_view prefix,
             absl::string_view suffix, const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::pair<
                  std::vector<double>,
                  absl::flat_hash_map<std::string, std::vector<double>>>> {
            return lm.Gradient(prefix, suffix, options);
          },
          py::arg("prefix"), py::arg("suffix"), py::arg("options") = nullptr);

  py::class_<sax::client::pybind::MultimodalModel>(m, "MultimodalModel")
      .def("Generate", &sax::client::pybind::MultimodalModel::Generate,
           py::arg("request"), py::arg("options") = nullptr)
      .def("Score", &sax::client::pybind::MultimodalModel::Score,
           py::arg("request"), py::arg("options") = nullptr);

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
          "TextAndImageToImage",
          [](sax::client::pybind::VisionModel& vm, absl::string_view text,
             absl::string_view image_bytes,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<
                  std::vector<std::pair<pybind11::bytes, double>>> {
            return vm.TextAndImageToImage(text, image_bytes, options);
          },
          py::arg("text"), py::arg("image_bytes"), py::arg("options") = nullptr)
      .def(
          "ImageToImage",
          [](sax::client::pybind::VisionModel& vm, absl::string_view image,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<
                  std::vector<std::pair<pybind11::bytes, double>>> {
            return vm.ImageToImage(image, options);
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
             std::vector<std::tuple<double, double, double, double>> boxes,
             const sax::client::ModelOptions* options)
              -> absl::StatusOr<std::vector<std::tuple<
                  double, double, double, double, pybind11::bytes, double,
                  std::tuple<int32_t, int32_t, pybind11::bytes>>>> {
            return vm.Detect(image_bytes, text, boxes, options);
          },
          py::arg("image_bytes"), py::arg("text") = std::vector<std::string>{},
          py::arg("boxes") =
              std::vector<std::tuple<double, double, double, double>>{},
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
      .def("CM", &sax::client::pybind::Model::CM)
      .def("MM", &sax::client::pybind::Model::MM);

  m.def("StartDebugPort", &sax::client::pybind::StartDebugPort);

  py::class_<sax::client::AdminOptions>(m, "AdminOptions")
      .def(py::init<>())
      .def("__copy__",
           [](const sax::client::AdminOptions& self) {
             return sax::client::AdminOptions(self);
           })
      .def("__deepcopy__",
           [](sax::client::AdminOptions& self, py::dict) {
             return sax::client::AdminOptions(self);
           })
      .def_readwrite("timeout", &sax::client::AdminOptions::timeout);

  m.def("Publish", &sax::client::pybind::Publish, py::arg("id"),
        py::arg("model_path"), py::arg("checkpoint_path"),
        py::arg("num_replicas"), py::arg("overrides") = std::nullopt,
        py::arg("options") = nullptr);

  m.def("Unpublish", &sax::client::pybind::Unpublish, py::arg("id"),
        py::arg("options") = nullptr);

  m.def("Update", &sax::client::pybind::Update, py::arg("id"),
        py::arg("model_path"), py::arg("checkpoint_path"),
        py::arg("num_replicas"), py::arg("options") = nullptr);

  m.def("List", &sax::client::pybind::List, py::arg("id"),
        py::arg("options") = nullptr);

  py::class_<sax::client::ModelDetail>(m, "ModelDetail")
      .def_readonly("model", &sax::client::ModelDetail::model)
      .def_readonly("ckpt", &sax::client::ModelDetail::ckpt)
      .def_readonly("max_replicas", &sax::client::ModelDetail::max_replicas)
      .def_readonly("active_replicas",
                    &sax::client::ModelDetail::active_replicas)
      .def_readonly("overrides", &sax::client::ModelDetail::overrides);

  m.def("ListDetail", &sax::client::pybind::ListDetail, py::arg("id"),
        py::arg("options") = nullptr);

  m.def("ListAll", &sax::client::pybind::ListAll, py::arg("id"),
        py::arg("options") = nullptr);

  py::class_<sax::client::ModelServerTypeStat>(m, "ModelServerTypeStat")
      .def_readonly("chip_type", &sax::client::ModelServerTypeStat::chip_type)
      .def_readonly("chip_topology",
                    &sax::client::ModelServerTypeStat::chip_topology)
      .def_readonly("num_replicas",
                    &sax::client::ModelServerTypeStat::num_replicas);

  m.def("Stats", &sax::client::pybind::Stats, py::arg("id"),
        py::arg("options") = nullptr);

  m.def("WaitForReady", &sax::client::pybind::WaitForReady, py::arg("id"),
        py::arg("num_replicas"), py::arg("options") = nullptr);
}
