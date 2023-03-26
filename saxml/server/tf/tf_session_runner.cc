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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "saxml/server/tf/np_conversions.h"
#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/gil.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/import_status_module.h"
#include "pybind11_abseil/status_casters.h"
#include "pybind11_protobuf/native_proto_caster.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace py = pybind11;

namespace sax {
namespace pybind {

// A class for executing tf sessions in Python.
class PybindTFSessionRunner {
 public:
  PybindTFSessionRunner(absl::string_view name) : name_(name) {}
  PybindTFSessionRunner(const PybindTFSessionRunner &) = delete;
  PybindTFSessionRunner(PybindTFSessionRunner &&) = default;
  PybindTFSessionRunner &operator=(const PybindTFSessionRunner &) = delete;
  PybindTFSessionRunner &operator=(PybindTFSessionRunner &&) = default;
  ~PybindTFSessionRunner() = default;

  absl::Status Initialize(const tensorflow::GraphDef &graph);
  absl::StatusOr<std::vector<py::array>> Run(
      const std::vector<std::string> &feeds,
      const std::vector<std::string> &fetches,
      const std::vector<py::array> &inputs);

 private:
  std::string name_;
  std::unique_ptr<tensorflow::Session> session_;
  tensorflow::RunOptions run_options_;
};

absl::Status PybindTFSessionRunner::Initialize(
    const tensorflow::GraphDef &graph) {
  py::gil_scoped_release release;
  tensorflow::SessionOptions options;
  // CPU-only.
  options.config.add_device_filters("/device:CPU:*");
  auto *const session_metadata =
      options.config.mutable_experimental()->mutable_session_metadata();
  session_metadata->set_name(name_);

  tensorflow::Session *sess_ptr;
  absl::Status status =
      tensorflow::ToAbslStatus(tensorflow::NewSession(options, &sess_ptr));
  session_ = std::unique_ptr<tensorflow::Session>(sess_ptr);
  if (!status.ok()) {
    return status;
  }
  return tensorflow::ToAbslStatus(session_->Create(graph));
}

absl::StatusOr<std::vector<py::array>> PybindTFSessionRunner::Run(
    const std::vector<std::string> &feeds,
    const std::vector<std::string> &fetches,
    const std::vector<py::array> &inputs) {
  if (inputs.size() != feeds.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "feeds size != inputs size, ", feeds.size(), " vs ", inputs.size()));
  }
  if (fetches.empty()) {
    return absl::InvalidArgumentError("fetches is empty");
  }
  if (session_ == nullptr) {
    return absl::FailedPreconditionError("Called before initialized.");
  }
  std::vector<std::pair<std::string, tensorflow::Tensor>> input_pairs;
  input_pairs.reserve(inputs.size());
  for (int i = 0; i < inputs.size(); i++) {
    tensorflow::Tensor t;
    absl::Status status =
        tensorflow::ToAbslStatus(NdArrayToTensor(inputs[i].ptr(), &t));
    if (!status.ok()) {
      return status;
    }
    input_pairs.push_back(std::make_pair(feeds[i], std::move(t)));
  }
  std::vector<tensorflow::Tensor> outputs;
  {
    py::gil_scoped_release release;
    absl::Status status = tensorflow::ToAbslStatus(
        session_->Run(run_options_, input_pairs, fetches, {}, &outputs,
                      /*run_metadata=*/nullptr));
    if (!status.ok()) {
      return status;
    }
  }
  std::vector<py::array> out_arrays;
  out_arrays.reserve(outputs.size());
  for (int i = 0; i < outputs.size(); i++) {
    PyObject *array;
    absl::Status status =
        tensorflow::ToAbslStatus(TensorToNdArray(outputs[i], &array));
    if (!status.ok()) {
      return status;
    }
    out_arrays.push_back(py::reinterpret_steal<py::array>(array));
  }
  return out_arrays;
}

PYBIND11_MODULE(tf_session_runner, m) {
  m.doc() = "pybind11 wrapper for tf_session_runner.";

  sax::ImportNumpy();
  pybind11_protobuf::ImportNativeProtoCasters();
  pybind11::google::ImportStatusModule();

  py::class_<PybindTFSessionRunner>(m, "TFSessionRunner")
      .def(py::init<absl::string_view>(), py::arg("name"))
      .def("initialize", &PybindTFSessionRunner::Initialize, py::arg("graph"),
           "Initializes the session runner with a graph.")
      .def("run", &PybindTFSessionRunner::Run, py::arg("feeds"),
           py::arg("fetches"), py::arg("inputs"),
           "Runs on numpy inputs. Must be already initialized.");
}

}  // namespace pybind
}  // namespace sax
