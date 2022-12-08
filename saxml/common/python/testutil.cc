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

#include "saxml/common/testutil.h"

#include "pybind11/pybind11.h"

namespace sax {

PYBIND11_MODULE(testutil, m) {
  pybind11::enum_<ModelType>(m, "ModelType")
      .value("Language", ModelType::Language)
      .value("Vision", ModelType::Vision)
      .value("Audio", ModelType::Audio)
      .value("Custom", ModelType::Custom)
      .export_values();

  m.def("SetUp", &SetUp, "Set up a Sax test environment");
  m.def("StartLocalTestCluster", &StartLocalTestCluster,
        "Set up and start a Sax test environment", pybind11::arg("sax_cell"),
        pybind11::arg("model_type") = ModelType::Language,
        pybind11::arg("admin_port") = 0);

  m.def("StopLocalTestCluster", &StopLocalTestCluster,
        "Stop and clean up a Sax test environment", pybind11::arg("sax_cell"));
}

}  // namespace sax
