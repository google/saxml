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

// The implementation simply delegates to cgo wrapped testutil.go
#include "saxml/common/testutil.h"

#include <string>

#include "saxml/common/testutilwrapper.h"

namespace sax {

void SetUp(const std::string& sax_cell) {
  sax_set_up(const_cast<char*>(sax_cell.data()), sax_cell.size());
}

void StartLocalTestCluster(const std::string& sax_cell, ModelType model_type,
                           int admin_port) {
  sax_start_local_test_cluster(const_cast<char*>(sax_cell.data()),
                               sax_cell.size(), (int)model_type, admin_port);
}

void StopLocalTestCluster(const std::string& sax_cell) {
  sax_stop_local_test_cluster(const_cast<char*>(sax_cell.data()),
                              sax_cell.size());
}

}  // namespace sax
