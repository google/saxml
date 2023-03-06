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

#ifndef SAXML_COMMON_TESTUTIL_H_
#define SAXML_COMMON_TESTUTIL_H_

#include <string>

namespace sax {

enum ModelType { Language, Vision, Audio, Custom };

// Sets up and starts a Sax testing environment.
void SetUp(const std::string& sax_cell);

// Sets up and starts a Sax testing environment.
//
// It starts one stub admin server and one stub model server of the given type.
//
// model_type defaults to Language is unspecified.
// admin_port is automatically picked if unspecified.
void StartLocalTestCluster(const std::string& sax_cell,
                           ModelType model_type = Language, int admin_port = 0);

// Stops and cleans up a running Sax testing environment.
void StopLocalTestCluster(const std::string& sax_cell);

}  // namespace sax

#endif  // SAXML_COMMON_TESTUTIL_H_
