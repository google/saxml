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

#ifndef SAXML_COMMON_LOCATION_H_
#define SAXML_COMMON_LOCATION_H_

#include <string>

namespace sax {

// Join is called by model servers to join the admin server in a SAX cell.
// ip_port and specs are those of the model server's. It returns an empty string
// if the call is successful, or a non-empty error message otherwise.
//
// A background address watcher starts running indefinitely on successful calls.
// This address watcher will attempt to join initially after a small delay
// and then periodically, as well as whenever the admin server address changes
// if the platform supports address watching.
//
// If admin_port is not 0, this process will start an admin server for sax_cell.
// in the background.
std::string Join(const std::string& sax_cell, const std::string& ip_port,
                 const std::string& debug_addr,
                 const std::string& serialized_specs, int admin_port = 0);

}  // namespace sax

#endif  // SAXML_COMMON_LOCATION_H_
