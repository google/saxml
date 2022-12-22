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

#include "saxml/common/location.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "saxml/common/locationwrapper.h"

namespace sax {

std::string Join(const std::string& sax_cell, const std::string& ip_port,
                 const std::string& debug_addr,
                 const std::string& serialized_specs, int admin_port) {
  const char* result =
      sax_join(const_cast<char*>(sax_cell.data()), sax_cell.size(),
               const_cast<char*>(ip_port.data()), ip_port.size(),
               const_cast<char*>(debug_addr.data()), debug_addr.size(),
               const_cast<char*>(serialized_specs.data()),
               serialized_specs.size(), admin_port);
  auto ret = std::string(result, strlen(result));
  free(reinterpret_cast<void*>(const_cast<char*>(result)));
  return ret;
}

}  // namespace sax
