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

#include "saxml/common/ipaddr.h"

#include <cstdlib>
#include <cstring>
#include <string>

#include "saxml/common/ipaddrwrapper.h"

namespace sax {

std::string MyIPAddr() {
  const char* result = sax_ipaddr();
  auto ret = std::string(result, strlen(result));
  free(reinterpret_cast<void*>(const_cast<char*>(result)));
  return ret;
}

}  // end namespace sax
