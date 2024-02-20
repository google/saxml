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

#ifndef THIRD_PARTY_PY_SAXML_COMMON_IPADDR_H_
#define THIRD_PARTY_PY_SAXML_COMMON_IPADDR_H_

#include <string>

namespace sax {

// Returns an ip address of this process that other processes can connect to.
std::string MyIPAddr();

}  // namespace sax


#endif  // THIRD_PARTY_PY_SAXML_COMMON_IPADDR_H_
