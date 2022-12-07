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

#include "saxml/client/cc/sax.h"

#include "gtest/gtest.h"
#include "saxml/common/platform/env.h"

namespace sax {
namespace client {
namespace {

using AMAsrHyp = AudioModel::AsrHyp;
using LMScoredText = LanguageModel::ScoredText;
using VMScoredText = VisionModel::ScoredText;

TEST(InvalidFormatSaxModel, ReturnsErrors) {
  // Check the Sax Go client can correctly initialize its platform environment.
  sax::client::StartDebugPort(PickUnusedPortOrDie());

  Model* model = nullptr;
  EXPECT_FALSE(Model::Open("", &model).ok());
}

}  // namespace
}  // namespace client
}  // namespace sax
