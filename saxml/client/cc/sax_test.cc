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

#include "net/proto2/contrib/parse_proto/parse_text_proto.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "saxml/common/platform/env.h"
#include "saxml/protobuf/common.pb.h"

namespace sax {
namespace client {
namespace {

using AMAsrHyp = AudioModel::AsrHyp;
using LMScoredText = LanguageModel::ScoredText;
using VMScoredText = VisionModel::ScoredText;
using ::proto2::contrib::parse_proto::ParseTextProtoOrDie;
using ::testing::EqualsProto;

TEST(InvalidFormatSaxModel, ReturnsErrors) {
  // Check the Sax Go client can correctly initialize its platform environment.
  sax::client::StartDebugPort(PickUnusedPortOrDie());

  Model* model = nullptr;
  EXPECT_FALSE(Model::Open("", &model).ok());
}

TEST(TestToProto, Valid) {
  const float kTemp = 0.1;
  const float kDecodeSteps = 32;
  const float kTopK = 10;

  ModelOptions options;
  options.SetExtraInput("temperature", kTemp);
  options.SetExtraInput("per_example_max_decode_steps", kDecodeSteps);
  options.SetExtraInput("per_example_top_k", kTopK);

  ExtraInputs result;
  options.ToProto(&result);

  ExtraInputs expected = ParseTextProtoOrDie(
      absl::StrFormat(R"pb(
                        items { key: "temperature" value: %f }
                        items { key: "per_example_max_decode_steps" value: %f }
                        items { key: "per_example_top_k" value: %f }
                      )pb",
                      kTemp, kDecodeSteps, kTopK));

  EXPECT_THAT(result, EqualsProto(expected));
}

TEST(TestFromProtoToProto, Valid) {
  ExtraInputs expected = ParseTextProtoOrDie(R"pb(
    items { key: "temperature" value: 0.9 }
    items { key: "per_example_max_decode_steps" value: 256 }
    items { key: "per_example_top_k" value: 0 }
    tensors {
      key: "prompt_embeddings"
      value: { values: 0.1 values: 0.2 }
    }
  )pb");

  ModelOptions options;
  options.FromProto(expected);

  ExtraInputs result;
  options.ToProto(&result);
  EXPECT_THAT(result, EqualsProto(expected));
}
}  // namespace
}  // namespace client
}  // namespace sax
