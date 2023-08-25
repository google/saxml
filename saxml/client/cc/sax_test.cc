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

#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "saxml/common/platform/env.h"
#include "saxml/protobuf/common.pb.h"

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

TEST(TestToProto, Valid) {
  const float kTemp = 0.1;
  const float kSteps = 32;
  const float kTopK = 10;

  ModelOptions options;
  options.SetExtraInput("temperature", kTemp);
  options.SetExtraInput("per_example_max_decode_steps", kSteps);
  options.SetExtraInput("per_example_top_k", kTopK);

  ExtraInputs result;
  options.ToProto(&result);

  EXPECT_FLOAT_EQ(result.items().at("temperature"), kTemp);
  EXPECT_FLOAT_EQ(result.items().at("per_example_max_decode_steps"), kSteps);
  EXPECT_FLOAT_EQ(result.items().at("per_example_top_k"), kTopK);
}

TEST(TestFromProto, Valid) {
  const float kTemp = 0.1;
  const float kSteps = 32;
  const float kTopK = 10;
  const float kVal0 = 0.1;
  const float kVal1 = 0.2;
  const std::string kStr = "match";

  ExtraInputs expected;
  expected.mutable_items()->insert({"temperature", kTemp});
  expected.mutable_items()->insert({"per_example_max_decode_steps", kSteps});
  expected.mutable_items()->insert({"per_example_top_k", kTopK});
  Tensor tensor;
  tensor.mutable_values()->Add(kVal0);
  tensor.mutable_values()->Add(kVal1);
  expected.mutable_tensors()->insert({"prompt_embeddings", tensor});
  expected.mutable_strings()->insert({"regex", kStr});

  ModelOptions options;
  options.FromProto(expected);

  EXPECT_FLOAT_EQ(options.GetExtraInput("temperature"), kTemp);
  EXPECT_FLOAT_EQ(options.GetExtraInput("per_example_max_decode_steps"),
                  kSteps);
  EXPECT_FLOAT_EQ(options.GetExtraInput("per_example_top_k"), kTopK);
  std::vector<float> values = options.GetExtraInputTensor("prompt_embeddings");
  EXPECT_FLOAT_EQ(values[0], kVal0);
  EXPECT_FLOAT_EQ(values[1], kVal1);
  EXPECT_EQ(options.GetExtraInputString("regex"), kStr);
}

TEST(SaxModelOptions, Copy) {
  const float kTemp = 0.1;
  const float kSteps = 32;
  const float kTopK = 10;
  const int kTimeout = 17;

  ModelOptions options;
  options.SetExtraInput("temperature", kTemp);
  options.SetExtraInput("per_example_max_decode_steps", kSteps);
  options.SetExtraInput("per_example_top_k", kTopK);
  options.SetTimeout(kTimeout);

  ModelOptions other_options(options);
  ExtraInputs other_options_proto;
  other_options.ToProto(&other_options_proto);

  EXPECT_FLOAT_EQ(other_options_proto.items().at("temperature"), kTemp);
  EXPECT_FLOAT_EQ(
      other_options_proto.items().at("per_example_max_decode_steps"), kSteps);
  EXPECT_FLOAT_EQ(other_options_proto.items().at("per_example_top_k"), kTopK);
  EXPECT_EQ(other_options.GetTimeout(), kTimeout);
}

}  // namespace
}  // namespace client
}  // namespace sax
