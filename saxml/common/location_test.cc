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

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/channel.h"
#include "grpcpp/client_context.h"
#include "saxml/common/platform/env.h"
#include "saxml/common/testutil.h"
#include "saxml/protobuf/admin.grpc.pb.h"
#include "saxml/protobuf/admin.pb.h"

namespace sax {
namespace {

TEST(LocationTest, Join) {
  std::string sax_cell = "/sax/test-join-cc";
  const int admin_port = PickUnusedPortOrDie();

  // Start the admin server and one model server by default.
  sax::StartLocalTestCluster(sax_cell, sax::ModelType::Language, admin_port);

  // Pretend that the second server joined.
  const int server_port = PickUnusedPortOrDie();
  std::string model_addr = absl::StrCat("localhost:", server_port);
  ModelServer specs;
  ASSERT_EQ(Join(sax_cell, model_addr, "", specs.SerializeAsString()), "");

  absl::SleepFor(absl::Seconds(3));  // wait for the initial Join to happen
  std::string admin_addr = absl::StrCat("localhost:", admin_port);
  std::shared_ptr<grpc::Channel> channel = CreateGRPCChannel(admin_addr);
  std::unique_ptr<::sax::Admin::Stub> stub = ::sax::Admin::NewStub(channel);
  grpc::ClientContext ctx;
  WatchLocRequest req;
  WatchLocResponse resp;
  ASSERT_TRUE(stub->WatchLoc(&ctx, req, &resp).ok());
  const WatchResult& result = resp.result();
  EXPECT_TRUE(result.has_fullset());
  std::vector<std::string> model_addrs;
  model_addrs.reserve(result.values_size());
  for (const std::string& value : result.values()) {
    model_addrs.push_back(value);
  }
  for (const WatchResult_Mutation& change : result.changelog()) {
    if (change.has_addition()) {
      model_addrs.push_back(change.addition());
    }
    if (change.has_deletion()) {
      model_addrs.erase(std::remove(model_addrs.begin(), model_addrs.end(),
                                    change.deletion()),
                        model_addrs.end());
    }
  }
  EXPECT_THAT(model_addrs, testing::Contains(model_addr));

  // Stop the servers.
  sax::StopLocalTestCluster(sax_cell);
}

}  // namespace
}  // namespace sax
