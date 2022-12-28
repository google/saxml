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

package connection

import (
	"context"
	"errors"
	"strconv"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	saxerrors "saxml/common/errors"
	"saxml/common/platform/env"
	_ "saxml/common/platform/cloud" // registers a platform
	"saxml/common/testutil"

	pb "saxml/protobuf/lm_go_proto_grpc"
	pbgrpc "saxml/protobuf/lm_go_proto_grpc"
)

func TestConnection(t *testing.T) {
	var ports []int
	for i := 0; i < 2; i++ {
		port, err := env.Get().PickUnusedPort()
		if err != nil {
			t.Fatalf("Failed to get unused port: %v", err)
		}
		ports = append(ports, port)
		testutil.StartStubModelServer(t, port)
	}

	addresses := []string{}
	for i := 0; i < 2; i++ {
		addresses = append(addresses, "localhost:"+strconv.Itoa(ports[i]))
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			connTable := newConnTable()
			conn, err := connTable.getOrCreate(context.Background(), addresses[j])
			if err != nil {
				t.Fatalf("Creating connection for address %s failed with %v\n", addresses[j], err)
			}
			req := &pb.ScoreRequest{
				ModelKey: "m1",
				Suffix:   []string{"abc"},
				Prefix:   "xyz",
			}
			modelServer := pbgrpc.NewLMServiceClient(conn)
			res, err := modelServer.Score(context.Background(), req)
			if err != nil {
				t.Fatalf("Unable to Score() against address %s due to %v\n", addresses[j], err)
			}

			logP := res.GetLogp()
			want := []float64{float64(len("abcxyz")) * 0.1} // match stub implementation.
			if diff := cmp.Diff(logP, want); diff != "" {
				t.Errorf("TestConnection wants %f but gets %f\n", want, logP)
			}
			time.Sleep(3 * time.Second) // so purge runs.
		}
	}
}

func TestFail(t *testing.T) {
	port, err := env.Get().PickUnusedPort()
	if err != nil {
		t.Fatalf("Failed to get unused port: %v", err)
	}
	addr := "localhost:" + strconv.Itoa(port)
	connTable := newConnTable()
	conn, err := connTable.getOrCreate(context.Background(), addr)
	if err == nil {
		t.Fatalf("Creating connection for address %s should fail but conn = [%v] is returned\n", addr, conn)
	}
	if !errors.Is(err, saxerrors.ErrUnavailable) {
		t.Fatalf("Creating connection for address %s got error %v, want error unavailable", addr, err)
	}
}
