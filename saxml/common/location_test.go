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

package location_test

import (
	"context"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"saxml/common/addr"
	"saxml/common/location"
	"saxml/common/platform/env"
	_ "saxml/common/platform/register" // registers a platform
	"saxml/common/testutil"
	"saxml/common/watchable"

	pb "saxml/protobuf/admin_go_proto_grpc"
)

// Tests the address setter and getter using a test cell.
func TestSetFetchAddr(t *testing.T) {
	ctx := context.Background()
	saxCell := "/sax/test-addr"
	testutil.SetUp(ctx, t, saxCell)

	port := 10000
	c, err := addr.SetAddr(ctx, port, saxCell)
	if err != nil {
		t.Fatalf("SetAddr(%v, %s) error %v, want no error", port, saxCell, err)
	}
	defer close(c)

	got, err := addr.FetchAddr(ctx, saxCell)
	wantSuffix := strconv.Itoa(port)
	if err != nil {
		t.Errorf("FetchAddr(%s) error %v, want no error", saxCell, err)
	} else if !strings.HasSuffix(got, wantSuffix) {
		t.Errorf("FetchAddr(%s) = %s, want suffix %s", saxCell, got, wantSuffix)
	}
}

// Test the address watcher using a test cell.
func TestJoin(t *testing.T) {
	ctx := context.Background()
	saxCell := "/sax/test-join"
	testutil.SetUp(ctx, t, saxCell)

	// Start an admin server.
	port, err := env.Get().PickUnusedPort()
	if err != nil {
		t.Fatalf("PickUnusedPort() error %v, want no error", err)
	}
	testutil.StartStubAdminServerT(t, port, nil, saxCell)

	// Start the address watcher.
	modelAddr := "localhost:10000"
	specs := &pb.ModelServer{
		ChipType:     pb.ModelServer_CHIP_TYPE_TPU_V4,
		ChipTopology: pb.ModelServer_CHIP_TOPOLOGY_2X2,
	}
	if err := location.Join(ctx, saxCell, modelAddr, "", specs, 0); err != nil {
		t.Fatalf("Join(%s) error %v, want no error", saxCell, err)
	}

	// We should see the joined model server after a small delay.
	time.Sleep(time.Second)
	resp, err := testutil.CallAdminServer(ctx, saxCell, &pb.WatchLocRequest{Seqno: 0})
	if err != nil {
		t.Fatalf("CallAdminServer(%s) error %v, want no error", saxCell, err)
	}
	result := watchable.FromProto(resp.(*pb.WatchLocResponse).GetResult())
	dataset := result.Data
	if dataset == nil {
		dataset = watchable.NewDataSet()
	}
	dataset.Apply(result.Log)
	got := dataset.ToList()
	if len(got) != 1 || got[0] != modelAddr {
		t.Errorf("WatchLoc got %v, want [%q]", got, modelAddr)
	}
}

// Test the address watcher in a test cell where no admin server has ever run.
func TestJoinEmptyCell(t *testing.T) {
	ctx := context.Background()
	saxCell := "/sax/test-join-empty"
	testutil.SetUp(ctx, t, saxCell)

	// Start the address watcher.
	modelAddr := "localhost:10000"
	specs := &pb.ModelServer{
		ChipType:     pb.ModelServer_CHIP_TYPE_TPU_V4,
		ChipTopology: pb.ModelServer_CHIP_TOPOLOGY_2X2,
	}

	// Join returns an error only for serious problems such as the Sax cell isn't created.
	// Even if the admin server momentarily disappears or has never existed, the Join call doesn't
	// return any error but the best-effort background watcher will print log warnings.
	if err := location.Join(ctx, saxCell, modelAddr, "", specs, 0); err != nil {
		t.Errorf("Join(%s) error %v, want no error", err, saxCell)
	}
}

// Tests leader election between a few participants.
func TestLeaderElection(t *testing.T) {
	ctx := context.Background()
	saxCell := "/sax/test-election"
	testutil.SetUp(ctx, t, saxCell)

	numParticipants := 5
	var wg sync.WaitGroup
	wg.Add(numParticipants)
	// Use a global variable to verify mutual exclusivity between SetAddr --> close(c) sequences.
	// Mutual exclusivity serializes access to `global` so there is no need for mutex protection.
	// Do not use FetchAddr because it uses Svelte and isn't reliable enough in repeated runs.
	var global int
	for i := 0; i < numParticipants; i++ {
		go func() {
			defer wg.Done()
			port := rand.Intn(10000)

			// Wait until this participant becomes the leader.
			c, err := addr.SetAddr(ctx, port, saxCell)
			if err != nil {
				t.Errorf("SetAddr(%v, %s) error %v, want no error", port, saxCell, err)
			}
			defer close(c)

			// Assign a unique value to `global` and then sleep for a while.
			global = port
			time.Sleep(time.Duration(rand.Intn(1000)) * time.Millisecond)

			// Verify this participant has held the lock during the sleep by reading `global`.
			got := global
			if got != port {
				t.Errorf("global = %v, want %v", got, port)
			}
		}()
	}
	wg.Wait()
}
