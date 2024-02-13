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

// Package location provides functions to set, get, and join admin server locations.
package location

import (
	"context"
	"path/filepath"
	"time"

	log "github.com/golang/glog"
	"google.golang.org/protobuf/proto"
	"saxml/admin/admin"
	"saxml/common/addr"
	"saxml/common/cell"
	"saxml/common/errors"
	"saxml/common/platform/env"
	"saxml/common/retrier"

	pb "saxml/protobuf/admin_go_proto_grpc"
	pbgrpc "saxml/protobuf/admin_go_proto_grpc"
)

const (
	// Join RPC dial timeout.
	dialTimeout = time.Second * 10

	// Join RPC call timeout.
	joinTimeout = time.Second * 10

	// Call Join at least every this much time, to make sure model servers accidentally dropped by
	// an admin server (that never changes addresses) still has a chance to join.
	joinPeriod = time.Minute * 15

	// Timeout for repeated Join RPC calls. When a model server just boots up and calls Join, it may
	// not be ready to respond to GetStatus calls issued by the admin server Join RPC handler yet.
	// Retry Join calls for this much time to allow the model server to become ready.
	retryTimeout = time.Minute
)

// join makes a Join RPC call to an admin server address.
func join(ctx context.Context, addr string, ipPort string, debugAddr string, dataAddr string, specs *pb.ModelServer) error {
	dialCtx, dialCancel := context.WithTimeout(ctx, dialTimeout)
	defer dialCancel()
	conn, err := env.Get().DialContext(dialCtx, addr)
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pbgrpc.NewAdminClient(conn)

	req := &pb.JoinRequest{
		Address:      ipPort,
		DebugAddress: debugAddr,
		DataAddress:  dataAddr,
		ModelServer:  proto.Clone(specs).(*pb.ModelServer),
	}
	joinCtx, joinCancel := context.WithTimeout(ctx, joinTimeout)
	defer joinCancel()
	_, err = client.Join(joinCtx, req)
	return err
}

// Join is called by model servers to join the admin server in a Sax cell. ipPort and specs
// are those of the model server's.
//
// A background address watcher starts running indefinitely on successful calls. This address
// watcher will attempt to rejoin periodically.
//
// If admin_port is not 0, start an admin server for sax_cell at the given port in the background.
func Join(ctx context.Context, saxCell string, ipPort string, debugAddr string, dataAddr string, specs *pb.ModelServer, adminPort int) error {
	if err := cell.Exists(ctx, saxCell); err != nil {
		return err
	}
	path, err := cell.Path(ctx, saxCell)
	if err != nil {
		return err
	}
	fname := filepath.Join(path, addr.LocationFile)

	// If multiple model servers call Join with non-zero admin port values, all but one model server
	// will be stuck at leader election. Put the admin server start call in a goroutine so Join calls
	// aren't blocked.
	if adminPort != 0 {
		go func() {
			adminServer := admin.NewServer(saxCell, adminPort)
			log.Infof("Starting admin server at :%v", adminPort)
			adminServer.EnableStatusPages()
			if err := adminServer.Start(ctx); err != nil {
				log.Errorf("Failed to start admin server at :%v: %v", adminPort, err)
				return
			}
			log.Infof("Started admin server at :%v", adminPort)
		}()
	}

	// If the platform supports it, subscribe to ongoing admin server address updates.
	var updates <-chan []byte
	updates, err = env.Get().Watch(ctx, fname)
	if err != nil {
		return err
	}

	retryJoinWithTimeout := func(ctx context.Context, addr string) error {
		ctx, cancel := context.WithTimeout(ctx, retryTimeout)
		defer cancel()
		return retrier.Do(
			ctx, func() error {
				err := join(ctx, addr, ipPort, debugAddr, dataAddr, specs)
				return err
			}, errors.JoinShouldRetry,
		)
	}

	// Start a best-effort background address watcher that runs indefinitely and ensures the server
	// has joined the latest admin server.
	go func() {
		// Delay the first call by a few seconds so the calling model server can get ready to handle
		// GetStatus calls issued by the admin server being joined.
		time.Sleep(2 * time.Second)
		// The address watcher may send the old address a few times repeatedly during an address change.
		// Use this variable to filter out unnecessary Join calls.
		var joinedAddr string
		// Regardless of address updates, we want to call Join on the admin server at least once this
		// much time in case address watching doesn't work.
		timer := time.NewTimer(joinPeriod)
		for {
			select {
			// Call Join every time the admin address changes.
			case bytes := <-updates:
				log.Info("Calling Join due to address update")
				addr, err := addr.ParseAddr(bytes)
				if err != nil {
					log.Errorf("ParseAddr error: %v", err)
					continue
				}
				if addr == joinedAddr {
					log.Infof("Not calling Join on old address %v", addr)
					continue
				}
				if err := retryJoinWithTimeout(ctx, addr); err != nil {
					log.Errorf("Failed to join %v: %v", addr, err)
					continue
				}
				log.Infof("Joined %v", addr)
				// On success, remember the address so this select branch calls Join only when a new address
				// is received.
				joinedAddr = addr
			// Call Join at least every `joinPeriod` regardless of address changes.
			case <-timer.C:
				timer.Reset(joinPeriod)
				log.Info("Calling Join at fixed interval")
				addr, err := addr.FetchAddr(ctx, saxCell)
				if err != nil {
					log.Errorf("FetchAddr error: %v", err)
					continue
				}
				if err := retryJoinWithTimeout(ctx, addr); err != nil {
					log.Errorf("Failed to join %v: %v", addr, err)
					continue
				}
				log.Infof("Joined %v", addr)
			}
		}
	}()

	return nil
}
