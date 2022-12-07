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
	"fmt"
	"net"
	"path/filepath"
	"strconv"
	"time"

	log "github.com/golang/glog"
	"google.golang.org/protobuf/proto"
	"saxml/common/cell"
	"saxml/common/errors"
	"saxml/common/ipaddr"
	"saxml/common/platform/env"
	"saxml/common/retrier"

	pb "saxml/protobuf/admin_go_proto_grpc"
	pbgrpc "saxml/protobuf/admin_go_proto_grpc"
)

const (
	// locationFile is the name of the file storing the admin server location in the
	// <root>/sax/<cell> directory.
	locationFile = "location.proto"

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
	retryTimeout = time.Minute * 2
)

// parseAddr reads the admin server address from bytes.
func parseAddr(bytes []byte) (string, error) {
	location := &pb.Location{}
	if err := proto.Unmarshal(bytes, location); err != nil {
		return "", err
	}
	addr := location.GetLocation()
	if addr == "" {
		return "", fmt.Errorf("got an empty location: %w", errors.ErrFailedPrecondition)
	}
	return addr, nil
}

// FetchAddr fetches the admin server address for a Sax cell.
func FetchAddr(ctx context.Context, saxCell string) (string, error) {
	if err := cell.Exists(ctx, saxCell); err != nil {
		return "", err
	}
	path, err := cell.Path(ctx, saxCell)
	if err != nil {
		return "", err
	}
	fname := filepath.Join(path, locationFile)

	bytes, err := env.Get().ReadCachedFile(ctx, fname)
	if err != nil {
		return "", err
	}
	addr, err := parseAddr(bytes)
	if err != nil {
		return "", err
	}
	log.Infof("FetchAddr %s %q", fname, addr)
	return addr, nil
}

// join makes a Join RPC call to an admin server address.
func join(ctx context.Context, addr string, ipPort string, specs *pb.ModelServer) error {
	dialCtx, dialCancel := context.WithTimeout(ctx, dialTimeout)
	defer dialCancel()
	conn, err := env.Get().DialContext(dialCtx, addr)
	if err != nil {
		return err
	}
	defer conn.Close()
	client := pbgrpc.NewAdminClient(conn)

	req := &pb.JoinRequest{
		Address:     ipPort,
		ModelServer: proto.Clone(specs).(*pb.ModelServer),
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
func Join(ctx context.Context, saxCell string, ipPort string, specs *pb.ModelServer) error {
	if err := cell.Exists(ctx, saxCell); err != nil {
		return err
	}
	path, err := cell.Path(ctx, saxCell)
	if err != nil {
		return err
	}
	fname := filepath.Join(path, locationFile)

	// If the platform supports it, subscribe to ongoing admin server address updates.
	var updates <-chan []byte
	updates, err = env.Get().Watch(ctx, fname)
	if err != nil {
		return err
	}

	retryJoinWithTimeout := func(ctx context.Context, addr string) {
		ctx, cancel := context.WithTimeout(ctx, retryTimeout)
		defer cancel()
		retrier.Do(
			ctx, func() error { return join(ctx, addr, ipPort, specs) }, errors.JoinShouldRetry,
		)
	}

	// Start a best-effort background address watcher that runs indefinitely and ensures the server
	// has joined the latest admin server.
	go func() {
		// Delay the first call by a few seconds so the calling model server can get ready to handle
		// GetStatus calls.
		timer := time.NewTimer(2 * time.Second)
		for {
			select {
			// Call Join every time the admin address changes.
			case bytes := <-updates:
				addr, err := parseAddr(bytes)
				if err != nil {
					log.Errorf("Failed to get admin address to rejoin, retrying later: %v", err)
				} else {
					retryJoinWithTimeout(ctx, addr)
				}
				timer.Reset(joinPeriod)
			// Call Join at least every `joinPeriod` regardless of address change updates.
			case <-timer.C:
				addr, err := FetchAddr(ctx, saxCell)
				if err != nil {
					log.Errorf("Failed to get admin address to rejoin, retrying later: %v", err)
				} else {
					retryJoinWithTimeout(ctx, addr)
				}
				timer.Reset(joinPeriod)
			}
		}
	}()

	return nil
}

// SetAddr makes this task the admin server for a Sax cell. This function blocks until it
// successfully becomes the leader or encounters an error. On success, callers can close the
// returned channel to safely release the address lock.
//
// In tests, users should arrange to call SetAddr (directly or by creating an admin server) before
// FetchAddr is called anywhere.
func SetAddr(ctx context.Context, port int, saxCell string) (chan<- struct{}, error) {
	if err := cell.Exists(ctx, saxCell); err != nil {
		return nil, err
	}
	path, err := cell.Path(ctx, saxCell)
	if err != nil {
		return nil, err
	}
	fname := filepath.Join(path, locationFile)

	addr := net.JoinHostPort(ipaddr.MyIPAddr().String(), strconv.Itoa(port))
	location := &pb.Location{Location: addr}
	content, err := proto.Marshal(location)
	if err != nil {
		return nil, err
	}

	// If the platform supports it, block until this process becomes the leader.
	closer, err := env.Get().Lead(ctx, fname)
	if err != nil {
		return nil, err
	}

	log.Infof("SetAddr %s %q", fname, addr)
	return closer, env.Get().WriteFile(ctx, fname, content)
}
