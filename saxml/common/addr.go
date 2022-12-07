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

// Package addr provides functions to set and get admin server addresses.
package addr

import (
	"context"
	"fmt"
	"net"
	"path/filepath"
	"strconv"

	log "github.com/golang/glog"
	"google.golang.org/protobuf/proto"
	"saxml/common/cell"
	"saxml/common/errors"
	"saxml/common/ipaddr"
	"saxml/common/platform/env"
	pb "saxml/protobuf/admin_go_proto_grpc"
)

const (
	// LocationFile is the name of the file storing the admin server location in the
	// <root>/sax/<cell> directory.
	LocationFile = "location.proto"
)

// ParseAddr reads the admin server address from bytes.
func ParseAddr(bytes []byte) (string, error) {
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
	fname := filepath.Join(path, LocationFile)

	bytes, err := env.Get().ReadCachedFile(ctx, fname)
	if err != nil {
		return "", err
	}
	addr, err := ParseAddr(bytes)
	if err != nil {
		return "", err
	}
	log.Infof("FetchAddr %s %q", fname, addr)
	return addr, nil
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
	fname := filepath.Join(path, LocationFile)

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
