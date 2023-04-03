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

// Package state provides functions for server state persistence.
package state

import (
	"context"
	"fmt"
	"path"

	"google.golang.org/protobuf/proto"
	"saxml/common/platform/env"

	pb "saxml/protobuf/admin_go_proto_grpc"
)

const stateFile = "state.proto"

// State persists the state of an admin server.
type State struct {
	// fsPath is the path to the directory containing state files.
	fsPath string
}

// Write writes a server state to the backing store.
func (s *State) Write(ctx context.Context, state *pb.State) error {
	path := path.Join(s.fsPath, stateFile)

	out, err := proto.Marshal(state)
	if err != nil {
		return err
	}

	return env.Get().WriteFileAtomically(ctx, path, out)
}

// Read reads a server state from the backing store.
func (s *State) Read(ctx context.Context) (*pb.State, error) {
	path := path.Join(s.fsPath, stateFile)

	// Return any file system error to let the caller handle it.
	exist, err := env.Get().FileExists(ctx, path)
	if err != nil {
		return nil, err
	}

	// Return an empty state for when a server runs for the first time.
	if !exist {
		return &pb.State{}, nil
	}

	// Read in the stored server state.
	in, err := env.Get().ReadFile(ctx, path)
	if err != nil {
		return nil, fmt.Errorf("Read %s error: %v", path, err)
	}
	state := &pb.State{}
	if err := proto.Unmarshal(in, state); err != nil {
		return nil, fmt.Errorf("State (%q) unmarshal error: %v", in, err)
	}
	return state, nil
}

// New creates a new admin server state with backing store.
func New(fsPath string) *State {
	return &State{fsPath: fsPath}
}
