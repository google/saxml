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

package sax

import (
	"context"

	"google.golang.org/grpc"

	pb "saxml/protobuf/modelet_go_proto_grpc"
	pbgrpc "saxml/protobuf/modelet_go_proto_grpc"
)

// Saver saves the checkpoint of a model.
type Saver struct {
	model *Model
}

// Save checkpoint of the model.
func (e *Saver) Save(ctx context.Context, checkpointPath string) error {
	req := &pb.SaveRequest{
		ModelKey:       e.model.modelID,
		CheckpointPath: checkpointPath,
	}
	save := func(conn *grpc.ClientConn) error {
		_, err := pbgrpc.NewModeletClient(conn).Save(ctx, req)
		return err
	}
	return e.model.run(ctx, "Save", save)
}
