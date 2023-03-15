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
	"fmt"
	"strings"

	"google.golang.org/grpc"

	pb "saxml/protobuf/modelet_go_proto_grpc"
	pbgrpc "saxml/protobuf/modelet_go_proto_grpc"
)

// Exporter exports methods of a model.
type Exporter struct {
	model *Model
}

// Export exports a method of the model managed by the Exporter.
func (e *Exporter) Export(ctx context.Context, methodNames []string, exportPath, rngSeedMode string, signatures []string) error {
	var reqRngSeedMode pb.ExportRequest_RngSeedMode
	switch strings.ToLower(rngSeedMode) {
	case "stateless":
		reqRngSeedMode = pb.ExportRequest_STATELESS
	case "stateful":
		reqRngSeedMode = pb.ExportRequest_STATEFUL
	case "fixed":
		reqRngSeedMode = pb.ExportRequest_FIXED
	default:
		return fmt.Errorf("invalid RNG seed mode \"%s\"", rngSeedMode)
	}
	req := &pb.ExportRequest{
		ModelKey:              e.model.modelID,
		MethodNames:           methodNames,
		ExportPath:            exportPath,
		SerializedModelFormat: pb.ExportRequest_TF_SAVEDMODEL_V0,
		RngSeedMode:           reqRngSeedMode,
		Signatures:            signatures,
	}
	export := func(conn *grpc.ClientConn) error {
		_, err := pbgrpc.NewModeletClient(conn).Export(ctx, req)
		return err
	}
	return e.model.run(ctx, "Export", export)
}
