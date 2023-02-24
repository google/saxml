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

	pb "saxml/protobuf/custom_go_proto_grpc"
	pbgrpc "saxml/protobuf/custom_go_proto_grpc"
)

// CustomModel represents a custom model in sax.
// Public methods are thread safe.
type CustomModel struct {
	model *Model
}

// Custom call against a Custom model.
func (m *CustomModel) Custom(ctx context.Context, request []byte, methodName string, options ...ModelOptionSetter) ([]byte, error) {
	opts := NewModelOptions(options...)
	req := &pb.CustomRequest{
		ModelKey:    m.model.modelID,
		Request:     request,
		ExtraInputs: opts.ExtraInputs(),
		MethodName:  methodName,
	}

	var resp *pb.CustomResponse
	err := m.model.run(ctx, "CustomCall", func(conn *grpc.ClientConn) error {
		var customCallErr error
		resp, customCallErr = pbgrpc.NewCustomServiceClient(conn).Custom(ctx, req)
		return customCallErr
	})
	if err != nil {
		return []byte{}, err
	}
	return resp.Response, nil
}
