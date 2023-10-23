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

	mmpb "saxml/protobuf/multimodal_go_proto_grpc"
	pbgrpc "saxml/protobuf/multimodal_go_proto_grpc"
)

// MultimodalModel represents a multimodal model in sax.
// Public methods are thread safe.
type MultimodalModel struct {
	model *Model
}

// Generate performs generation for `dataItems` on a multimodal model.
func (m *MultimodalModel) Generate(ctx context.Context, req *mmpb.GenerateRequest, options ...ModelOptionSetter) (*mmpb.GenerateResponse, error) {
	opts := NewModelOptions(options...)
	rpcReq := &mmpb.GenerateRpcRequest{
		ModelKey:    m.model.modelID,
		Request:     req,
		ExtraInputs: opts.ExtraInputs(),
	}

	var rpcResp *mmpb.GenerateRpcResponse
	err := m.model.run(ctx, "Generate", func(conn *grpc.ClientConn) error {
		var genErr error
		rpcResp, genErr = pbgrpc.NewMultimodalServiceClient(conn).Generate(ctx, rpcReq)
		return genErr
	})
	if err != nil {
		return nil, err
	}
	return rpcResp.GetResponse(), nil
}

// Score performs scoring for `prefixItems` and `suffixItems` on a multimodal model.
func (m *MultimodalModel) Score(ctx context.Context, req *mmpb.ScoreRequest, options ...ModelOptionSetter) (*mmpb.ScoreResponse, error) {
	opts := NewModelOptions(options...)
	rpcReq := &mmpb.ScoreRpcRequest{
		ModelKey:    m.model.modelID,
		Request:     req,
		ExtraInputs: opts.ExtraInputs(),
	}

	var rpcResp *mmpb.ScoreRpcResponse
	err := m.model.run(ctx, "Score", func(conn *grpc.ClientConn) error {
		var scoreErr error
		rpcResp, scoreErr = pbgrpc.NewMultimodalServiceClient(conn).Score(ctx, rpcReq)
		return scoreErr
	})
	if err != nil {
		return nil, err
	}
	return rpcResp.GetResponse(), nil
}
