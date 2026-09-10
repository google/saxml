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
	"io"

	log "github.com/golang/glog"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
	"saxml/common/retrier"

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

// CustomStreamResult represents one response or error in streaming custom call.
type CustomStreamResult struct {
	Err      error
	Response []byte
}

// CustomStream performs streaming custom call against a Custom model.
func (m *CustomModel) CustomStream(ctx context.Context, request []byte, methodName string, options ...ModelOptionSetter) chan CustomStreamResult {
	opts := NewModelOptions(options...)
	req := &pb.CustomRequest{
		ModelKey:    m.model.modelID,
		Request:     request,
		ExtraInputs: opts.ExtraInputs(),
		MethodName:  methodName,
	}

	res := make(chan CustomStreamResult)
	go func() {
		var trailer metadata.MD
		err := m.model.run(ctx, "customStream", func(conn *grpc.ClientConn) error {
			client := pbgrpc.NewCustomServiceClient(conn)
			stream, err := client.CustomStream(ctx, req, grpc.Trailer(&trailer))
			if err != nil {
				return err
			}
			if err := opts.ExtractQueryCost(&trailer); err != nil {
				log.Errorf("ExtractQueryCost: %v", err)
			}
			first := true
			for {
				resp, err := stream.Recv()
				if err == nil {
					res <- CustomStreamResult{Response: resp.GetResponse()}
					first = false
					continue
				}
				if err == io.EOF {
					res <- CustomStreamResult{Err: err}
					return nil
				}
				if first {
					return err
				}
				return retrier.CreatePermanentError(err)
			}
		})
		if err != nil {
			res <- CustomStreamResult{Err: err}
		}
		close(res)
	}()
	return res
}
