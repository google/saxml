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

// Package sax provides a library to interact with sax admin and LM services.
package sax

import (
	"context"
	"fmt"
	"net"
	"time"

	log "github.com/golang/glog"
	"google.golang.org/grpc"
	"saxml/client/go/connection"
	"saxml/client/go/location"
	"saxml/client/go/saxadmin"
	"saxml/common/errors"
	"saxml/common/naming"
	"saxml/common/platform/env"
	"saxml/common/retrier"

	pb "saxml/protobuf/common_go_proto"
)

// RPC timeout, only intended for admin methods. Data methods have no timeout in general.
const timeout = 10 * time.Second

// Model represents a published model in the sax system.
// It's the entry point for creating task specific models such as `LanguageModel`.
//
// Example usage:
//
//	model := sax.Open("/sax/bar/glam");
//	languageModel := model.LM()
//	logP := languageModel.Score("....")
type Model struct {
	modelID  string
	location *location.Table // Keeps track a list of addresses for this model.
}

// run runs a callback function (`callMethod`) against sax system with retries through gRPC.
// `methodName` is only for logging purpose.
// `callMethod` is the callback function that performs model logic (e.g. score, sample).
func (m *Model) run(ctx context.Context, methodName string, callMethod func(conn *grpc.ClientConn) error) error {
	makeQuery := func() error {
		address, err := m.location.Pick(ctx)
		if err != nil {
			return err
		}
		modelServerConn, err := connection.GetOrCreate(ctx, address)
		if err == nil {
			err = callMethod(modelServerConn)
		}
		if errors.ServerShouldPoison(err) {
			m.location.Poison(address)
		}
		return err
	}
	err := retrier.Do(ctx, makeQuery, errors.ServerShouldRetry)
	if err != nil {
		log.V(1).Infof("%s() failed: %s", methodName, err)
		return err
	}
	return nil
}

// LM creates a language model.
func (m *Model) LM() *LanguageModel {
	return &LanguageModel{model: m}
}

// VM creates a vision model.
func (m *Model) VM() *VisionModel {
	return &VisionModel{model: m}
}

// AM creates an audio model.
func (m *Model) AM() *AudioModel {
	return &AudioModel{model: m}
}

// CM creates a custom model.
func (m *Model) CM() *CustomModel {
	return &CustomModel{model: m}
}

// Exporter creates a model exporter.
func (m *Model) Exporter() *Exporter {
	return &Exporter{model: m}
}

// Saver creates a model saver.
func (m *Model) Saver() *Saver {
	return &Saver{model: m}
}

// Options contains options for creating sax client.
// Default options are set in Open().
type Options struct {
	// `numConn` is the preferred number of modelet servers to connect to.
	numConn int
	// Add other possible options.
}

// OptionSetter are setters for sax options.
type OptionSetter func(*Options)

// WithNumConn sets the number of connections for sax.
func WithNumConn(num int) OptionSetter {
	return func(o *Options) {
		o.numConn = num
	}
}

// ModelOptions contains options for model methods.
type ModelOptions struct {
	kv  map[string]float32
	kvT map[string][]float32
}

// ExtraInputs creates a ExtraInputs proto from a ModelOptions.
func (mo *ModelOptions) ExtraInputs() *pb.ExtraInputs {
	tensors := make(map[string]*pb.Tensor)
	for key, value := range mo.kvT {
		tensors[key] = &pb.Tensor{Values: value}
	}
	return &pb.ExtraInputs{Items: mo.kv, Tensors: tensors}
}

// ModelOptionSetter are setters for sax options.
type ModelOptionSetter func(*ModelOptions)

// WithExtraInput sets options (key-value pairs) for the query.
func WithExtraInput(name string, value float32) ModelOptionSetter {
	return func(o *ModelOptions) {
		o.kv[name] = value
	}
}

// WithExtraInputTensor sets options (key-tensor pairs) for the query.
func WithExtraInputTensor(name string, value []float32) ModelOptionSetter {
	return func(o *ModelOptions) {
		o.kvT[name] = value
	}
}

// NewModelOptions creates a ModelOption by applying a list of key value pairs.
func NewModelOptions(setters ...ModelOptionSetter) *ModelOptions {
	opts := &ModelOptions{
		kv:  make(map[string]float32),
		kvT: make(map[string][]float32),
	}
	for _, op := range setters {
		op(opts)
	}
	return opts
}

// StartDebugPort starts a http server at `port` to get debug information.
func StartDebugPort(port int) {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("Debug port %d failed", port)
	}
	gRPCServer, err := env.Get().NewServer()
	if err != nil {
		log.Fatalf("Debug grpc server at port %d failed", port)
	}
	go func() {
		log.Infof("Debug server started at port %d.", port)
		gRPCServer.Serve(lis)
	}()
}

// Open returns a model interface for users to run inference with.
// `id` is a model ID of form /sax/<cell>/<model>, e.g. /sax/test/glam_64b64e.
func Open(id string, options ...OptionSetter) (*Model, error) {
	// Default options.
	opts := &Options{
		numConn: 3,
	}
	for _, s := range options {
		s(opts)
	}
	if opts.numConn <= 0 {
		return nil, fmt.Errorf("open() expect positive numConn %w", errors.ErrInvalidArgument)
	}
	modelID, err := naming.NewModelFullName(id)
	if err != nil {
		return nil, err
	}
	admin := saxadmin.Open(modelID.CellFullName())
	model := &Model{
		modelID:  id,
		location: location.NewLocationTable(admin, id, opts.numConn),
	}
	return model, nil
}
