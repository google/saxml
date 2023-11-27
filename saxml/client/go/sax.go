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
	"strconv"
	"strings"
	"time"

	log "github.com/golang/glog"
	"github.com/cenkalti/backoff"
	"google.golang.org/grpc"
	"google.golang.org/grpc/metadata"
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
	modelID           string
	connectionFactory connection.Factory
	retryingBehavior  func(err error) bool
}

// QueryCost represents the cost of the query.
type QueryCost struct {
	// Cost measured in TPU milliseconds
	TpuMs int
}

// run runs a callback function (`callMethod`) against sax system with retries through gRPC.
// `methodName` is only for logging purpose.
// `callMethod` is the callback function that performs model logic (e.g. score, sample).
func (m *Model) run(ctx context.Context, methodName string, callMethod func(conn *grpc.ClientConn) error) error {
	makeQuery := func() error {
		modelServerConn, err := m.connectionFactory.GetOrCreate(ctx)
		if err == nil {
			err = callMethod(modelServerConn)
		} else if errors.IsNotFound(err) {
			// If the model does not exist anymore, no point to retry.
			err = backoff.Permanent(err)
		}
		return err
	}
	err := retrier.Do(ctx, makeQuery, m.retryingBehavior)
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

// MM creates a multimodal model.
func (m *Model) MM() *MultimodalModel {
	return &MultimodalModel{model: m}
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
	// `proxyAddr` is the (optional) proxy to route SAX model traffic through.
	proxyAddr string
	// `failFast` disables some retrying behavior when true. Useful for when the model servers are unresponsive.
	failFast bool
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

// WithProxy redirects all SAX model traffic via a proxy.
func WithProxy(addr string) OptionSetter {
	return func(o *Options) {
		o.proxyAddr = addr
	}
}

// WithFailFast changes the retry behaviour to not wait for servers to become available.
func WithFailFast(failFast bool) OptionSetter {
	return func(o *Options) {
		o.failFast = failFast
	}
}

// ModelOptions contains options for model methods.
type ModelOptions struct {
	kv        map[string]float32
	kvT       map[string][]float32
	kvS       map[string]string
	queryCost *QueryCost
}

// ExtraInputs creates a ExtraInputs proto from a ModelOptions.
func (mo *ModelOptions) ExtraInputs() *pb.ExtraInputs {
	tensors := make(map[string]*pb.Tensor)
	for key, value := range mo.kvT {
		tensors[key] = &pb.Tensor{Values: value}
	}
	return &pb.ExtraInputs{Items: mo.kv, Tensors: tensors, Strings: mo.kvS}
}

// ExtractQueryCost extracts query costs from metadata and adds it to model options.
func (mo *ModelOptions) ExtractQueryCost(md *metadata.MD) error {
	if mo.queryCost == nil {
		return nil
	}
	if tr := md.Get("query_cost_v0"); mo.queryCost != nil && len(tr) > 0 {
		queryCost, err := strconv.Atoi(tr[0])
		if err != nil {
			return fmt.Errorf("metadata.Get(query_cost_v0) expected int, got %s", tr[0])
		}
		(*mo.queryCost).TpuMs = queryCost
	}
	return nil
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

// WithExtraInputString sets options (key-string pairs) for the query.
func WithExtraInputString(name string, value string) ModelOptionSetter {
	return func(o *ModelOptions) {
		o.kvS[name] = value
	}
}

// WithQueryCost fetches and outputs query costs after the model runs a query.
func WithQueryCost(cost *QueryCost) ModelOptionSetter {
	return func(o *ModelOptions) {
		o.queryCost = cost
	}
}

// NewModelOptions creates a ModelOption by applying a list of key value pairs.
func NewModelOptions(setters ...ModelOptionSetter) *ModelOptions {
	opts := &ModelOptions{
		kv:  make(map[string]float32),
		kvT: make(map[string][]float32),
		kvS: make(map[string]string),
	}
	for _, op := range setters {
		op(opts)
	}
	return opts
}

// StartDebugPort starts a http server at `port` to get debug information.
func StartDebugPort(ctx context.Context, port int) {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("Debug port %d failed", port)
	}
	gRPCServer, err := env.Get().NewServer(ctx)
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
		numConn: 37,
	}
	for _, s := range options {
		s(opts)
	}
	if opts.numConn <= 0 {
		return nil, fmt.Errorf("open() expect positive numConn %w", errors.ErrInvalidArgument)
	}

	retryingBehavior := errors.ServerShouldRetry
	if opts.failFast {
		retryingBehavior = errors.IsNotFound
	}
	if opts.proxyAddr != "" {
		model := &Model{
			modelID:           id,
			connectionFactory: &connection.DirectConnectionFactory{Address: opts.proxyAddr},
			retryingBehavior:  retryingBehavior,
		}
		return model, nil
	}
	if strings.HasPrefix(id, "google:///") {
		// This is a self-hosted model.Skip sax cell resolution and connect to it directly.
		model := &Model{
			modelID:           id,
			connectionFactory: &connection.DirectConnectionFactory{Address: id},
			retryingBehavior:  retryingBehavior,
		}
		return model, nil
	}
	modelID, err := naming.NewModelFullName(id)
	if err != nil {
		return nil, err
	}
	admin := saxadmin.Open(modelID.CellFullName())
	model := &Model{
		modelID:           id,
		connectionFactory: connection.SaxConnectionFactory{Location: location.NewLocationTable(admin, id, opts.numConn)},
		retryingBehavior:  retryingBehavior,
	}
	return model, nil
}
