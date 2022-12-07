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

	pb "saxml/protobuf/lm_go_proto_grpc"
	pbgrpc "saxml/protobuf/lm_go_proto_grpc"
)

// LanguageModel represents a language model in sax.
// Public methods are thread safe.
type LanguageModel struct {
	model *Model
}

// Score performs scoring for a `prefix`, `suffix` pair on a language model.
// Note: Score() does not manipulate prefix or suffix; users add <EOS> explicitly if needed.
func (l *LanguageModel) Score(ctx context.Context, prefix string, suffix []string, options ...ModelOptionSetter) ([]float64, error) {
	opts := NewModelOptions(options...)
	req := &pb.ScoreRequest{
		ModelKey:    l.model.modelID,
		Suffix:      suffix,
		Prefix:      prefix,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.ScoreResponse
	err := l.model.runGRPC(ctx, "Score", func(conn *grpc.ClientConn) error {
		var scoreErr error
		resp, scoreErr = pbgrpc.NewLMServiceClient(conn).Score(ctx, req)
		return scoreErr
	})
	if err != nil {
		return []float64{0.0}, err
	}

	logP := resp.GetLogp()
	return logP, nil
}

// GenerateResult is a tuple of text and score as the result for generate operation.
type GenerateResult struct {
	Text  string
	Score float64
}

func extractGenerateResponse(res *pb.GenerateResponse) []GenerateResult {
	results := make([]GenerateResult, 0, len(res.GetTexts()))
	for _, one := range res.GetTexts() {
		candidate := GenerateResult{Text: one.GetText(), Score: one.GetScore()}
		results = append(results, candidate)
	}
	return results
}

// Generate performs sampling decoding for `text` on a language model.
func (l *LanguageModel) Generate(ctx context.Context, text string, options ...ModelOptionSetter) ([]GenerateResult, error) {
	opts := NewModelOptions(options...)
	req := &pb.GenerateRequest{
		ModelKey:    l.model.modelID,
		Text:        text,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.GenerateResponse
	err := l.model.runGRPC(ctx, "generate", func(conn *grpc.ClientConn) error {
		var sampleErr error
		resp, sampleErr = pbgrpc.NewLMServiceClient(conn).Generate(ctx, req)
		return sampleErr
	})
	if err != nil {
		return nil, err
	}
	res := extractGenerateResponse(resp)
	return res, nil
}

// Embed performs embedding for a text.
func (l *LanguageModel) Embed(ctx context.Context, text string, options ...ModelOptionSetter) ([]float64, error) {
	opts := NewModelOptions(options...)
	req := &pb.EmbedRequest{
		ModelKey:    l.model.modelID,
		Text:        text,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.EmbedResponse
	err := l.model.runGRPC(ctx, "Embed", func(conn *grpc.ClientConn) error {
		var embErr error
		resp, embErr = pbgrpc.NewLMServiceClient(conn).Embed(ctx, req)
		return embErr
	})
	if err != nil {
		return nil, err
	}
	return resp.GetEmbedding(), nil
}
