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

	"google.golang.org/grpc"
	"saxml/common/retrier"

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
	err := l.model.run(ctx, "Score", func(conn *grpc.ClientConn) error {
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
	err := l.model.run(ctx, "generate", func(conn *grpc.ClientConn) error {
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

// GenerateStreamItem represents one partially or fully decoded suffix.
type GenerateStreamItem struct {
	Text      string
	PrefixLen int
	Score     float64
}

// StreamResult is the result for streaming generate.
type StreamResult struct {
	// Err is the error for current channel call.
	//   nil means no error;
	//   EOF means success;
	//   other error means failed streaming attempt.
	Err   error
	Items []GenerateStreamItem
}

func extractGenerateStreamResponse(res *pb.GenerateStreamResponse) []GenerateStreamItem {
	results := make([]GenerateStreamItem, 0, len(res.GetItems()))
	for _, item := range res.GetItems() {
		candidate := GenerateStreamItem{Text: item.GetText(), PrefixLen: int(item.GetPrefixLen()), Score: item.GetScore()}
		results = append(results, candidate)
	}
	return results
}

// GenerateStream performs streaming sampling decoding for `text` on a language model.
//
// Example:
//
//		var texts []string
//		var scores []float64
//		ch := lm.GenerateStream(ctx, prefix)
//		for res := range ch {
//			switch res.Err {
//			case nil:
//				for i, item := range res.Items {
//					if i >= len(texts) {
//						texts = append(texts, "")
//					}
//					texts[i] = texts[i][:item.PrefixLen] + item.Text
//					if i >= len(scores) {
//						scores = append(scores, 0.0)
//					}
//					scores[i] = item.Score
//				}
//			case io.EOF:
//	     log.Info("EOF")
//			default:
//				log.Fatal(err)
//			}
//		}
func (l *LanguageModel) GenerateStream(ctx context.Context, text string, options ...ModelOptionSetter) chan StreamResult {
	req := &pb.GenerateRequest{
		ModelKey:    l.model.modelID,
		Text:        text,
		ExtraInputs: NewModelOptions(options...).ExtraInputs(),
	}

	res := make(chan StreamResult)
	go func() {
		err := l.model.run(ctx, "generateStream", func(conn *grpc.ClientConn) error {
			client := pbgrpc.NewLMServiceClient(conn)
			stream, err := client.GenerateStream(ctx, req)
			if err != nil {
				return err
			}
			// If the model doesn't exist or is being loaded on the model server, the GenerateStream call
			// above doesn't return any error. Instead, the first Recv call below returns a NotFound
			// error. That's why we need special handling for the first Recv return value.
			first := true
			for {
				resp, err := stream.Recv()
				if err == nil {
					// Start streaming results to the res channel. Any error happening after this point
					// becomes a permanent error.
					res <- StreamResult{Items: extractGenerateStreamResponse(resp)}
					first = false
					continue
				}
				if err == io.EOF {
					// On successful completion of streaming, send io.EOF to the res channel and return
					// success (nil) to the retrier.
					res <- StreamResult{Err: err}
					return nil
				}
				if first {
					// Special handling for the first Recv call: return the error to the retrier to let it
					// decide what to do. The res channel is unused so far and can be used to stream correct
					// results if the retrier decides to retry.
					return err
				}
				// Once streaming has started, explicitly use permanent error to skip retries.
				return retrier.CreatePermanentError(err)
			}
		})
		if err != nil {
			// Errors getting returned from the retrier are non-retriable. Let users know about them.
			res <- StreamResult{Err: err}
		}
		close(res)
	}()
	return res
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
	err := l.model.run(ctx, "Embed", func(conn *grpc.ClientConn) error {
		var embErr error
		resp, embErr = pbgrpc.NewLMServiceClient(conn).Embed(ctx, req)
		return embErr
	})
	if err != nil {
		return nil, err
	}
	return resp.GetEmbedding(), nil
}
