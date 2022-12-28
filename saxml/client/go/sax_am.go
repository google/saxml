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

	pb "saxml/protobuf/audio_go_proto_grpc"
	pbgrpc "saxml/protobuf/audio_go_proto_grpc"
)

// AudioModel represents an audio model in sax.
// Public methods are thread safe.
type AudioModel struct {
	model *Model
}

// AsrHypothesis is a tuple of text and score.
type AsrHypothesis struct {
	Text  string
	Score float64
}

func extractAsrResponse(res *pb.AsrResponse) []AsrHypothesis {
	var result []AsrHypothesis
	for _, one := range res.GetHyps() {
		candidate := AsrHypothesis{Text: one.GetText(), Score: one.GetScore()}
		result = append(result, candidate)
	}
	return result
}

// Recognize against an ASR model.
func (m *AudioModel) Recognize(ctx context.Context, audioBytes []byte, options ...ModelOptionSetter) ([]AsrHypothesis, error) {
	opts := NewModelOptions(options...)
	req := &pb.AsrRequest{
		ModelKey:    m.model.modelID,
		AudioBytes:  audioBytes,
		AudioFormat: pb.AsrRequest_WAVEFORM_FILE,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.AsrResponse
	err := m.model.run(ctx, "SpeechRecognition", func(conn *grpc.ClientConn) error {
		var asrErr error
		resp, asrErr = pbgrpc.NewAudioServiceClient(conn).Recognize(ctx, req)
		return asrErr
	})
	if err != nil {
		return nil, err
	}
	res := extractAsrResponse(resp)
	return res, nil
}
