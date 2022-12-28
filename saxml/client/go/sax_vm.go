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

	pb "saxml/protobuf/vision_go_proto_grpc"
	pbgrpc "saxml/protobuf/vision_go_proto_grpc"
)

// VisionModel represents a vision model in sax.
// Public methods are thread safe.
type VisionModel struct {
	model *Model
}

// ClassifyResult is a tuple of text and score as the result for classification operation.
type ClassifyResult struct {
	Text  string
	Score float64
}

func extractClassfyResponse(res *pb.ClassifyResponse) []ClassifyResult {
	var result []ClassifyResult
	for _, one := range res.GetTexts() {
		candidate := ClassifyResult{Text: one.GetText(), Score: one.GetScore()}
		result = append(result, candidate)
	}
	return result
}

// Classify performs classificiation for a serialized image (`imageBytes`) against a vision model.
func (v *VisionModel) Classify(ctx context.Context, imageBytes []byte, options ...ModelOptionSetter) ([]ClassifyResult, error) {
	opts := NewModelOptions(options...)
	req := &pb.ClassifyRequest{
		ModelKey:    v.model.modelID,
		ImageBytes:  imageBytes,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.ClassifyResponse
	err := v.model.run(ctx, "Classify", func(conn *grpc.ClientConn) error {
		var classifyErr error
		resp, classifyErr = pbgrpc.NewVisionServiceClient(conn).Classify(ctx, req)
		return classifyErr
	})
	if err != nil {
		return nil, err
	}
	res := extractClassfyResponse(resp)
	return res, nil
}

// GeneratedImage is a tuple of image (in bytes array) and its log probability.
type GeneratedImage struct {
	Image []byte
	Logp  float64
}

func extractGeneratedImageResponse(res *pb.TextToImageResponse) []GeneratedImage {
	var result []GeneratedImage
	for _, one := range res.GetImages() {
		candidate := GeneratedImage{Image: one.GetImage(), Logp: one.GetScore()}
		result = append(result, candidate)
	}
	return result
}

// TextToImage generates a list of image (`imageBytes`) and log probability for a given text.
func (v *VisionModel) TextToImage(ctx context.Context, text string, options ...ModelOptionSetter) ([]GeneratedImage, error) {
	opts := NewModelOptions(options...)
	req := &pb.TextToImageRequest{
		ModelKey:    v.model.modelID,
		Text:        text,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.TextToImageResponse
	err := v.model.run(ctx, "TextToImage", func(conn *grpc.ClientConn) error {
		var textToImageErr error
		resp, textToImageErr = pbgrpc.NewVisionServiceClient(conn).TextToImage(ctx, req)
		return textToImageErr
	})
	if err != nil {
		return nil, err
	}
	res := extractGeneratedImageResponse(resp)
	return res, nil
}

// Embed performs embedding for an image as byte array.
func (v *VisionModel) Embed(ctx context.Context, imageBytes []byte, options ...ModelOptionSetter) ([]float64, error) {
	opts := NewModelOptions(options...)
	req := &pb.EmbedRequest{
		ModelKey:    v.model.modelID,
		ImageBytes:  imageBytes,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.EmbedResponse
	err := v.model.run(ctx, "Embed", func(conn *grpc.ClientConn) error {
		var sampleErr error
		resp, sampleErr = pbgrpc.NewVisionServiceClient(conn).Embed(ctx, req)
		return sampleErr
	})
	if err != nil {
		return nil, err
	}
	return resp.GetEmbedding(), nil
}

// DetectResult contains a representation for a single bounding box.
type DetectResult struct {
	CenterX float64
	CenterY float64
	Width   float64
	Height  float64
	Text    string
	Score   float64
}

func extractBoundingBox(res *pb.DetectResponse) []DetectResult {
	var result []DetectResult
	for _, one := range res.GetBoundingBoxes() {
		candidate := DetectResult{
			CenterX: one.GetCx(),
			CenterY: one.GetCy(),
			Width:   one.GetW(),
			Height:  one.GetH(),
			Text:    one.GetText(),
			Score:   one.GetScore()}
		result = append(result, candidate)
	}
	return result
}

func (v *VisionModel) Detect(ctx context.Context, imageBytes []byte, text []string, options ...ModelOptionSetter) ([]DetectResult, error) {
	opts := NewModelOptions(options...)
	req := &pb.DetectRequest{
		ModelKey:    v.model.modelID,
		ImageBytes:  imageBytes,
		Text:        text,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.DetectResponse
	err := v.model.run(ctx, "Detect", func(conn *grpc.ClientConn) error {
		var detectErr error
		resp, detectErr = pbgrpc.NewVisionServiceClient(conn).Detect(ctx, req)
		return detectErr
	})
	if err != nil {
		return nil, err
	}
	res := extractBoundingBox(resp)
	return res, nil
}

// ImageToTextResult is a tuple of text and score as the result for image-to-text operation.
type ImageToTextResult struct {
	Text  string
	Score float64
}

func extractImageToTextResponse(res *pb.ImageToTextResponse) []ImageToTextResult {
	var result []ImageToTextResult
	for _, one := range res.GetTexts() {
		candidate := ImageToTextResult{Text: one.GetText(), Score: one.GetScore()}
		result = append(result, candidate)
	}
	return result
}

// ImageToText performs captioning for a serialized image (`imageBytes`) against a vision model.
func (v *VisionModel) ImageToText(ctx context.Context, imageBytes []byte, text string, options ...ModelOptionSetter) ([]ImageToTextResult, error) {
	opts := NewModelOptions(options...)
	req := &pb.ImageToTextRequest{
		ModelKey:    v.model.modelID,
		ImageBytes:  imageBytes,
		Text:        text,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.ImageToTextResponse
	err := v.model.run(ctx, "ImageToText", func(conn *grpc.ClientConn) error {
		var imageToTextErr error
		resp, imageToTextErr = pbgrpc.NewVisionServiceClient(conn).ImageToText(ctx, req)
		return imageToTextErr
	})
	if err != nil {
		return nil, err
	}
	res := extractImageToTextResponse(resp)
	return res, nil
}

// VideoToTextResult is a tuple of text and score as the result for video-to-text operation.
type VideoToTextResult struct {
	Text  string
	Score float64
}

func extractVideoToTextResponse(res *pb.VideoToTextResponse) []VideoToTextResult {
	var result []VideoToTextResult
	for _, one := range res.GetTexts() {
		candidate := VideoToTextResult{Text: one.GetText(), Score: one.GetScore()}
		result = append(result, candidate)
	}
	return result
}

// VideoToText performs captioning for multiple image frames against a vision model.
// Specifically:
// - 'imageFrames' is a list of bytes where each element is a serialized image frame.
// - 'text' is the (optional) prefix text for prefix decoding.
func (v *VisionModel) VideoToText(ctx context.Context, imageFrames [][]byte, text string, options ...ModelOptionSetter) ([]VideoToTextResult, error) {
	opts := NewModelOptions(options...)
	req := &pb.VideoToTextRequest{
		ModelKey:    v.model.modelID,
		ImageFrames: imageFrames,
		Text:        text,
		ExtraInputs: opts.ExtraInputs(),
	}

	var resp *pb.VideoToTextResponse
	err := v.model.run(ctx, "VideoToText", func(conn *grpc.ClientConn) error {
		var videoToTextErr error
		resp, videoToTextErr = pbgrpc.NewVisionServiceClient(conn).VideoToText(ctx, req)
		return videoToTextErr
	})
	if err != nil {
		return nil, err
	}
	res := extractVideoToTextResponse(resp)
	return res, nil
}
