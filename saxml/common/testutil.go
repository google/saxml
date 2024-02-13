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

// Package testutil provides helper functions to set up test environments.
package testutil

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	log "github.com/golang/glog"
	// unused internal test dependency
	"google.golang.org/protobuf/proto"
	"saxml/common/addr"
	"saxml/common/cell"
	"saxml/common/config"
	"saxml/common/errors"
	"saxml/common/naming"
	"saxml/common/platform/env"
	"saxml/common/watchable"

	apb "saxml/protobuf/admin_go_proto_grpc"
	agrpc "saxml/protobuf/admin_go_proto_grpc"
	ampb "saxml/protobuf/audio_go_proto_grpc"
	amgrpc "saxml/protobuf/audio_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
	cmpb "saxml/protobuf/custom_go_proto_grpc"
	cmgrpc "saxml/protobuf/custom_go_proto_grpc"
	lmpb "saxml/protobuf/lm_go_proto_grpc"
	lmgrpc "saxml/protobuf/lm_go_proto_grpc"
	mpb "saxml/protobuf/modelet_go_proto_grpc"
	mgrpc "saxml/protobuf/modelet_go_proto_grpc"
	mmpb "saxml/protobuf/multimodal_go_proto_grpc"
	mmgrpc "saxml/protobuf/multimodal_go_proto_grpc"
	vmpb "saxml/protobuf/vision_go_proto_grpc"
	vmgrpc "saxml/protobuf/vision_go_proto_grpc"
)

func setUpInternal(ctx context.Context, saxCell string, fsRoot string, adminACL string) error {
	if err := env.Get().CreateDir(ctx, cell.Sax(ctx), ""); err != nil {
		return err
	}
	if _, err := naming.SaxCellToCell(saxCell); err != nil {
		return err
	}
	if err := cell.Create(ctx, saxCell, adminACL); err != nil {
		return err
	}
	if err := config.Create(ctx, saxCell, fsRoot, adminACL); err != nil {
		return err
	}
	return nil
}

// SetUpInternal is exported for the C wrapper.
func SetUpInternal(ctx context.Context, saxCell string, fsRoot string) error {
	return setUpInternal(ctx, saxCell, fsRoot, "")
}

// SetUp creates the necessary environment for a given Sax cell name.
//
// Tests in the same process should use different Sax cells.
func SetUp(ctx context.Context, t *testing.T, saxCell string, adminACL string) {
	t.Helper()
	fsRoot := t.TempDir()
	if err := setUpInternal(ctx, saxCell, fsRoot, adminACL); err != nil {
		t.Fatalf("SetUp(%v, %v) failed: %v", saxCell, fsRoot, err)
	}
}

type stubAdminServer struct {
	saxCell        string
	modelAddresses *watchable.Watchable
}

func (s *stubAdminServer) Publish(ctx context.Context, in *apb.PublishRequest) (*apb.PublishResponse, error) {
	return &apb.PublishResponse{}, nil
}

func (s *stubAdminServer) Update(ctx context.Context, in *apb.UpdateRequest) (*apb.UpdateResponse, error) {
	return &apb.UpdateResponse{}, nil
}

func (s *stubAdminServer) Unpublish(ctx context.Context, in *apb.UnpublishRequest) (*apb.UnpublishResponse, error) {
	return &apb.UnpublishResponse{}, nil
}

func (s *stubAdminServer) List(ctx context.Context, in *apb.ListRequest) (*apb.ListResponse, error) {
	addresses := s.modelAddressesList(ctx)
	if in.GetModelId() == "" {
		// This is "listall".
		return &apb.ListResponse{}, nil
	}
	// This is "list". Echo the queried model name.
	fullName, err := naming.NewModelFullName(in.GetModelId())
	if err != nil {
		return nil, err
	}
	if fullName.CellFullName() != s.saxCell {
		return nil, fmt.Errorf("Want %v, got %v: %w", s.saxCell, fullName.CellFullName(), errors.ErrInvalidArgument)
	}
	out := &apb.ListResponse{
		PublishedModels: []*apb.PublishedModel{
			&apb.PublishedModel{
				Model: &apb.Model{
					ModelId:              in.GetModelId(),
					ModelPath:            "/sax/models/xyz",
					CheckpointPath:       "/tmp/abc",
					RequestedNumReplicas: 1,
					Overrides:            map[string]string{"foo": "bar"},
				},
				ModeletAddresses: []string{addresses[0]},
			},
		},
	}
	return out, nil
}

func (s *stubAdminServer) modelAddressesList(ctx context.Context) []string {
	result, err := s.modelAddresses.Watch(ctx, 0)
	if err != nil {
		log.Fatalf("Unexpected modelAddressesList error: %v", err)
	}
	dataset := result.Data
	if dataset == nil {
		dataset = watchable.NewDataSet()
	}
	dataset.Apply(result.Log)
	return dataset.ToList()
}

func (s *stubAdminServer) WatchLoc(ctx context.Context, in *apb.WatchLocRequest) (*apb.WatchLocResponse, error) {
	result, err := s.modelAddresses.Watch(ctx, in.GetSeqno())
	if err != nil {
		return nil, err
	}
	return &apb.WatchLocResponse{
		AdminServerId: "localhost:12345_01234567789abcdef",
		Result:        result.ToProto(),
	}, nil
}

func (s *stubAdminServer) WaitForReady(ctx context.Context, in *apb.WaitForReadyRequest) (*apb.WaitForReadyResponse, error) {
	return &apb.WaitForReadyResponse{}, nil
}

func (s *stubAdminServer) Join(ctx context.Context, in *apb.JoinRequest) (*apb.JoinResponse, error) {
	addr := in.GetAddress()
	if !strings.HasPrefix(addr, "localhost:") {
		return nil, fmt.Errorf("Model address %q should start with \"localhost:\"", addr)
	}
	s.modelAddresses.Add(addr)
	return &apb.JoinResponse{}, nil
}

func (s *stubAdminServer) Stats(ctx context.Context, in *apb.StatsRequest) (*apb.StatsResponse, error) {
	var modelServerTypeStats []*apb.ModelServerTypeStat
	modelServerTypeStats = append(modelServerTypeStats, &apb.ModelServerTypeStat{
		ChipType:     apb.ModelServer_CHIP_TYPE_TPU_V2,
		ChipTopology: apb.ModelServer_CHIP_TOPOLOGY_1X1,
		NumReplicas:  int32(len(s.modelAddressesList(ctx))),
	})

	return &apb.StatsResponse{ModelServerTypeStats: modelServerTypeStats}, nil
}

// StartStubAdminServer starts a new admin server with stub implementations.
// Close the returned channel to close the server.
func StartStubAdminServer(adminPort int, modelPorts []int, saxCell string) (chan struct{}, error) {
	ctx := context.Background()
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", adminPort))
	if err != nil {
		return nil, err
	}

	gRPCServer, err := env.Get().NewServer(ctx)
	if err != nil {
		return nil, err
	}

	addresses := watchable.New()
	for _, port := range modelPorts {
		addresses.Add(fmt.Sprintf("localhost:%d", port))
	}
	adminServer := &stubAdminServer{
		saxCell:        saxCell,
		modelAddresses: addresses,
	}
	agrpc.RegisterAdminServer(gRPCServer.GRPCServer(), adminServer)

	c, err := addr.SetAddr(ctx, adminPort, saxCell)
	if err != nil {
		return nil, err
	}

	go gRPCServer.Serve(lis)
	closer := make(chan struct{})
	go func() {
		<-closer
		gRPCServer.Stop()
		close(c)
	}()

	return closer, nil
}

// StartStubAdminServerT starts a new admin server with stub implementations.
// It is automatically closed when the test ends.
func StartStubAdminServerT(t *testing.T, adminPort int, modelPorts []int, saxCell string) {
	t.Helper()
	ch, err := StartStubAdminServer(adminPort, modelPorts, saxCell)
	if err != nil {
		t.Fatalf("StartStubAdminServer failed: %v", err)
	}
	t.Cleanup(func() { close(ch) })
}

// CallAdminServer calls a gRPC method on the admin server for saxCell and returns the result.
func CallAdminServer(ctx context.Context, saxCell string, req any) (resp any, err error) {
	addr, err := addr.FetchAddr(ctx, saxCell)
	if err != nil {
		return nil, err
	}
	conn, err := env.Get().DialContext(ctx, addr)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	client := agrpc.NewAdminClient(conn)

	switch req := req.(type) {
	case *apb.WatchLocRequest:
		return client.WatchLoc(ctx, req)
	default:
		return nil, fmt.Errorf("Unknown request type %T", req)
	}
}

type stubModeletServer struct {
	loadDelay time.Duration

	mu           sync.Mutex
	loadedModels map[string]bool // model key as key
}

func (s *stubModeletServer) Load(ctx context.Context, in *mpb.LoadRequest) (*mpb.LoadResponse, error) {
	if s.loadDelay > 0 {
		time.Sleep(s.loadDelay)
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.loadedModels[in.GetModelKey()] = true
	return &mpb.LoadResponse{}, nil
}

func (s *stubModeletServer) UpdateLoaded(ctx context.Context, in *mpb.UpdateLoadedRequest) (*mpb.UpdateLoadedResponse, error) {
	return &mpb.UpdateLoadedResponse{}, nil
}

func (s *stubModeletServer) Unload(ctx context.Context, in *mpb.UnloadRequest) (*mpb.UnloadResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.loadedModels, in.GetModelKey())
	return &mpb.UnloadResponse{}, nil
}

func (s *stubModeletServer) Export(ctx context.Context, in *mpb.ExportRequest) (*mpb.ExportResponse, error) {
	return nil, errors.ErrUnimplemented
}

func (s *stubModeletServer) Save(ctx context.Context, in *mpb.SaveRequest) (*mpb.SaveResponse, error) {
	return nil, errors.ErrUnimplemented
}

func (s *stubModeletServer) GetStatus(ctx context.Context, in *mpb.GetStatusRequest) (*mpb.GetStatusResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	var models []*mpb.GetStatusResponse_ModelWithStatus
	for key := range s.loadedModels {
		model := &mpb.GetStatusResponse_ModelWithStatus{
			ModelKey:    key,
			ModelStatus: cpb.ModelStatus_LOADED,
		}
		models = append(models, model)
	}
	return &mpb.GetStatusResponse{Models: models}, nil
}

type stubLanguageModelServer struct {
	scoreDelay       time.Duration
	unavailableModel string
}

func (s *stubLanguageModelServer) Score(ctx context.Context, in *lmpb.ScoreRequest) (*lmpb.ScoreResponse, error) {
	prefix := in.GetPrefix()
	suffixes := in.GetSuffix()
	extra := in.GetExtraInputs().GetItems()
	temperature := 1.0
	if val, found := extra["temperature"]; found {
		temperature = float64(val)
	}
	extraVector := []float64{}
	extraT := in.GetExtraInputs().GetTensors()
	if val, found := extraT["extra_tensor"]; found {
		vector := val.GetValues()
		for _, value := range vector {
			extraVector = append(extraVector, float64(value))
		}
	}
	extraString := ""
	extraS := in.GetExtraInputs().GetStrings()
	if val, found := extraS["extra_string"]; found {
		extraString = val
	}
	product := 1.0
	for _, value := range extraVector {
		product *= value
	}
	time.Sleep(s.scoreDelay)
	var logP []float64
	for _, suffix := range suffixes {
		logP = append(logP, float64(len(prefix)+len(suffix)+len(extraString))*0.1*temperature*product)
	}
	return &lmpb.ScoreResponse{
		Logp: logP,
	}, nil
}

func (s *stubLanguageModelServer) Generate(ctx context.Context, in *lmpb.GenerateRequest) (*lmpb.GenerateResponse, error) {
	if in.GetModelKey() == s.unavailableModel {
		return nil, errors.ErrNotFound
	}
	text := in.GetText()
	if text == "bad-input" {
		return nil, fmt.Errorf("bad input %w", errors.ErrInvalidArgument)
	}
	extra := in.GetExtraInputs().GetItems()
	temperature := 1.0
	if val, found := extra["temperature"]; found {
		temperature = float64(val)
	}
	extraVector := []float64{}
	extraT := in.GetExtraInputs().GetTensors()
	if val, found := extraT["extra_tensor"]; found {
		vector := val.GetValues()
		for _, value := range vector {
			extraVector = append(extraVector, float64(value))
		}
	}
	extraString := ""
	extraS := in.GetExtraInputs().GetStrings()
	if val, found := extraS["extra_string"]; found {
		extraString = val
	}
	product := 1.0
	for _, value := range extraVector {
		product *= value
	}
	return &lmpb.GenerateResponse{
		Texts: []*lmpb.DecodedText{
			&lmpb.DecodedText{
				Text:  text + "_0",
				Score: float64(len(text)+len(extraString)) * 0.1 * temperature * product,
			},
			&lmpb.DecodedText{
				Text:  text + "_1",
				Score: float64(len(text)+len(extraString)) * 0.2 * temperature * product,
			},
		},
	}, nil
}

func (s *stubLanguageModelServer) GenerateStream(in *lmpb.GenerateRequest, stream lmgrpc.LMService_GenerateStreamServer) error {
	if in.GetModelKey() == s.unavailableModel {
		return errors.ErrNotFound
	}
	text := in.GetText()
	extra := in.GetExtraInputs().GetItems()
	temperature := 1.0
	if val, found := extra["temperature"]; found {
		temperature = float64(val)
	}
	for i := 0; i < 10; i++ {
		response := &lmpb.GenerateStreamResponse{
			Items: []*lmpb.GenerateStreamItem{
				&lmpb.GenerateStreamItem{
					Text: text + "_0_" + strconv.Itoa(i),
					// TODO(sax-dev): Also fill in PrefixLen for additional testing.
					Scores: []float64{float64(len(text)) * 0.1 * temperature},
				},
				&lmpb.GenerateStreamItem{
					Text:   text + "_1_" + strconv.Itoa(i),
					Scores: []float64{float64(len(text)) * 0.2 * temperature},
				},
			},
		}
		if err := stream.Send(response); err != nil {
			return err
		}
	}
	return nil
}

func (s *stubLanguageModelServer) Embed(ctx context.Context, in *lmpb.EmbedRequest) (*lmpb.EmbedResponse, error) {
	value := float64(len(in.GetText()))
	return &lmpb.EmbedResponse{
		Embedding: []float64{
			value * 0.6,
			value * 0.5,
			value * 0.3,
			value * 0.4,
		},
	}, nil
}

func (s *stubLanguageModelServer) Gradient(ctx context.Context, in *lmpb.GradientRequest) (*lmpb.GradientResponse, error) {
	return &lmpb.GradientResponse{
		Score: []float64{float64(len(in.GetPrefix())) * 0.2},
		Gradients: map[string]*lmpb.GradientResponse_Gradient{
			"gradient": &lmpb.GradientResponse_Gradient{
				Values: []float64{float64(len(in.GetSuffix())) * 0.8},
			},
		},
	}, nil
}

type stubVisionModelServer struct{}

func (s *stubVisionModelServer) Classify(ctx context.Context, in *vmpb.ClassifyRequest) (*vmpb.ClassifyResponse, error) {
	text := string(in.GetImageBytes())
	return &vmpb.ClassifyResponse{
		Texts: []*vmpb.DecodedText{
			&vmpb.DecodedText{
				Text:  text + "_0",
				Score: float64(len(text)) * 0.1,
			},
			&vmpb.DecodedText{
				Text:  text + "_1",
				Score: float64(len(text)) * 0.2,
			},
		},
	}, nil
}

func (s *stubVisionModelServer) TextToImage(ctx context.Context, in *vmpb.TextToImageRequest) (*vmpb.TextToImageResponse, error) {
	text := in.GetText()
	return &vmpb.TextToImageResponse{
		Images: []*vmpb.ImageGenerations{
			&vmpb.ImageGenerations{
				Image: []byte(text + "_3"),
				Score: float64(len(text)) * 0.3,
			},
			&vmpb.ImageGenerations{
				Image: []byte(text + "_4"),
				Score: float64(len(text)) * 0.4,
			},
		},
	}, nil
}

func (s *stubVisionModelServer) TextAndImageToImage(ctx context.Context, in *vmpb.TextAndImageToImageRequest) (*vmpb.TextAndImageToImageResponse, error) {
	text := in.GetText()
	image := string(in.GetImageBytes())
	return &vmpb.TextAndImageToImageResponse{
		Images: []*vmpb.ImageGenerations{
			&vmpb.ImageGenerations{
				Image: []byte(text + "_3" + image),
				Score: float64(len(text)) * 0.3,
			},
			&vmpb.ImageGenerations{
				Image: []byte(text + "_4" + image),
				Score: float64(len(text)) * 0.4,
			},
		},
	}, nil
}

func (s *stubVisionModelServer) ImageToImage(ctx context.Context, in *vmpb.ImageToImageRequest) (*vmpb.ImageToImageResponse, error) {
	image := string(in.GetImageBytes())
	return &vmpb.ImageToImageResponse{
		Images: []*vmpb.ImageGenerations{
			&vmpb.ImageGenerations{
				Image: []byte(image + "_3"),
				Score: float64(len(image)) * 0.3,
			},
			&vmpb.ImageGenerations{
				Image: []byte(image + "_4"),
				Score: float64(len(image)) * 0.4,
			},
		},
	}, nil
}

func (s *stubVisionModelServer) Embed(ctx context.Context, in *vmpb.EmbedRequest) (*vmpb.EmbedResponse, error) {
	value := float64(len(in.GetImageBytes()))
	return &vmpb.EmbedResponse{
		Embedding: []float64{
			value * 0.5,
			value * 0.6,
			value * 0.1,
			value * 0.2,
		},
	}, nil
}

func (s *stubVisionModelServer) Detect(ctx context.Context, in *vmpb.DetectRequest) (*vmpb.DetectResponse, error) {
	imageBytes := string(in.GetImageBytes())
	text := in.GetText()

	// Use the last entry of the text argument and add it to Text output, if it exists.
	var txt string
	if len(text) == 0 {
		txt = ""
	} else {
		txt = text[len(text)-1]
	}

	return &vmpb.DetectResponse{
		BoundingBoxes: []*vmpb.BoundingBox{
			&vmpb.BoundingBox{
				Cx:    0.3,
				Cy:    0.4,
				W:     0.5,
				H:     0.6,
				Text:  imageBytes + "_0" + txt,
				Score: float64(len(imageBytes)) * 0.1,
			},
		},
	}, nil
}

func (s *stubVisionModelServer) ImageToText(ctx context.Context, in *vmpb.ImageToTextRequest) (*vmpb.ImageToTextResponse, error) {
	text := string(in.GetImageBytes())
	prefix := string(in.GetText())
	return &vmpb.ImageToTextResponse{
		Texts: []*vmpb.DecodedText{
			&vmpb.DecodedText{
				Text:  text + prefix + "_2",
				Score: float64(len(text)) * 0.15,
			},
			&vmpb.DecodedText{
				Text:  text + prefix + "_3",
				Score: float64(len(text)) * 0.30,
			},
		},
	}, nil
}

func (s *stubVisionModelServer) VideoToText(ctx context.Context, in *vmpb.VideoToTextRequest) (*vmpb.VideoToTextResponse, error) {
	text := ""
	for _, frame := range in.GetImageFrames() {
		text = text + string(frame)
	}
	prefix := string(in.GetText())
	return &vmpb.VideoToTextResponse{
		Texts: []*vmpb.DecodedText{
			&vmpb.DecodedText{
				Text:  text + prefix + "_2",
				Score: float64(len(text)) * 0.15,
			},
			&vmpb.DecodedText{
				Text:  text + prefix + "_3",
				Score: float64(len(text)) * 0.30,
			},
		},
	}, nil
}

type stubAudioModelServer struct{}

func (s *stubAudioModelServer) Recognize(ctx context.Context, in *ampb.AsrRequest) (*ampb.AsrResponse, error) {
	text := string(in.GetAudioBytes())
	return &ampb.AsrResponse{
		Hyps: []*ampb.AsrHypothesis{
			&ampb.AsrHypothesis{
				Text:  text + "_0",
				Score: float64(len(text)) * 0.1,
			},
			&ampb.AsrHypothesis{
				Text:  text + "_1",
				Score: float64(len(text)) * 0.2,
			},
		},
	}, nil
}

type stubCustomModelServer struct{}

func (s *stubCustomModelServer) Custom(ctx context.Context, in *cmpb.CustomRequest) (*cmpb.CustomResponse, error) {
	text := in.GetRequest()
	request := &mmpb.GenerateRequest{}
	err := proto.Unmarshal(text, request)
	if err == nil {
		response := &mmpb.GenerateResponse{}
		response.Results = make([]*mmpb.GenerateResult, len(request.GetItems()))
		for i, item := range request.GetItems() {
			response.Results[i] = &mmpb.GenerateResult{
				Items: []*mmpb.DataItem{item},
				Score: float64(i) * 2,
			}
		}
		data, _ := proto.Marshal(response)
		return &cmpb.CustomResponse{Response: data}, nil
	}
	return &cmpb.CustomResponse{
		Response: append(text, []byte("_1")...),
	}, nil
}

type stubMultimodalModelServer struct{}

func (m *stubMultimodalModelServer) Generate(ctx context.Context, in *mmpb.GenerateRpcRequest) (*mmpb.GenerateRpcResponse, error) {
	res := &mmpb.GenerateRpcResponse{}
	res.Response = &mmpb.GenerateResponse{}
	res.Response.Results = make([]*mmpb.GenerateResult, len(in.GetRequest().GetItems()))
	for i, item := range in.GetRequest().GetItems() {
		res.Response.Results[i] = &mmpb.GenerateResult{
			Items: []*mmpb.DataItem{item},
			Score: float64(i) * 2,
		}
	}
	return res, nil
}

func (m *stubMultimodalModelServer) Score(ctx context.Context, in *mmpb.ScoreRpcRequest) (*mmpb.ScoreRpcResponse, error) {
	res := &mmpb.ScoreRpcResponse{}
	res.Response = &mmpb.ScoreResponse{}
	res.Response.Results = make([]*mmpb.ScoreResult, len(in.GetRequest().GetPrefixItems()))
	for i := range in.GetRequest().GetPrefixItems() {
		res.Response.Results[i] = &mmpb.ScoreResult{
			Score: float64(i) * 2,
		}
	}
	return res, nil
}

// StartStubModelServer starts a new model server of a given type with stub implementations, which
// also runs a modelet service.
// Close the returned channel to close the server.
func StartStubModelServer(modelType ModelType, modelPort int, scoreDelay time.Duration, unavailableModel string, loadDelay time.Duration) (chan struct{}, error) {
	ctx := context.Background()
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", modelPort))
	if err != nil {
		return nil, err
	}

	gRPCServer, err := env.Get().NewServer(ctx)
	if err != nil {
		return nil, err
	}

	switch modelType {
	case Language:
		lmgrpc.RegisterLMServiceServer(gRPCServer.GRPCServer(), &stubLanguageModelServer{
			scoreDelay:       scoreDelay,
			unavailableModel: unavailableModel,
		})
	case Vision:
		vmgrpc.RegisterVisionServiceServer(gRPCServer.GRPCServer(), &stubVisionModelServer{})
	case Audio:
		amgrpc.RegisterAudioServiceServer(gRPCServer.GRPCServer(), &stubAudioModelServer{})
	case Custom:
		cmgrpc.RegisterCustomServiceServer(gRPCServer.GRPCServer(), &stubCustomModelServer{})
	case Multimodal:
		mmgrpc.RegisterMultimodalServiceServer(gRPCServer.GRPCServer(), &stubMultimodalModelServer{})
	}

	modeletServer := &stubModeletServer{
		loadDelay:    loadDelay,
		loadedModels: make(map[string]bool),
	}
	mgrpc.RegisterModeletServer(gRPCServer.GRPCServer(), modeletServer)

	go gRPCServer.Serve(lis)
	closer := make(chan struct{})
	go func() {
		<-closer
		gRPCServer.Stop()
	}()

	return closer, nil
}

// StartStubModelServerT starts a new language model server with stub implementations, which
// also runs a modelet service.
// It is automatically closed when the test ends.
func StartStubModelServerT(t *testing.T, port int) {
	t.Helper()
	ch, err := StartStubModelServer(Language, port, 0, "", 0)
	if err != nil {
		t.Fatalf("StartStubModelServer failed: %v", err)
	}
	t.Cleanup(func() { close(ch) })
}

// CallModeletServer calls a modelet gRPC method on the model server at addr and returns the result.
func CallModeletServer(ctx context.Context, addr string, req any) (resp any, err error) {
	conn, err := env.Get().DialContext(ctx, addr)
	if err != nil {
		return nil, err
	}
	defer conn.Close()
	client := mgrpc.NewModeletClient(conn)

	switch req := req.(type) {
	case *mpb.LoadRequest:
		return client.Load(ctx, req)
	case *mpb.UnloadRequest:
		return client.Unload(ctx, req)
	case *mpb.GetStatusRequest:
		return client.GetStatus(ctx, req)
	default:
		return nil, fmt.Errorf("Unknown request type %T", req)
	}
}

// ModelType represents the type of models a model server supports.
type ModelType int

// ModelType values.
const (
	Language ModelType = iota
	Vision
	Audio
	Custom
	Multimodal
)

// Cluster is a test cluster with a stub admin server and one or many stub model server(s).
type Cluster struct {
	// Required.
	saxCell string

	// Optional.
	fsRoot            string          // default: automatically create a temp dir
	adminPort         int             // default: automatically pick one
	numberModelets    int             // default: 1
	modelType         ModelType       // default: language
	scoreDelays       []time.Duration // A list of delays for language servers.
	unavailableModels []string        // A list of unavailable models for language servers.
}

// SetAdminPort sets the admin server port of a test cluster.
func (c *Cluster) SetAdminPort(port int) *Cluster {
	c.adminPort = port
	return c
}

// SetFsRoot sets the file system root parameter of a test cluster.
func (c *Cluster) SetFsRoot(fsRoot string) *Cluster {
	c.fsRoot = fsRoot
	return c
}

// SetNumberModelets creates the desired number of modelets. Defaults to 1.
func (c *Cluster) SetNumberModelets(number int) *Cluster {
	c.numberModelets = number
	return c
}

// SetModelType sets the model type for the model servers.
func (c *Cluster) SetModelType(mt ModelType) *Cluster {
	c.modelType = mt
	return c
}

// SetScoreDelay sets the score delay parameter of a test cluster.
// Each delay applies to it corresponding modelet server.
func (c *Cluster) SetScoreDelay(delays []time.Duration) *Cluster {
	c.scoreDelays = delays
	return c
}

// SetUnavailableModels sets the unavailable model parameter of a test cluster.
// An unavailable model simulates a model being loaded or unloaded. Using it returns an error.
func (c *Cluster) SetUnavailableModels(unavailable []string) *Cluster {
	c.unavailableModels = unavailable
	return c
}

// StartInternal is exported for the C wrapper.
func (c *Cluster) StartInternal(ctx context.Context) (closers []chan struct{}, err error) {
	if err := SetUpInternal(ctx, c.saxCell, c.fsRoot); err != nil {
		return nil, err
	}

	var modelPorts []int
	for i := 0; i < c.numberModelets; i++ {
		modelPort, err := env.Get().PickUnusedPort()
		if err != nil {
			return nil, fmt.Errorf("start failed: pick model port error: %w", err)
		}
		var delay time.Duration
		if i < len(c.scoreDelays) {
			delay = c.scoreDelays[i]
		}
		var unavailable string
		if i < len(c.unavailableModels) {
			unavailable = c.unavailableModels[i]
		}
		closer, err := StartStubModelServer(c.modelType, modelPort, delay, unavailable, 0)
		if err != nil {
			return nil, fmt.Errorf("start failed: start model server error: %w", err)
		}
		closers = append(closers, closer)
		modelPorts = append(modelPorts, modelPort)
	}

	if c.adminPort == 0 {
		adminPort, err := env.Get().PickUnusedPort()
		if err != nil {
			return nil, fmt.Errorf("start failed: pick admin port error: %w", err)
		}
		c.adminPort = adminPort
	}
	closer, err := StartStubAdminServer(c.adminPort, modelPorts, c.saxCell)
	if err != nil {
		return nil, fmt.Errorf("start failed: start admin server error: %w", err)
	}
	return append(closers, closer), nil
}

// Start starts a test cluster with a stub admin server and a stub model server.
// They are automatically closed when the test ends.
func (c *Cluster) Start(ctx context.Context, t *testing.T) {
	t.Helper()
	if c.fsRoot == "" {
		c.fsRoot = t.TempDir()
	}
	closers, err := c.StartInternal(ctx)
	if err != nil {
		t.Fatalf("Start failed: %v", err)
	}
	t.Cleanup(func() {
		for _, c := range closers {
			close(c)
		}
	})
}

// NewCluster creates a test cluster with default parameter values.
func NewCluster(saxCell string) *Cluster {
	return &Cluster{saxCell: saxCell, numberModelets: 1}
}
