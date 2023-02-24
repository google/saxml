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

// Package admin implements a server that manages modelet servers.
package admin

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"path/filepath"
	"strconv"
	"sync"

	log "github.com/golang/glog"
	"google.golang.org/protobuf/encoding/prototext"
	"saxml/admin/mgr"
	"saxml/admin/validator"
	"saxml/common/addr"
	"saxml/common/config"
	"saxml/common/ipaddr"
	"saxml/common/naming"
	"saxml/common/platform/env"
	"saxml/common/state"

	pb "saxml/protobuf/admin_go_proto_grpc"
	pbgrpc "saxml/protobuf/admin_go_proto_grpc"
)

// Server implements an admin server.
type Server struct {
	// The SAX cell this server runs in.
	saxCell string

	// The port this server runs on.
	port int

	// serverID is the unique id for this server.
	serverID string

	// The gRPC server where this server is registered.
	gRPCServer env.Server

	// Close this channel to release the address lock.
	addrCloser chan<- struct{}

	// Mgr manages the internal state of the server.
	Mgr *mgr.Mgr

	// mu protects fields below.
	mu sync.Mutex

	// cfg is the server config, updated by a background watcher goroutine.
	cfg *pb.Config
}

func (s *Server) adminACL() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cfg.GetAdminAcl()
}

func (s *Server) Publish(ctx context.Context, in *pb.PublishRequest) (*pb.PublishResponse, error) {
	// Only cell admins can publish models in this cell.
	if err := s.gRPCServer.CheckACLs(ctx, []string{s.adminACL()}); err != nil {
		return nil, fmt.Errorf("permission error: %w", err)
	}

	model := in.GetModel()
	if err := validator.ValidateModelProto(model, s.saxCell); err != nil {
		return nil, err
	}

	if err := s.Mgr.Publish(model); err != nil {
		return nil, err
	}

	return &pb.PublishResponse{}, nil
}

func (s *Server) checkAdminACL(ctx context.Context, fullName naming.ModelFullName) error {
	var acls []string
	acl := s.adminACL()
	if acl != "" {
		acls = append(acls, acl)
	}
	pubModel, err := s.Mgr.List(fullName)
	if err != nil {
		return err
	}
	if acl := pubModel.GetModel().GetAdminAcl(); acl != "" {
		acls = append(acls, acl)
	}
	if err := s.gRPCServer.CheckACLs(ctx, acls); err != nil {
		return fmt.Errorf("permission error: %w", err)
	}
	return nil
}

func (s *Server) Update(ctx context.Context, in *pb.UpdateRequest) (*pb.UpdateResponse, error) {
	model := in.GetModel()
	if err := validator.ValidateModelProto(model, s.saxCell); err != nil {
		return nil, err
	}
	fullName, err := naming.NewModelFullName(model.GetModelId())
	if err != nil {
		return nil, err
	}

	// Either the cell admin or the model admin can update the model.
	if err := s.checkAdminACL(ctx, fullName); err != nil {
		return nil, err
	}
	if err := s.Mgr.Update(fullName, model); err != nil {
		return nil, err
	}

	return &pb.UpdateResponse{}, nil
}

func (s *Server) Unpublish(ctx context.Context, in *pb.UnpublishRequest) (*pb.UnpublishResponse, error) {
	modelFullName := in.GetModelId()
	if err := validator.ValidateModelFullName(modelFullName, s.saxCell); err != nil {
		return nil, err
	}
	fullName, err := naming.NewModelFullName(modelFullName)
	if err != nil {
		return nil, err
	}

	// Either the cell admin or the model admin can unpublish the model.
	if err := s.checkAdminACL(ctx, fullName); err != nil {
		return nil, err
	}
	if err := s.Mgr.Unpublish(fullName); err != nil {
		return nil, err
	}

	return &pb.UnpublishResponse{}, nil
}

func (s *Server) List(ctx context.Context, in *pb.ListRequest) (*pb.ListResponse, error) {
	modelFullName := in.GetModelId()

	// List one model specifically asked about.
	if modelFullName != "" {
		if err := validator.ValidateModelFullName(modelFullName, s.saxCell); err != nil {
			return nil, err
		}
		fullName, err := naming.NewModelFullName(modelFullName)
		if err != nil {
			return nil, err
		}

		pubModel, err := s.Mgr.List(fullName)
		if err != nil {
			return nil, err
		}

		return &pb.ListResponse{PublishedModels: []*pb.PublishedModel{pubModel}}, nil
	}

	// List all models if none is specifically asked about.
	return &pb.ListResponse{PublishedModels: s.Mgr.ListAll()}, nil
}

func (s *Server) locate(ctx context.Context, modelFullName string) ([]*pb.JoinedModelServer, error) {
	// Locate joined model servers.
	if modelFullName != "" {
		if err := validator.ValidateModelFullName(modelFullName, s.saxCell); err != nil {
			return nil, err
		}
		fullName, err := naming.NewModelFullName(modelFullName)
		if err != nil {
			return nil, err
		}

		pubModel, err := s.Mgr.List(fullName)
		if err != nil {
			return nil, err
		}
		addrs := pubModel.GetModeletAddresses()
		return s.Mgr.LocateSome(addrs)
	}
	return s.Mgr.LocateAll()
}

func (s *Server) Stats(ctx context.Context, in *pb.StatsRequest) (*pb.StatsResponse, error) {
	modelFullName := in.GetModelId()

	servers, err := s.locate(ctx, modelFullName)
	if err != nil {
		return nil, err
	}

	type ServerType struct {
		chipType pb.ModelServer_ChipType
		chipTopo pb.ModelServer_ChipTopology
	}
	replicasPerServerType := make(map[ServerType]int32)

	for _, server := range servers {
		serverType := ServerType{server.GetModelServer().GetChipType(), server.GetModelServer().GetChipTopology()}
		replicasPerServerType[serverType] = replicasPerServerType[serverType] + 1
	}

	modelServerTypeStats := []*pb.ModelServerTypeStat{}
	for serverType, replicas := range replicasPerServerType {
		modelServerTypeStats = append(modelServerTypeStats, &pb.ModelServerTypeStat{
			ChipType:     serverType.chipType,
			ChipTopology: serverType.chipTopo,
			NumReplicas:  replicas,
		})
	}
	return &pb.StatsResponse{ModelServerTypeStats: modelServerTypeStats}, nil
}

// WatchLoc handles WatchLoc rpc requests.
func (s *Server) WatchLoc(ctx context.Context, in *pb.WatchLocRequest) (*pb.WatchLocResponse, error) {
	if err := validator.ValidateWatchLocRequest(in); err != nil {
		return nil, err
	}
	seqno := in.GetSeqno()
	if in.GetAdminServerId() != s.serverID {
		seqno = 0
	}
	result, err := s.Mgr.WatchLoc(ctx, in.GetModelId(), seqno)
	if err != nil {
		return nil, err
	}
	return &pb.WatchLocResponse{
		AdminServerId: s.serverID,
		Result:        result.ToProto(),
	}, nil
}

func (s *Server) Join(ctx context.Context, in *pb.JoinRequest) (*pb.JoinResponse, error) {
	if err := validator.ValidateJoinRequest(in); err != nil {
		return nil, err
	}

	if err := s.Mgr.Join(ctx, in.GetAddress(), in.GetDebugAddress(), in.GetModelServer()); err != nil {
		return nil, err
	}

	return &pb.JoinResponse{}, nil
}

// Start starts running the server.
func (s *Server) Start(ctx context.Context) error {
	if _, err := naming.SaxCellToCell(s.saxCell); err != nil {
		return err
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		return fmt.Errorf("net.Listen on port %v error: %w", s.port, err)
	}
	gRPCServer, err := env.Get().NewServer()
	if err != nil {
		return fmt.Errorf("NewServer error: %w", err)
	}

	s.cfg, err = config.Load(ctx, s.saxCell)
	log.Infof("Loaded config: %v", prototext.Format(s.cfg))
	if err != nil {
		return fmt.Errorf("config.Load error: %w", err)
	}
	fsRoot := s.cfg.GetFsRoot()
	if fsRoot == "" {
		return fmt.Errorf("no fs_root specified")
	}
	fsPath := filepath.Join(fsRoot, s.saxCell)
	if err := env.Get().CreateDir(ctx, fsPath, ""); err != nil {
		return fmt.Errorf("CreateDir from %v error: %w", fsPath, err)
	}

	go func() {
		ch, err := config.Watch(ctx, s.saxCell)
		if err != nil {
			log.Errorf("config.Watch error, no more updates: %v", err)
			return
		}
		for cfg := range ch {
			log.Infof("Updated config: %v", prototext.Format(cfg))
			s.mu.Lock()
			s.cfg = cfg
			s.mu.Unlock()
		}
	}()

	s.Mgr = mgr.New(state.New(fsPath))

	s.gRPCServer = gRPCServer
	pbgrpc.RegisterAdminServer(gRPCServer.GRPCServer(), s)

	// Set the admin address for this cell. Block until done.
	s.addrCloser, err = addr.SetAddr(ctx, s.port, s.saxCell)
	if err != nil {
		return fmt.Errorf("addr.SetAddr error: %w", err)
	}

	// Start the manager.
	if err := s.Mgr.Start(ctx); err != nil {
		return fmt.Errorf("s.Mgr.Start error: %w", err)
	}

	// This goroutine exits when s.Close is called.
	go func() {
		log.Infof("Starting the server on port %v", s.port)
		if err := gRPCServer.Serve(lis); err != nil {
			log.Errorf("Stopped the server due to error: %v", err)
			return
		}
		log.Infof("Stopped the server")
	}()

	return nil
}

// Close closes a running server.
func (s *Server) Close() {
	s.Mgr.Close()
	if s.gRPCServer != nil {
		s.gRPCServer.Stop()
	}
	if s.addrCloser != nil {
		close(s.addrCloser)
	}
}

// NewServer creates an admin server for `saxCell`. `store` is the backing store for server states.
func NewServer(saxCell string, port int) *Server {
	return &Server{
		saxCell:  saxCell,
		port:     port,
		serverID: fmt.Sprintf("%s_%016x", net.JoinHostPort(ipaddr.MyIPAddr().String(), strconv.Itoa(port)), rand.Uint64()),
	}
}
