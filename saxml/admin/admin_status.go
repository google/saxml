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

package admin

import (
	"context"
	"fmt"
	"net/http"
	"time"

	log "github.com/golang/glog"
	"saxml/common/naming"
	"saxml/common/platform/env"

	apb "saxml/protobuf/admin_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
)

const (
	// Status page path prefixes relative to the server root.
	modelHandlerPattern  = "/model/"
	serverHandlerPattern = "/server/"

	getStatusTimeout = time.Second * 2
)

func (s *Server) createModelInfos(models []*apb.PublishedModel, servers []*apb.JoinedModelServer) []*env.ModelInfo {
	addrs := make(map[string]bool)
	for _, server := range servers {
		addrs[server.Address] = true
	}
	modelStats := s.Mgr.GetStatsPerModel(addrs)
	var ret []*env.ModelInfo
	for _, m := range models {
		var rate float32
		if name, err := naming.NewModelFullName(m.GetModel().GetModelId()); err == nil {
			rate = modelStats[name]
		}
		ret = append(ret, &env.ModelInfo{Model: m, SuccessesPerSecond: rate})
	}
	return ret
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	models := s.Mgr.ListAll()
	servers, err := s.Mgr.LocateAll()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to locate all servers: %v", err), http.StatusInternalServerError)
		return
	}

	data := &env.StatusPageData{
		Kind:    env.RootStatusPage,
		SaxCell: s.saxCell,
		Models:  s.createModelInfos(models, servers),
		Servers: servers}
	if err := s.gRPCServer.WriteStatusPage(w, data); err != nil {
		http.Error(w, fmt.Sprintf("Page generation failed: %v", err), http.StatusInternalServerError)
		return
	}
}

func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
	modelName := r.URL.Path[len(modelHandlerPattern):]
	modelFullName, err := naming.NewModelFullName("/" + modelName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Invalid model name %q: %v", modelName, err), http.StatusInternalServerError)
		return
	}
	model, err := s.Mgr.List(modelFullName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to list model %q: %v", modelFullName.ModelFullName(), err), http.StatusInternalServerError)
		return
	}
	addrs := model.GetModeletAddresses()
	servers, err := s.Mgr.LocateSome(addrs)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to locate servers %v: %v", addrs, err), http.StatusInternalServerError)
	}

	data := &env.StatusPageData{
		Kind:    env.ModelStatusPage,
		SaxCell: s.saxCell,
		Models:  s.createModelInfos([]*apb.PublishedModel{model}, servers),
		Servers: servers}
	if err := s.gRPCServer.WriteStatusPage(w, data); err != nil {
		http.Error(w, fmt.Sprintf("Page generation failed: %v", err), http.StatusInternalServerError)
		return
	}
}

func (s *Server) handleServer(w http.ResponseWriter, r *http.Request) {
	addr := r.URL.Path[len(serverHandlerPattern):]
	server, err := s.Mgr.Locate(addr)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to locate server %v: %v", addr, err), http.StatusInternalServerError)
		return
	}

	// Try to add failure reasons into `server` for failed models.
	server.FailureReasons = make(map[string]string)
	ctx, cancel := context.WithTimeout(r.Context(), getStatusTimeout)
	defer cancel()
	res, err := s.Mgr.GetStatus(ctx, server.GetAddress(), true)
	if err != nil {
		log.Errorf("Failed to get status from server %v: %v", server.GetAddress(), err)
	} else {
		for _, model := range res.GetModels() {
			if model.GetModelStatus() == cpb.ModelStatus_FAILED {
				server.FailureReasons[model.GetModelKey()] = model.GetFailureReason()
			}
		}
	}

	modelFullNames := []naming.ModelFullName{}
	for fullName := range server.GetLoadedModels() {
		modelFullName, err := naming.NewModelFullName(fullName)
		if err != nil {
			http.Error(w, fmt.Sprintf("Invalid model name %q: %v", fullName, err), http.StatusInternalServerError)
			return
		}
		modelFullNames = append(modelFullNames, modelFullName)
	}
	models, err := s.Mgr.ListSome(modelFullNames)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to list models %v: %v", modelFullNames, err), http.StatusInternalServerError)
		return
	}

	servers := []*apb.JoinedModelServer{server}
	data := &env.StatusPageData{
		Kind:    env.ServerStatusPage,
		SaxCell: s.saxCell,
		Models:  s.createModelInfos(models, servers),
		Servers: servers}
	if err := s.gRPCServer.WriteStatusPage(w, data); err != nil {
		http.Error(w, fmt.Sprintf("Page generation failed: %v", err), http.StatusInternalServerError)
		return
	}
}

// EnableStatusPages registers a few http handlers for this server.
//
// The HTTP handlers are registered in http.DefaultServeMux, which is hard-coded to be used by the
// built-in HTTP server in grpcprod.Server. DefaultServeMux doesn't get reset between tests,
// doesn't allow handlers to be unregistered, and will panic when the same pattern is registered
// the second time. Do not call this function in tests.
func (s *Server) EnableStatusPages() {
	http.HandleFunc("/", s.handleRoot)
	http.HandleFunc(modelHandlerPattern, s.handleModel)
	http.HandleFunc(serverHandlerPattern, s.handleServer)
}
