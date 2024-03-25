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

// Package assigner provides Assigner to facilitate assigning models to servers.
package assigner

import (
	"sort"
	"strings"

	"saxml/admin/protobuf"
	"saxml/admin/utils"
	"saxml/common/naming"
	apb "saxml/protobuf/admin_go_proto_grpc"
)

// ServerAddr represents a model server address. E.g., 1.2.3.4:14001.
type ServerAddr string

// ParamPath is an identifier for a model configuration. E.g.,
// params.foo.bar.MyModel.
type ParamPath string

// ServerInfo contains metadata about a model server.
type ServerInfo struct {
	memoryCapacity    int64
	servableModelPath []ParamPath
	tags              map[string]bool
	loadedModel       map[naming.ModelFullName]protobuf.ModelStatus
}

// NewServerInfo constructs a ServerInfo based on the given model
// server specification.
func NewServerInfo(serverSpec *protobuf.ModelServer) *ServerInfo {
	s := &ServerInfo{
		memoryCapacity:    utils.GetServerMemoryCapacity(serverSpec),
		servableModelPath: []ParamPath{},
		tags:              make(map[string]bool),
		loadedModel:       make(map[naming.ModelFullName]protobuf.ModelStatus),
	}
	for _, path := range serverSpec.ServableModelPaths {
		s.servableModelPath = append(s.servableModelPath, ParamPath(path))
	}
	for _, tag := range serverSpec.Tags {
		s.tags[tag] = true
	}
	return s
}

// AddLoadedModel indicates the server has been assigned a model with
// fullName and is currently with the status.
func (s *ServerInfo) AddLoadedModel(fullName naming.ModelFullName, status protobuf.ModelStatus) {
	s.loadedModel[naming.ModelFullName(fullName)] = status
}

// ModelInfo contains metadata about a model.
type ModelInfo struct {
	modelPath      ParamPath
	neededReplicas int
	memoryRequired int64
	constraints    []string
}

// NewModelInfo constructs a ModelInfo given a model definition.
func NewModelInfo(spec *apb.Model) *ModelInfo {
	return &ModelInfo{
		modelPath:      ParamPath(spec.GetModelPath()),
		neededReplicas: int(spec.GetRequestedNumReplicas()),
		memoryRequired: utils.GetMemoryRequired(spec),
		constraints:    utils.GetConstraints(spec),
	}
}

// Action represents an intention, either load or unload a model onto or from a server.
type Action struct {
	Addr  ServerAddr
	Model naming.ModelFullName
}

// Assigner comes up a plan to unload and load models on servers.
//
// E.g.,
//
//	a := assigner.New()
//	// Tells the assigner about servers.
//	a.AddServer("foo", &assigner.ServerInfo{...})
//	a.AddServer("bar", &assigner.ServerInfo{...})
//	// Tells the assigner about models.
//	a.AddModel("/sax/test/modelA", specA)
//	a.AddModel("/sax/test/modelB", specB)
//
//	a.Assign()
//	for _, unload := a.Unloads() {
//	   // take action to do unloads.
//	}
//	 for _, load := a.Loads() {
//	   // take action to do unloads.
//	 }
type Assigner struct {
	// The known state.
	servers map[ServerAddr]*ServerInfo
	models  map[naming.ModelFullName]*ModelInfo

	// Derived state.
	params   map[ParamPath][]ServerAddr
	assigned map[naming.ModelFullName][]ServerAddr

	// Actions to take.
	toLoad   []Action
	toUnload []Action
}

// New constructs an Assigner object.
func New() *Assigner {
	a := &Assigner{
		servers:  make(map[ServerAddr]*ServerInfo),
		models:   make(map[naming.ModelFullName]*ModelInfo),
		params:   make(map[ParamPath][]ServerAddr),
		assigned: make(map[naming.ModelFullName][]ServerAddr),
	}
	return a
}

// AddServer tells the assigner about a model server with its address
// and server information.
func (a *Assigner) AddServer(addr ServerAddr, server *ServerInfo) {
	a.servers[addr] = server
	for _, path := range server.servableModelPath {
		a.params[path] = append(a.params[path], addr)
	}
}

// AddModel tells the assigner about a model with its full name and
// model information.
func (a *Assigner) AddModel(name naming.ModelFullName, minfo *ModelInfo) {
	a.models[name] = minfo
	a.assigned[name] = []ServerAddr{}
}

// Assign computes an assignment of models to servers based on the
// given server and model information.
func (a *Assigner) Assign() {
	// Compute which servers have been assigned to serve a model, for
	// every model.
	for addr, server := range a.servers {
		for loaded := range server.loadedModel {
			a.assigned[loaded] = append(a.assigned[loaded], addr)
			if _, ok := a.models[loaded]; !ok {
				// The model has been unpublished.
				a.toUnload = append(a.toUnload, Action{addr, loaded})
			}
		}
	}

	// Sort the assignments so that loaded models come first in the
	// list.  Since we unload starting at the end of the list, this
	// encourages us to keep loaded models and unload models from
	// tasks that are loading/failed/unloading.
	for loaded, addrs := range a.assigned {
		sort.Slice(addrs, func(i int, j int) bool {
			iLoaded := a.servers[addrs[i]].loadedModel[loaded] == protobuf.Loaded
			jLoaded := a.servers[addrs[j]].loadedModel[loaded] == protobuf.Loaded
			return iLoaded && !jLoaded
		})
	}

	// For each model, we compute its memory required.
	reqMem := make(map[naming.ModelFullName]int64)
	var modelNames []naming.ModelFullName
	for name, model := range a.models {
		modelNames = append(modelNames, name)
		var required int64 = -1
		if model.memoryRequired > 0 {
			// The model metadata specified the model's memory requirement.
			required = model.memoryRequired
		} else if addrs, ok := a.params[model.modelPath]; ok {
			// Let us assume the model needs to occupy the server with
			// the most memory.
			for _, addr := range addrs {
				server := a.servers[addr]
				if server.memoryCapacity > required {
					required = server.memoryCapacity
				}
			}
		}
		if required >= 0 {
			reqMem[name] = required
		}
	}

	// Computes how much memory available in each server.
	availMem := make(map[ServerAddr]int64)
	for addr, server := range a.servers {
		avail := server.memoryCapacity
		for loaded := range server.loadedModel {
			if used, ok := reqMem[loaded]; ok {
				avail -= used
			} else {
				// If the model does not exist in reqMem, the model is
				// about to be unloaded.
			}
		}
		availMem[addr] = avail
	}

	// The main assignment loop
	sort.Slice(modelNames, func(i, j int) bool {
		// Now, we prefer assign "bigger" models first.
		cmp := reqMem[modelNames[i]] - reqMem[modelNames[j]]
		if cmp != 0 {
			return cmp > 0
		}
		return strings.Compare(modelNames[i].ModelName(), modelNames[j].ModelName()) < 0
	})
	for _, name := range modelNames {
		model := a.models[name]
		addrs := a.assigned[name]

		// Unload one replica from the end of the server address list
		// because LOADING replicas are ordered at the end.
		for model.neededReplicas < len(addrs) {
			last := len(addrs) - 1
			addr := addrs[last]
			a.toUnload = append(a.toUnload, Action{addr, name})
			addrs = addrs[:last]
		}

		if model.neededReplicas == len(addrs) {
			continue
		}

		// Try to find servers which have this much available memory.
		required := reqMem[name]

		// Keeps severs which supports the model in their available
		// memory's non-increasing order.
		type serverMemItem struct {
			addr     ServerAddr
			availMem int64
		}
		candidates := []*serverMemItem{}
		for _, addr := range a.params[model.modelPath] {
			avail := availMem[addr]
			if required > avail {
				continue
			}
			server := a.servers[addr]
			// All constraints need to be met by tags of the server.
			constrainsMet := true
			for _, tag := range model.constraints {
				if _, ok := server.tags[tag]; !ok {
					constrainsMet = false
					break
				}
			}
			if constrainsMet {
				candidates = append(candidates, &serverMemItem{
					addr:     addr,
					availMem: avail,
				})
			}
		}
		sort.Slice(candidates, func(i int, j int) bool {
			return candidates[i].availMem > candidates[j].availMem
		})

		n := model.neededReplicas - len(addrs)
		if n < len(candidates) {
			candidates = candidates[:n]
		}
		for _, item := range candidates {
			a.toLoad = append(a.toLoad, Action{item.addr, name})
			availMem[item.addr] -= required
		}
	}

	// Compute the new assignment.
	//
	// newAssigned is a map-of-map so that we can compute the unique
	// set of server addrs for a model. The consumer of GetAssignment,
	// mgr.go, only needs a slice of server addrs. Maybe we can
	// simplify these types later.
	newAssigned := make(map[naming.ModelFullName]map[ServerAddr]bool)
	getOrCreate := func(name naming.ModelFullName) map[ServerAddr]bool {
		ret, ok := newAssigned[name]
		if !ok {
			ret = map[ServerAddr]bool{}
			newAssigned[name] = ret
		}
		return ret
	}
	for name, addrs := range a.assigned {
		servers := getOrCreate(name)
		for _, addr := range addrs {
			servers[addr] = true
		}
	}
	for _, act := range a.toUnload {
		delete(getOrCreate(act.Model), act.Addr)
	}
	for _, act := range a.toLoad {
		getOrCreate(act.Model)[act.Addr] = true
	}
	a.assigned = make(map[naming.ModelFullName][]ServerAddr)
	for name, addrs := range newAssigned {
		for addr := range addrs {
			a.assigned[name] = append(a.assigned[name], addr)
		}
	}
}

// GetToLoad returns a list of actions for loading models onto servers.
func (a *Assigner) GetToLoad() []Action { return a.toLoad }

// GetToUnload returns a list of actions for unloading models from servers.
func (a *Assigner) GetToUnload() []Action { return a.toUnload }

// GetAssignment returns a model assignment (i.e., model -> server
// lists) after the computed load and unload actions are executed.
func (a *Assigner) GetAssignment() map[naming.ModelFullName][]ServerAddr {
	return a.assigned
}
