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

// Package mgr provides an admin server state manager that uses a GetStatus-based protocol.
package mgr

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	log "github.com/golang/glog"
	"google.golang.org/protobuf/proto"
	"github.com/pborman/uuid"
	"saxml/admin/protobuf"
	"saxml/admin/state"
	"saxml/admin/validator"
	"saxml/common/errors"
	"saxml/common/eventlog"
	"saxml/common/naming"
	"saxml/common/platform/env"
	"saxml/common/waitable"
	"saxml/common/watchable"

	apb "saxml/protobuf/admin_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
	mpb "saxml/protobuf/modelet_go_proto_grpc"
)

var (
	// The interval between consecutive Refresh calls, which reassign models to model servers.
	refreshPeriod = time.Second * 10

	// Model servers that haven't sent any valid GetStatus response for this long will get pruned.
	// This should be longer than refreshPeriod in the state package.
	pruneTimeout = time.Second * 25
)

// SetOptionsForTesting updates refreshPeriod and pruneTimeout for tests.
//
// refresh is usually set to one hour to disable automatic refresh in favor of manual refresh.
// prune can be one hour to disable model server pruning, or 0 to immediately prune all servers.
func SetOptionsForTesting(refresh, prune time.Duration) {
	refreshPeriod = refresh
	pruneTimeout = prune
}

// modelFullName identifies a model in the form of /sax/<cell>/<model>.
type modelFullName = naming.ModelFullName

// modeletAddr locates a model server in the form of <ip>:<port>.
type modeletAddr string

// modelState tracks a model's server assignment.
type modelState struct {
	// specs is the proto definition of the model.
	specs *apb.Model

	// addrWatcher keeps track of the set of model server addresses serving the model.
	addrWatcher *watchable.Watchable

	// waiter keeps track of requests waiting for certain numbers of replicas to be ready.
	waiter *waitable.Waitable
}

// modeletState synchronizes state with the model server.
type modeletState = state.State

// Store represents the backing store of the admin server state.
type Store interface {
	Read(ctx context.Context) (*apb.State, error)
	Write(ctx context.Context, state *apb.State) error
}

// Mgr manages the admin server state and remote model server states.
type Mgr struct {
	// Guards the maps below.
	mu sync.RWMutex
	// Published models. The address watcher in each model keeps a list of model servers loading or
	// having loaded the model. Due to load/unload delays and failures, this list can be a subset of
	// modelets[*].WantedModels(), and is used by the client through the WatchLoc method to find a
	// ready server.
	models map[modelFullName]*modelState
	// Joined model servers. The WantedModels method on each model server returns the list of models
	// currently assigned to it. This is used as ground truth to compute new assignment.
	modelets map[modeletAddr]*modeletState
	// The reverse map of modelets[*].WantedModels() computed at reassignment time. When model servers
	// join and leave, this map is not kept in sync. Therefore, it should only be used by List* and
	// Locate* methods for human usage or status page display.
	assignment map[modelFullName][]modeletAddr
	// Recently unpublished model full names. They still have pending load/unload ops.
	// Models cannot be published under any name inside until it's removed from this set.
	pendingUnpublished map[modelFullName]bool

	// The backing store of this admin server's state.
	store Store

	// Ticker for calling refresh periodically.
	ticker     *time.Ticker
	tickerStop chan bool

	eventLogger eventlog.Logger
}

// Publish publishes a model.
func (m *Mgr) Publish(specs *apb.Model) error {
	fullName, err := naming.NewModelFullName(specs.GetModelId())
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.models[fullName]; ok {
		return fmt.Errorf("model %s already exists: %w", fullName, errors.ErrAlreadyExists)
	}
	if _, ok := m.pendingUnpublished[fullName]; ok {
		return fmt.Errorf("model %s is being unpublished, please retry later: %w", fullName, errors.ErrAlreadyExists)
	}
	log.Infof("Published with overrides: %v", specs.GetOverrides())

	specsWithUUID := proto.Clone(specs).(*apb.Model)
	specsWithUUID.Uuid = uuid.NewRandom()
	m.models[fullName] = &modelState{
		specs:       specsWithUUID,
		addrWatcher: watchable.New(),
		waiter:      waitable.New(),
	}

	m.eventLogger.Log(eventlog.Deploy, specsWithUUID)

	return nil
}

// Update updates a model.
func (m *Mgr) Update(fullName modelFullName, newSpecs *apb.Model) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	existing, ok := m.models[fullName]
	if !ok {
		return fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}
	if err := validator.ValidateModelUpdate(existing.specs, newSpecs, fullName.CellFullName()); err != nil {
		return fmt.Errorf("invalid model update: %w", err)
	}

	// Copy UUID from existing model.
	specsWithUUID := proto.Clone(newSpecs).(*apb.Model)
	specsWithUUID.Uuid = existing.specs.Uuid
	existing.specs = specsWithUUID

	addrs, ok := m.assignment[fullName]
	if !ok {
		// No server serving this model.
		return nil
	}
	for _, addr := range addrs {
		state, ok := m.modelets[modeletAddr(addr)]
		if ok {
			state.Update(fullName, newSpecs)
		} else {
			log.Errorf("Failed to find the server for addr %v", addr)
		}
	}
	return nil
}

// Unpublish unpublishes a model.
func (m *Mgr) Unpublish(fullName modelFullName) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	model, ok := m.models[fullName]
	if !ok {
		return fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}
	m.pendingUnpublished[fullName] = true
	delete(m.models, fullName)
	model.addrWatcher.Close()
	model.waiter.Close()
	return nil
}

func (m *Mgr) makePublishedModelLocked(fullName modelFullName, model *apb.Model) *apb.PublishedModel {
	addrs := []string{}
	for _, addr := range m.assignment[fullName] {
		addrs = append(addrs, string(addr))
	}
	cloned := proto.Clone(model).(*apb.Model)
	// Clean Uuid field to not expose it to users.
	cloned.Uuid = nil
	return &apb.PublishedModel{
		Model:            cloned,
		ModeletAddresses: addrs,
	}
}

// List returns information about one published model.
func (m *Mgr) List(fullName modelFullName) (*apb.PublishedModel, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	model, ok := m.models[fullName]
	if !ok {
		return nil, fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}
	return m.makePublishedModelLocked(fullName, model.specs), nil
}

// ListSome returns information about a few published models.
func (m *Mgr) ListSome(fullNames []modelFullName) ([]*apb.PublishedModel, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	publishedModels := []*apb.PublishedModel{}
	for _, fullName := range fullNames {
		model, ok := m.models[fullName]
		if !ok {
			return nil, fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
		}
		publishedModels = append(publishedModels, m.makePublishedModelLocked(fullName, model.specs))
	}
	return publishedModels, nil
}

// ListAll returns information about all published models.
func (m *Mgr) ListAll() []*apb.PublishedModel {
	m.mu.RLock()
	defer m.mu.RUnlock()

	publishedModels := []*apb.PublishedModel{}
	for fullName, model := range m.models {
		publishedModels = append(publishedModels, m.makePublishedModelLocked(fullName, model.specs))
	}
	return publishedModels
}

// WatchLoc watches changes of the model server addresses after the given seqno.
func (m *Mgr) WatchLoc(ctx context.Context, fullName string, seqno int32) (*watchable.WatchResult, error) {
	modelFullName, err := naming.NewModelFullName(fullName)
	if err != nil {
		return nil, err
	}

	m.mu.RLock()
	model, ok := m.models[modelFullName]
	if !ok {
		m.mu.RUnlock()
		return nil, fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}
	m.mu.RUnlock()

	return model.addrWatcher.Watch(ctx, seqno)
}

// WaitForReady returns when the number of loaded replicas reaches the given threshold.
func (m *Mgr) WaitForReady(ctx context.Context, fullName modelFullName, numReplicas int) error {
	m.mu.RLock()
	model, ok := m.models[fullName]
	if !ok {
		m.mu.RUnlock()
		return fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}
	m.mu.RUnlock()

	return model.waiter.Wait(ctx, numReplicas)
}

// FindModel returns model information given a known model full name, nil otherwise.
func (m *Mgr) FindModel(fullName modelFullName) *apb.Model {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if model, ok := m.models[fullName]; ok {
		return model.specs
	}
	return nil
}

// Join lets one model server join from an address.
func (m *Mgr) Join(ctx context.Context, addr, debugAddr, dataAddr string, specs *apb.ModelServer) error {
	maddr := modeletAddr(addr)

	createNewServerState := func() error {
		modelServer := state.New(addr, debugAddr, dataAddr, protobuf.NewModelServer(specs), m.eventLogger)
		if err := modelServer.Start(ctx, m); err != nil {
			return fmt.Errorf("failed to start a connection with %v: %w", addr, err)
		}

		m.mu.Lock()
		_, ok := m.modelets[maddr]
		if !ok {
			m.modelets[maddr] = modelServer

			// Only models loading or loaded should get added to the address watcher.
			for fullName := range modelServer.WantedModels() {
				model := m.models[fullName]
				model.addrWatcher.Add(modelServer.DataAddr)
			}
			// Be conservative and only add loaded models to the waiter. Models that are loading, when a
			// model server joins an admin server, are not counted toward a pending WaitForReady request.
			for fullName, seenModel := range modelServer.SeenModels() {
				model, ok := m.models[fullName]
				if !ok {
					continue
				}
				if seenModel.Info.Status == protobuf.Loaded {
					model.waiter.Add(1)
				}
			}
		}
		m.mu.Unlock()

		if ok {
			// Another Join succeeded before us.
			modelServer.Close()
		}
		return nil
	}

	// Let the server join, heartbeat, or replace an existing one at the same address if any.
	//
	// Do all the m.modelets mutation work under the lock, leaving the time-consuming RPC-related
	// work to after the unlock.
	m.mu.Lock()
	existing, ok := m.modelets[maddr]
	var same bool // only valid when ok
	if !ok {
		log.V(4).Infof("Modelet %s, %v has joined", addr, specs)
	} else {
		same = existing.Specs.Equal(specs)
		if same {
			log.V(4).Infof("Modelet %s, %v is healthy", addr, existing.Specs)
		} else {
			log.V(4).Infof("Modelet %s, %v has replaced %v", addr, specs, existing.Specs)
			delete(m.modelets, maddr)
		}
	}
	m.mu.Unlock()

	if !ok {
		return createNewServerState()
	}
	if same {
		return nil
	}
	existing.Close() // already deleted from m.modelets
	return createNewServerState()
}

// GetStatus returns information about one joined model server.
func (m *Mgr) GetStatus(ctx context.Context, addr string, full bool) (*mpb.GetStatusResponse, error) {
	m.mu.RLock()
	modelet, ok := m.modelets[modeletAddr(addr)]
	m.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("model server %v not found: %w", addr, errors.ErrNotFound)
	}
	return modelet.GetStatus(ctx, full)
}

func (m *Mgr) makeJoinedModelServerLocked(addr string, modelet *modeletState) (*apb.JoinedModelServer, error) {
	statuses := map[string]cpb.ModelStatus{}
	var successesPerSecond, errorsPerSecond, meanLatencyInSeconds float32 = 0., 0., 0.
	for fullName, status := range modelet.SeenModels() {
		s, err := status.Info.Status.ToProto()
		if err != nil {
			return nil, err
		}
		statuses[fullName.ModelFullName()] = s
		for _, stats := range status.Info.Stats {
			successesPerSecond += stats.SuccessesPerSecond
			errorsPerSecond += stats.ErrorsPerSecond
			meanLatencyInSeconds += stats.MeanLatencyInSeconds
		}
	}
	return &apb.JoinedModelServer{
		ModelServer:          modelet.Specs.ToProto(),
		Address:              addr,
		DebugAddress:         modelet.DebugAddr,
		DataAddress:          modelet.DataAddr,
		LastJoinMs:           modelet.LastPing().UnixMilli(),
		LoadedModels:         statuses,
		SuccessesPerSecond:   successesPerSecond,
		ErrorsPerSecond:      errorsPerSecond,
		MeanLatencyInSeconds: meanLatencyInSeconds,
	}, nil
}

// Locate returns information about one joined model server.
func (m *Mgr) Locate(addr string) (*apb.JoinedModelServer, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	modelet, ok := m.modelets[modeletAddr(addr)]
	if !ok {
		return nil, fmt.Errorf("model server %v not found: %w", addr, errors.ErrNotFound)
	}
	return m.makeJoinedModelServerLocked(addr, modelet)
}

// LocateSome returns information about a few joined model servers.
func (m *Mgr) LocateSome(addrs []string) ([]*apb.JoinedModelServer, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	joinedModelServers := []*apb.JoinedModelServer{}
	for _, addr := range addrs {
		modelet, ok := m.modelets[modeletAddr(addr)]
		if !ok {
			return nil, fmt.Errorf("model server %v not found: %w", addr, errors.ErrNotFound)
		}
		joinedModelServer, err := m.makeJoinedModelServerLocked(addr, modelet)
		if err != nil {
			return nil, err
		}
		joinedModelServers = append(joinedModelServers, joinedModelServer)
	}
	return joinedModelServers, nil
}

// LocateAll returns information about all joined model servers.
func (m *Mgr) LocateAll() ([]*apb.JoinedModelServer, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	joinedModelServers := []*apb.JoinedModelServer{}
	for addr, modelet := range m.modelets {
		joinedModelServer, err := m.makeJoinedModelServerLocked(string(addr), modelet)
		if err != nil {
			return nil, err
		}
		joinedModelServers = append(joinedModelServers, joinedModelServer)
	}
	return joinedModelServers, nil
}

// pruneModelets removes model servers that haven't called Join in the last `timeout` duration.
func (m *Mgr) pruneModelets(timeout time.Duration) {
	cutoff := time.Now().Add(-timeout) // modelets not seen after the cutoff are removed

	m.mu.Lock()
	defer m.mu.Unlock()

	for addr, modelet := range m.modelets {
		lastPing := modelet.LastPing()
		if lastPing.After(cutoff) {
			continue
		}
		// We are only responsible for handling WantedModels here. After that, we can delete addr from
		// modelets. If any model is scheduled to be unloaded and therefore not in WantedModels,
		// the corresponding unloadModels will eventually remove addr from those models' addrWatcher.
		for fullName := range modelet.WantedModels() {
			model, ok := m.models[fullName]
			if ok {
				model.addrWatcher.Del(modelet.DataAddr)
			}
		}
		// Subtract models that are loaded from the waiter. Also be conservative and subtract loading
		// models, regardless of the loading result later.
		for fullName, seenModel := range modelet.SeenModels() {
			model, ok := m.models[fullName]
			if !ok {
				continue
			}
			if seenModel.Info.Status == protobuf.Loaded {
				model.waiter.Add(-1)
			}
		}
		delete(m.modelets, addr)
		log.V(2).Infof("Pruned modelet %v with last ping at %v before cutoff %v", addr, lastPing, cutoff)
		go modelet.Close() // Close() may block for a while.
	}
}

// RefreshResult contains the result of a Server.Refresh call.
type RefreshResult struct {
	// The total number of model servers requested by all models when Refresh is called
	TotalRequested int

	// The number of model servers already assigned when Refresh is called.
	AlreadyAssigned int

	// Models unpublished but not yet handled when Refresh is called.
	pendingUnpublished map[modelFullName]bool

	// The new assignment computed by the call to Refresh.
	NewAssignment map[modelFullName][]modeletAddr

	// Model servers unassigned by the call to Refresh.
	NewlyUnassigned map[modeletAddr]modelFullName

	// Model servers assigned by the call to Refresh.
	NewlyAssigned map[modeletAddr]modelFullName

	// Temporarily store data address for later usage by unloadModels
	DataAddress map[modeletAddr]string
}

// ComputeAssignment computes new model-to-server assignment.
func (m *Mgr) ComputeAssignment() RefreshResult {
	log.V(1).Infof("Assigning model servers to models")
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Current assignment.
	var totalRequested, alreadyAssigned int
	pendingUnpublished := map[modelFullName]bool{}
	currentAssignment := map[modelFullName][]modeletAddr{}

	// Take a snapshot of models recently unpublished. Their unload ops, if needed, should be
	// generated below and placed in newlyUnassigned. Remove them from m.pendingUnpublished after
	// these unload ops are successfully issued.
	for fullName := range m.pendingUnpublished {
		pendingUnpublished[fullName] = true
	}

	// Iterates through model servers in sorted address order.
	// This way, newAssignment[*] are also sorted and stable.
	var addrs []string
	for addr := range m.modelets {
		addrs = append(addrs, string(addr))
	}
	sort.Strings(addrs)
	log.V(1).Infof("All model server addresses: %v", addrs)

	// Extract current assignment and gather busy model servers into a set.
	newlyUnassigned := map[modeletAddr]modelFullName{}
	busy := map[modeletAddr]bool{}
	for _, addr := range addrs {
		maddr := modeletAddr(addr)
		for fullName := range m.modelets[maddr].WantedModels() {
			if _, ok := m.models[fullName]; ok {
				// The model is still published.
				currentAssignment[fullName] = append(currentAssignment[fullName], maddr)
				busy[maddr] = true
			} else {
				// The model has been unpublished.
				newlyUnassigned[maddr] = fullName
			}
		}
	}

	// Sort the assignments so that loaded models come first in the list.
	// Since we unload starting at the end of the list, this encourages us to keep loaded models
	// and unload models from tasks that are loading/failed/unloading.
	modelIsLoaded := func(modelName naming.ModelFullName, addr modeletAddr) bool {
		for seenModelName, modelWithStatus := range m.modelets[addr].SeenModels() {
			if seenModelName == modelName {
				return modelWithStatus.Info.Status == protobuf.Loaded
			}
		}
		return false
	}
	log.V(1).Infof("Current assignment: %v", currentAssignment)
	for modelName, addresses := range currentAssignment {
		sort.Slice(addresses, func(i int, j int) bool {
			iLoaded := modelIsLoaded(modelName, addresses[i])
			jLoaded := modelIsLoaded(modelName, addresses[j])

			// i < j iff i is loaded and j is not.
			if iLoaded && !jLoaded {
				return true
			}

			return false
		})
	}
	log.V(1).Infof("Current assignment sorted: %v", currentAssignment)

	// Find and index idle model servers by servable model paths.
	idle := map[string]map[modeletAddr]bool{} // model path -> set of model servers able to serve it
	var pathAddr []string
	for addr, state := range m.modelets {
		if busy[addr] {
			continue
		}
		for _, path := range state.Specs.ServableModelPaths {
			if _, ok := idle[path]; !ok {
				idle[path] = map[modeletAddr]bool{}
			}
			idle[path][addr] = true
			pathAddr = append(pathAddr, fmt.Sprintf("%s %s", path, addr))
		}
	}
	sort.Strings(pathAddr)
	log.V(1).Infof("Available servers (<path> <address>):")
	for _, pa := range pathAddr {
		log.V(1).Infof("%s", pa)
	}

	// For each model, greedily assign as many available model servers as possible.
	newAssignment := map[modelFullName][]modeletAddr{}
	newlyAssigned := map[modeletAddr]modelFullName{}
	for fullName, model := range m.models {
		assigned := currentAssignment[fullName]
		requested := int(model.specs.GetRequestedNumReplicas())

		log.V(1).Infof("Model %s (%s) requests %v modelets", fullName, model.specs.GetModelPath(), model.specs.GetRequestedNumReplicas())
		totalRequested += int(model.specs.GetRequestedNumReplicas())

		log.V(1).Infof("Model %s has %v model servers already assigned", fullName, len(assigned))
		alreadyAssigned += len(assigned)

		// Unassign one replica at a time if fewer are needed.
		for len(assigned) > requested {
			last := len(assigned) - 1
			newlyUnassigned[assigned[last]] = fullName
			assigned = assigned[:last]
		}

		// Keep using items from the idle map until either fulfilled or out of items.
		taken := []modeletAddr{}
		for addr := range idle[model.specs.GetModelPath()] {
			if len(assigned) >= requested {
				break
			}
			taken = append(taken, addr)
			maddr := modeletAddr(addr)
			assigned = append(assigned, maddr)
			newlyAssigned[maddr] = fullName
		}

		// Update the idle map.
		for _, addr := range taken {
			for _, path := range m.modelets[addr].Specs.ServableModelPaths {
				delete(idle[path], addr)
			}
		}

		log.V(1).Infof("Model %s is assigned %v new model servers", fullName, len(newlyAssigned))
		newAssignment[fullName] = assigned
	}
	log.V(1).Infof("New assignment: %v", newAssignment)

	// In unloadModels, it needs dataAddr for newly unassigned models
	// Construct a temporary dictionary mapping from addr to dataAddr
	dataAddress := map[modeletAddr]string{}
	for maddr := range newlyUnassigned {
		dataAddress[maddr] = m.modelets[maddr].DataAddr
	}

	return RefreshResult{totalRequested, alreadyAssigned, pendingUnpublished, newAssignment, newlyUnassigned, newlyAssigned, dataAddress}
}

func (m *Mgr) installAssignment(assignment map[modelFullName][]modeletAddr) {
	log.V(1).Infof("Install new assignment %v", assignment)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.assignment = assignment
}

// loadModels loads models onto newly assigned modelets in parallel.
func (m *Mgr) loadModels(ctx context.Context, newlyAssigned map[modeletAddr]modelFullName) {
	load := func(ctx context.Context, fullName modelFullName, addr modeletAddr) error {
		m.mu.Lock()
		modelet, ok := m.modelets[addr]
		if !ok {
			m.mu.Unlock()
			return fmt.Errorf("model server %v has left", addr)
		}
		model, ok := m.models[fullName]
		if !ok {
			m.mu.Unlock()
			return fmt.Errorf("model %v has been unpublished", fullName)
		}
		m.mu.Unlock()

		if err := modelet.Load(ctx, fullName, model.specs, model.waiter); err != nil {
			return err
		}

		// We want to be conservative and add addr to addrWatch only if the server will definitely load
		// the model.
		m.mu.Lock()
		defer m.mu.Unlock()
		model, ok = m.models[fullName]
		if !ok {
			return fmt.Errorf("model %v has been unpublished", fullName)
		}
		model.addrWatcher.Add(modelet.DataAddr)
		return nil
	}

	for addr, fullName := range newlyAssigned {
		log.V(2).Infof("Loading model %v onto model server %v", fullName, addr)
		go func(fullName modelFullName, addr modeletAddr) {
			if err := load(ctx, fullName, modeletAddr(addr)); err != nil {
				log.Errorf("Failed to load model %v onto model server %v: %v", fullName, addr, err)
			} else {
				log.V(2).Infof("Loaded model %v onto model server %v", fullName, addr)
			}
		}(fullName, addr)
	}
}

// unloadModels unloads models from newly unassigned modelets in parallel.
// addrWatcher needs dataAddr to remove unloaded models, and `dataAddress“ is dictionary from
// addr to dataAddr.
func (m *Mgr) unloadModels(ctx context.Context, newlyUnassigned map[modeletAddr]modelFullName, dataAddress map[modeletAddr]string) {
	unload := func(ctx context.Context, fullName modelFullName, addr modeletAddr, dataAddr string) error {
		m.mu.Lock()
		model, ok := m.models[fullName]
		var waiter *waitable.Waitable
		if ok {
			// Regardless of whether Unload below will succeed or fail, we want to be conservative and
			// remove addr from addrWatcher now. In particular, if pruneModelets is called after
			// ComputeAssignment but before unloadModels, modelets may not have addr anymore, but
			// we still need to remove addr from the address watcher. See the matching comment in
			// pruneModelets for reference.
			model.addrWatcher.Del(dataAddr)
			waiter = model.waiter
		}
		modelet, ok := m.modelets[addr]
		if !ok {
			m.mu.Unlock()
			return fmt.Errorf("model server %v has left", addr)
		}
		m.mu.Unlock()

		// If waiter is nil, it means the model is already unpublished and its waiter closed.
		// Unload will do nothing to waiter.
		return modelet.Unload(ctx, fullName, waiter)
	}

	for addr, fullName := range newlyUnassigned {
		log.V(2).Infof("Unloading model %v from model server %v", fullName, addr)
		go func(fullName modelFullName, addr modeletAddr, dataAddr string) {
			if err := unload(ctx, fullName, modeletAddr(addr), dataAddr); err != nil {
				log.Errorf("Failed to unload model %v from model server %v: %v", fullName, addr, err)
			} else {
				log.V(2).Infof("Unloaded model %v from model server %v", fullName, addr)
			}
		}(fullName, addr, dataAddress[addr])
	}
}

// freeUnpublishedNames removes recently unpublished names from the pending list, so those names
// are available for publishing again.
func (m *Mgr) freeUnpublishedNames(unpublished map[modelFullName]bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for fullName := range unpublished {
		delete(m.pendingUnpublished, fullName)
	}
}

// Refresh updates manager state by reassigning model servers to models and running tasks to carry
// out the state change, such as prune dead model servers and load/unload models.
func (m *Mgr) Refresh(ctx context.Context) {
	// Remove dead model servers.
	m.pruneModelets(pruneTimeout)
	// Compute new assignment.
	result := m.ComputeAssignment()
	// Install the new assignment.
	m.installAssignment(result.NewAssignment)
	// Unload and load models according to assignment results.
	m.unloadModels(ctx, result.NewlyUnassigned, result.DataAddress)
	m.loadModels(ctx, result.NewlyAssigned)
	// Now that unload calls have been issued, make model full names unpublished before the
	// ComputeAssignment call available for use again.
	m.freeUnpublishedNames(result.pendingUnpublished)
}

// Restore restores the manager state from its backing store.
func (m *Mgr) Restore(ctx context.Context) error {
	if m.store == nil {
		return fmt.Errorf("no backing store specified: %w", errors.ErrFailedPrecondition)
	}
	state, err := m.store.Read(ctx)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	for _, model := range state.GetModels() {
		fullName, err := naming.NewModelFullName(model.GetModelId())
		if err != nil {
			return err
		}
		m.models[fullName] = &modelState{
			specs:       model,
			addrWatcher: watchable.New(),
			waiter:      waitable.New(),
		}
	}
	return nil
}

// Save saves the manager state to its backing store.
func (m *Mgr) Save(ctx context.Context) error {
	if m.store == nil {
		return fmt.Errorf("no backing store specified: %w", errors.ErrFailedPrecondition)
	}

	state := &apb.State{}
	m.mu.RLock()
	for _, model := range m.models {
		state.Models = append(state.Models, proto.Clone(model.specs).(*apb.Model))
	}
	m.mu.RUnlock()
	return m.store.Write(ctx, state)
}

// Start starts running the manager.
func (m *Mgr) Start(ctx context.Context) error {
	if err := m.Restore(ctx); err != nil {
		return err
	}
	log.Infof("Loaded manager state")

	// Start a goroutine that calls refresh periodically, stopping when m.Close is called.
	log.Infof("Refreshing manager state every %v", refreshPeriod)
	m.ticker = time.NewTicker(refreshPeriod)
	m.tickerStop = make(chan bool)
	go func() {
		for {
			select {
			case <-m.tickerStop:
				close(m.tickerStop)
				return
			case t := <-m.ticker.C:
				m.Refresh(context.TODO())
				log.V(3).Infof("Refreshed manager state at %v", t)

				if err := m.Save(context.TODO()); err != nil {
					log.Fatalf("Failed to save manager state: %v", err)
				} else {
					log.V(3).Infof("Saved manager state at %v", t)
				}
			}
		}
	}()

	return nil
}

// Close closes a running manager.
func (m *Mgr) Close() {
	// Don't close the channel here, to prevent the goroutine from seeing an empty action.
	m.ticker.Stop()
	m.tickerStop <- true
	<-m.tickerStop
	m.eventLogger.Close()
}

// New creates an empty manager with a backing store.
func New(store Store) *Mgr {
	return &Mgr{
		models:             make(map[modelFullName]*modelState),
		modelets:           make(map[modeletAddr]*modeletState),
		assignment:         make(map[modelFullName][]modeletAddr),
		pendingUnpublished: make(map[modelFullName]bool),
		store:              store,
		eventLogger:        env.Get().NewEventLogger(),
	}
}
