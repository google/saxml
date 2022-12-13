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
	"math/rand"
	"sort"
	"sync"
	"time"

	log "github.com/golang/glog"
	"google.golang.org/protobuf/proto"
	"saxml/admin/protobuf"
	"saxml/admin/state"
	"saxml/admin/validator"
	"saxml/common/errors"
	"saxml/common/naming"
	"saxml/common/watchable"

	apb "saxml/protobuf/admin_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
	mpb "saxml/protobuf/modelet_go_proto_grpc"
)

var (
	// Model servers that haven't sent any valid GetStatus response for this long will get deleted.
	// This should be longer than refreshPeriod in the state package.
	pingTimeout = time.Second * 25

	// The interval between consecutive refresh calls, including model-to-model-server reassignment.
	refreshPeriod = time.Second * 10
)

// SetOptionsForTesting updates refreshPeriod and pingTimeout for tests.
//
// refresh is usually set to one hour to disable automatic refresh in favor of manual refresh.
// timeout can be one hour to disable model server pruning, or 0 to immediately prune all servers.
func SetOptionsForTesting(refresh, timeout time.Duration) {
	refreshPeriod = refresh
	pingTimeout = timeout
}

// modelAddr locates a model server in the form of <ip>:<port>.
type modelAddr string

// modelAssignment represents one model-to-model-server assignment and is used on the corresponding
// Load call.
type modelAssignment modelAddr

// modelAssignmentList maintains a list of modelAssignment.
type modelAssignmentList struct {
	list []modelAssignment
}

func (l *modelAssignmentList) Len() int {
	return len(l.list)
}

func (l *modelAssignmentList) Append(item modelAssignment) {
	l.list = append(l.list, item)
}

func (l *modelAssignmentList) PopLast() modelAssignment {
	last := l.Len() - 1
	item := l.list[last]
	l.list = l.list[:last]
	return item
}

func (l modelAssignmentList) All() []modelAssignment {
	return l.list
}

// assignmentMap is a map type mapping model full names to a list of server assignments.
type assignmentMap map[naming.ModelFullName]*modelAssignmentList

func (a assignmentMap) Get(id naming.ModelFullName) *modelAssignmentList {
	if list, ok := a[id]; ok {
		return list
	}
	return &modelAssignmentList{}
}

func (a assignmentMap) GetOrAdd(id naming.ModelFullName) *modelAssignmentList {
	list, ok := a[id]
	if !ok {
		list = &modelAssignmentList{}
		a[id] = list
	}
	return list
}

func (a assignmentMap) Add(id naming.ModelFullName, ma modelAssignment) {
	list, ok := a[id]
	if !ok {
		list = &modelAssignmentList{}
		a[id] = list
	}
	list.Append(ma)
}

// Store represents the backing store of the server state.
type Store interface {
	Read(ctx context.Context) (*apb.State, error)
	Write(ctx context.Context, state *apb.State) error
}

type modelState struct {
	// spec is the proto definition of the model.
	spec *apb.Model

	// addrWatcher keeps track of the set of model server addresses serving the model.
	addrWatcher *watchable.Watchable
}

// Mgr manages the admin server state and remote model server states.
type Mgr struct {
	// Guards the maps below.
	mu sync.RWMutex
	// Published models.
	models map[naming.ModelFullName]*modelState
	// Joined model servers.
	modelets map[modelAddr]*state.State
	// Model servers are periodically assigned to model IDs.
	assignment assignmentMap
	// Recently unpublished model full names. They still have pending load/unload ops.
	// Models cannot be published under any name inside until it's removed from this set.
	pendingUnpublished map[naming.ModelFullName]struct{}

	// The backing store of this server.
	store Store

	// Ticker for calling refresh periodically.
	ticker     *time.Ticker
	tickerStop chan bool
}

// Publish publishes a model.
func (m *Mgr) Publish(model *apb.Model) error {
	fullName, err := naming.NewModelFullName(model.GetModelId())
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
	m.models[fullName] = &modelState{
		spec:        proto.Clone(model).(*apb.Model),
		addrWatcher: watchable.New(),
	}

	return nil
}

// Update updates a model's number of replicas.
func (m *Mgr) Update(fullName naming.ModelFullName, newSpec *apb.Model) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	existing, ok := m.models[fullName]
	if !ok {
		return fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}
	if err := validator.ValidateModelUpdate(existing.spec, newSpec, fullName.CellFullName()); err != nil {
		return fmt.Errorf("invalid model update: %w", err)
	}
	existing.spec = proto.Clone(newSpec).(*apb.Model)
	// TODO(zhifengc): Adds a method state.Update(newSpec) so that
	// acl changes can propagate to all model servers.
	return nil
}

// Unpublish unpublishes a model.
func (m *Mgr) Unpublish(fullName naming.ModelFullName) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, ok := m.models[fullName]; !ok {
		return fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}
	m.pendingUnpublished[fullName] = struct{}{}
	// TODO(zhifengc): Explicitly Close m.models[mid].addrWatcher.
	delete(m.models, fullName)
	return nil
}

func (m *Mgr) makePublishedModelLocked(fullName naming.ModelFullName, model *apb.Model) *apb.PublishedModel {
	addrs := []string{}
	for _, assignment := range m.assignment.Get(fullName).All() {
		addrs = append(addrs, string(assignment))
	}
	return &apb.PublishedModel{
		Model:            proto.Clone(model).(*apb.Model),
		ModeletAddresses: addrs,
	}
}

// List returns information about one published model.
func (m *Mgr) List(fullName naming.ModelFullName) (*apb.PublishedModel, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	model, ok := m.models[fullName]
	if !ok {
		return nil, fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}
	return m.makePublishedModelLocked(fullName, model.spec), nil
}

// ListSome returns information about a few published models.
func (m *Mgr) ListSome(ids []string) ([]*apb.PublishedModel, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	pubModels := []*apb.PublishedModel{}
	for _, id := range ids {
		mid, err := naming.NewModelFullName(id)
		if err != nil {
			return nil, err
		}
		model, ok := m.models[mid]
		if !ok {
			return nil, fmt.Errorf("model %s not found: %w", mid, errors.ErrNotFound)
		}
		pubModels = append(pubModels, m.makePublishedModelLocked(mid, model.spec))
	}
	return pubModels, nil
}

// ListAll returns information about all published models.
func (m *Mgr) ListAll() []*apb.PublishedModel {
	m.mu.RLock()
	defer m.mu.RUnlock()

	pubModels := []*apb.PublishedModel{}
	for mid, model := range m.models {
		pubModels = append(pubModels, m.makePublishedModelLocked(mid, model.spec))
	}
	return pubModels
}

// FindLoc returns modelet server addresses up to a requested number for a model ID.
func (m *Mgr) FindLoc(fullName naming.ModelFullName, upTo int) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if _, ok := m.models[fullName]; !ok {
		return nil, fmt.Errorf("model %s not found: %w", fullName, errors.ErrNotFound)
	}

	res := []string{}
	assignments := m.assignment.Get(fullName).All()
	have := len(assignments)
	if have == 0 {
		// Note: zero return is an allowed response.
		return res, nil
	}

	// Note: at this point, both upTo and have are positive.
	if upTo > have {
		upTo = have
	}

	// TODO(jianlijianli, jiawenhao): improve this with rejection sampling.
	idx := rand.Intn(have)
	for i := 0; i < upTo; i++ {
		res = append(res, string(assignments[idx]))
		idx = idx + 1
		// Wrap around. This won't cause duplication since upTo <= have.
		if idx == have {
			idx = 0
		}
	}
	return res, nil
}

// WatchLoc watches changes of the model server addresses after the given seqno.
func (m *Mgr) WatchLoc(ctx context.Context, id string, seqno int32) (*watchable.WatchResult, error) {
	mid, err := naming.NewModelFullName(id)
	if err != nil {
		return nil, err
	}
	m.mu.RLock()
	model, ok := m.models[mid]
	if !ok {
		m.mu.RUnlock()
		return nil, fmt.Errorf("model ID %v not found: %w", id, errors.ErrNotFound)
	}
	m.mu.RUnlock()

	return model.addrWatcher.Watch(ctx, seqno)
}

// FindModel returns model information given a known model full name, nil otherwise.
func (m *Mgr) FindModel(fullName naming.ModelFullName) *apb.Model {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.models[fullName].spec
}

// Join lets one modelet server join from an address.
func (m *Mgr) Join(ctx context.Context, addr string, modelet *apb.ModelServer) error {
	maddr := modelAddr(addr)

	createNewServerState := func() error {
		modelServer := state.New(addr, protobuf.NewModelServer(modelet))
		if err := modelServer.Start(ctx, m); err != nil {
			return fmt.Errorf("Failed to start a connection with %v: %w", addr, err)
		}

		m.mu.Lock()
		_, ok := m.modelets[maddr]
		if !ok {
			m.modelets[maddr] = modelServer

			for id := range modelServer.SeenModels() {
				model := m.models[id]
				model.addrWatcher.Add(addr)
			}
		}
		m.mu.Unlock()

		if ok {
			// A later Join succeeded before us.
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
		log.V(4).Infof("Modelet %s, %v has joined", addr, modelet)
	} else {
		same = existing.Specs.Equal(modelet)
		if same {
			log.V(4).Infof("Modelet %s, %v is healthy", addr, existing.Specs)
		} else {
			log.V(4).Infof("Modelet %s, %v has replaced %v", addr, modelet, existing.Specs)
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
	existing.Close()
	return createNewServerState()
}

func (m *Mgr) reverseAssignmentLocked() map[modelAddr][]string {
	addrToIDs := map[modelAddr][]string{}
	for id, assignments := range m.assignment {
		for _, assignment := range assignments.All() {
			addr := modelAddr(assignment)
			addrToIDs[addr] = append(addrToIDs[addr], id.ModelFullName())
		}
	}
	return addrToIDs
}

func (m *Mgr) makeJoinedModeletLocked(maddr modelAddr, modelet *state.State, addrToIDs map[modelAddr][]string) (*apb.JoinedModelServer, error) {
	statuses := map[string]cpb.ModelStatus{}
	for fullName, status := range modelet.SeenModels() {
		s, err := status.Status.ToProto()
		if err != nil {
			return nil, err
		}
		statuses[fullName.ModelFullName()] = s
	}
	return &apb.JoinedModelServer{
		ModelServer:  modelet.Specs.ToProto(),
		Address:      string(maddr),
		LastJoinMs:   modelet.LastPing().UnixMilli(),
		LoadedModels: statuses,
	}, nil
}

// GetStatus returns information about one joined modelet.
func (m *Mgr) GetStatus(ctx context.Context, addr string, full bool) (*mpb.GetStatusResponse, error) {
	m.mu.RLock()
	maddr := modelAddr(addr)
	modelet, ok := m.modelets[maddr]
	m.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("modelet %v not found: %w", maddr, errors.ErrNotFound)
	}
	return modelet.GetStatus(ctx, full)
}

// Locate returns information about one joined modelet.
func (m *Mgr) Locate(addr string) (*apb.JoinedModelServer, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	maddr := modelAddr(addr)
	modelet, ok := m.modelets[maddr]
	if !ok {
		return nil, fmt.Errorf("modelet %v not found: %w", maddr, errors.ErrNotFound)
	}
	ids := []string{}
	for id, assignments := range m.assignment {
		for _, assignment := range assignments.All() {
			if modelAddr(assignment) == maddr {
				ids = append(ids, id.ModelFullName())
			}
		}
	}
	addrToIDs := map[modelAddr][]string{maddr: ids}
	return m.makeJoinedModeletLocked(maddr, modelet, addrToIDs)
}

// LocateSome returns information about a few joined modelets.
func (m *Mgr) LocateSome(addrs []string) ([]*apb.JoinedModelServer, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	addrToIDs := m.reverseAssignmentLocked()
	joinedModelets := []*apb.JoinedModelServer{}
	for _, addr := range addrs {
		maddr := modelAddr(addr)
		info, ok := m.modelets[maddr]
		if !ok {
			return nil, fmt.Errorf("modelet %v not found: %w", maddr, errors.ErrNotFound)
		}
		joined, err := m.makeJoinedModeletLocked(maddr, info, addrToIDs)
		if err != nil {
			return nil, err
		}
		joinedModelets = append(joinedModelets, joined)
	}
	return joinedModelets, nil
}

// LocateAll returns information about all joined modelets.
func (m *Mgr) LocateAll() ([]*apb.JoinedModelServer, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	addrToIDs := m.reverseAssignmentLocked()
	joinedModelets := []*apb.JoinedModelServer{}
	for maddr, info := range m.modelets {
		joined, err := m.makeJoinedModeletLocked(maddr, info, addrToIDs)
		if err != nil {
			return nil, err
		}
		joinedModelets = append(joinedModelets, joined)
	}
	return joinedModelets, nil
}

// pruneModelets removes modelet servers that haven't called Join in the last `timeout` duration.
func (m *Mgr) pruneModelets(timeout time.Duration) {
	cutoff := time.Now().Add(-timeout) // modelets not seen after the cutoff are removed

	m.mu.Lock()
	for addr, modelServer := range m.modelets {
		lastPing := modelServer.LastPing()
		if lastPing.After(cutoff) {
			continue
		}
		for id := range modelServer.SeenModels() {
			model := m.models[id]
			model.addrWatcher.Del(string(addr))
		}
		delete(m.modelets, addr)
		log.V(2).Infof("Pruned modelet %v with last ping at %v before cutoff %v",
			addr, lastPing, cutoff)
		go modelServer.Close() // Close() may block for a while.
	}
	m.mu.Unlock()
}

// RefreshResult contains the result of a Server.Refresh call.
type RefreshResult struct {
	// The toal number of model servers requested by all models.
	NumRequested int

	// Before the call to Refresh, the number of model servers already assigned.
	NumAssigned int

	// Models unpublished before the call to Refresh.
	Unpublished map[naming.ModelFullName]bool

	// The new assignment, i.e., model -> list of addresses.
	NewAssignment assignmentMap

	// Model servers unassigned by the call to Refresh.
	NewlyUnassigned map[modelAssignment]naming.ModelFullName

	// Model servers assigned by the call to Refresh.
	NewlyAssigned map[modelAssignment]naming.ModelFullName
}

// ComputeAssignment computes new model-to-server assignment.
func (m *Mgr) ComputeAssignment() RefreshResult {
	log.V(1).Infof("Assigning model servers to models")
	m.mu.RLock()
	defer m.mu.RUnlock()

	var numRequested, numAssigned int
	unpublished := map[naming.ModelFullName]bool{}
	newAssignment := assignmentMap{}
	newlyUnassigned := map[modelAssignment]naming.ModelFullName{}
	newlyAssigned := map[modelAssignment]naming.ModelFullName{}

	// Take a snapshot of models recently unpublished. Their unload ops, if needed, should be
	// generated below and placed in newlyUnassigned. Remove them from m.unpublished after these
	// unload ops are issued.
	for fullName := range m.pendingUnpublished {
		unpublished[fullName] = true
	}

	// Iterates through model servers in sorted addr order.
	// This way, newAssigment[*] are also sorted and stable.
	var addrs []string
	for addr := range m.modelets {
		addrs = append(addrs, string(addr))
	}
	sort.Strings(addrs)
	log.V(1).Infof("All model server addresses: %v", addrs)

	// Extract current assignment and gather busy model servers into a set.
	busy := map[modelAddr]bool{}
	for _, addr := range addrs {
		maddr := modelAddr(addr)
		state := m.modelets[maddr]
		for fullName := range state.WantedModels() {
			ma := modelAssignment(maddr)
			if _, ok := m.models[fullName]; ok {
				// The model is still published.
				newAssignment.Add(fullName, ma)
				busy[maddr] = true
			} else {
				// The model has been unpublished.
				newlyUnassigned[ma] = fullName
			}
		}
	}
	log.V(1).Infof("Current assignment: %v", newAssignment)

	// Find and index idle model servers by servable model paths.
	idle := map[string]map[modelAddr]bool{} // model path -> set of modelet servers able to serve it
	var pathAddr []string
	for addr, state := range m.modelets {
		if busy[addr] {
			continue
		}
		for _, path := range state.Specs.ServableModelPaths {
			if _, ok := idle[path]; !ok {
				idle[path] = map[modelAddr]bool{}
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
	for id, model := range m.models {
		maList := newAssignment.GetOrAdd(id)
		requested := int(model.spec.GetRequestedNumReplicas())

		log.V(1).Infof("Model %s (%s) requests %v modelets", id, model.spec.GetModelPath(), model.spec.GetRequestedNumReplicas())
		numRequested += int(model.spec.GetRequestedNumReplicas())

		log.V(1).Infof("Model %s has %v model servers already assigned", id, maList.Len())
		numAssigned += maList.Len()

		// Unassign one replica at a time if fewer are needed.
		for maList.Len() > requested {
			newlyUnassigned[maList.PopLast()] = id
		}

		// Keep using items from the idle map until either fulfilled or out of items.
		taken := []modelAddr{}
		for addr := range idle[model.spec.GetModelPath()] {
			if maList.Len() >= requested {
				break
			}
			taken = append(taken, addr)
			ma := modelAssignment(addr)
			maList.Append(ma)
			newlyAssigned[ma] = id
		}

		// Update the idle map.
		for _, addr := range taken {
			for _, path := range m.modelets[addr].Specs.ServableModelPaths {
				delete(idle[path], addr)
			}
		}

		log.V(1).Infof("Model %s is assigned %v new model servers", id, len(newlyAssigned))
	}

	log.V(1).Infof("New assignment: %v", newAssignment)
	return RefreshResult{numRequested, numAssigned, unpublished, newAssignment, newlyUnassigned, newlyAssigned}
}

func (m *Mgr) installAssignment(assignment assignmentMap) {
	log.V(3).Infof("Install new assignment %v", assignment)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.assignment = assignment
}

// loadModels loads models onto newly assigned modelets in parallel.
func (m *Mgr) loadModels(ctx context.Context, newlyAssigned map[modelAssignment]naming.ModelFullName) {
	load := func(ctx context.Context, id naming.ModelFullName, addr modelAddr) error {
		m.mu.Lock()
		modelet, ok := m.modelets[addr]
		if !ok {
			m.mu.Unlock()
			return fmt.Errorf("model server %v has left", addr)
		}
		model, ok := m.models[id]
		if !ok {
			return fmt.Errorf("model %v has been unpublished", id)
		}
		m.mu.Unlock()

		if err := modelet.Load(ctx, id, model.spec); err != nil {
			return err
		}

		m.mu.Lock()
		defer m.mu.Unlock()
		model, ok = m.models[id]
		if !ok {
			return fmt.Errorf("model %v has been unpublished", id)
		}
		model.addrWatcher.Add(string(addr))
		return nil
	}

	for addr, id := range newlyAssigned {
		log.V(2).Infof("Loading model %v onto model server %v", id, addr)
		go func(id naming.ModelFullName, addr modelAssignment) {
			if err := load(ctx, id, modelAddr(addr)); err != nil {
				log.Errorf("Failed to load model %v onto model server %v: %v", id, addr, err)
			} else {
				log.V(2).Infof("Loaded model %v onto model server %v", id, addr)
			}
		}(id, addr)
	}
}

// unloadModels unloads models from newly unassigned modelets in parallel.
func (m *Mgr) unloadModels(ctx context.Context, newlyUnassigned map[modelAssignment]naming.ModelFullName) {
	unload := func(ctx context.Context, id naming.ModelFullName, addr modelAddr) error {
		m.mu.Lock()
		modelet := m.modelets[addr]
		if modelet == nil {
			m.mu.Unlock()
			return fmt.Errorf("model server %v has left", addr)
		}
		model, ok := m.models[id]
		if ok {
			model.addrWatcher.Del(string(addr))
		}
		m.mu.Unlock()

		return modelet.Unload(ctx, id)
	}

	for addr, id := range newlyUnassigned {
		log.V(2).Infof("Unloading model %v from model server %v", id, addr)
		go func(id naming.ModelFullName, addr modelAssignment) {
			if err := unload(ctx, id, modelAddr(addr)); err != nil {
				log.Errorf("Failed to unload model %v from model server %v: %v", id, addr, err)
			} else {
				log.V(2).Infof("Unloaded model %v from model server %v", id, addr)
			}
		}(id, addr)
	}
}

// freeUnpublishedNames removes recently unpublished names from the pending list, so those names
// are available for publishing again.
func (m *Mgr) freeUnpublishedNames(unpublished map[naming.ModelFullName]bool) {
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
	m.pruneModelets(pingTimeout)
	// Compute new model-to-model-server assignment.
	result := m.ComputeAssignment()
	// Install the new assignment.
	m.installAssignment(result.NewAssignment)
	// Unload and load models according to assignment results.
	m.unloadModels(ctx, result.NewlyUnassigned)
	m.loadModels(ctx, result.NewlyAssigned)
	// Make model full names unpublished before the computeAssignment call available for use again.
	m.freeUnpublishedNames(result.Unpublished)
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
			spec:        model,
			addrWatcher: watchable.New(),
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
		state.Models = append(state.Models, proto.Clone(model.spec).(*apb.Model))
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
					log.Errorf("Failed to save manager state: %v", err)
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
}

// New creates an empty manager with a backing store.
func New(store Store) *Mgr {
	return &Mgr{
		models:             make(map[naming.ModelFullName]*modelState),
		modelets:           make(map[modelAddr]*state.State),
		assignment:         make(assignmentMap),
		pendingUnpublished: make(map[naming.ModelFullName]struct{}),
		store:              store,
	}
}
