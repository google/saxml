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

// Package state provides a model server state manager.
package state

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"time"

	log "github.com/golang/glog"
	"google.golang.org/grpc"
	"saxml/admin/protobuf"
	"saxml/common/errors"
	"saxml/common/eventlog"
	"saxml/common/naming"
	"saxml/common/platform/env"
	"saxml/common/waitable"

	apb "saxml/protobuf/admin_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
	mpb "saxml/protobuf/modelet_go_proto_grpc"
	mgrpc "saxml/protobuf/modelet_go_proto_grpc"
)

var (
	// The interval between consecutive Refresh calls, which call GetStatus on the model server.
	// This should be shorter than pruneTimeout in the mgr package.
	refreshPeriod = time.Second * 10

	// Various RPC timeout thresholds.
	dialTimeout      = time.Second * 10
	getStatusTimeout = time.Second * 10
)

// SetOptionsForTesting updates refreshPeriod so that during tests, the state issues more frequent
// GetStatus calls.
func SetOptionsForTesting(refresh time.Duration) {
	refreshPeriod = refresh
}

// ModelFinder returns model information given a model full name.
type ModelFinder interface {
	FindModel(naming.ModelFullName) *apb.Model
}

// Model contains the static state of a model.
type Model struct {
	Path       string
	Checkpoint string
	Acls       map[string]string
	Overrides  map[string]string
	UUID       []byte
}

func cloneAcls(src map[string]string) map[string]string {
	dst := make(map[string]string)
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func cloneOverrides(src map[string]string) map[string]string {
	dst := make(map[string]string)
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func newModel(spec *apb.Model) *Model {
	return &Model{
		Path:       spec.GetModelPath(),
		Checkpoint: spec.GetCheckpointPath(),
		Acls:       cloneAcls(spec.GetAcls().GetItems()),
		Overrides:  spec.GetOverrides(),
		UUID:       spec.GetUuid(),
	}
}

func (m *Model) clone() *Model {
	return &Model{
		Path:       m.Path,
		Checkpoint: m.Checkpoint,
		Acls:       cloneAcls(m.Acls),
		Overrides:  cloneOverrides(m.Overrides),
		UUID:       bytes.Clone(m.UUID),
	}
}

// MethodStats represents a few statistics for a model method reported by a server.
type MethodStats struct {
	SuccessesPerSecond   float32
	ErrorsPerSecond      float32
	MeanLatencyInSeconds float32
}

// ModelInfo represents the status of a model and method stats reported by a server.
type ModelInfo struct {
	Status protobuf.ModelStatus // Loaded, unloaded, etc.
	Stats  map[string]MethodStats
}

// ModelWithStatus represents a model's static + dynamic state.
type ModelWithStatus struct {
	Model
	Info ModelInfo
}

func (m *ModelWithStatus) clone() *ModelWithStatus {
	return &ModelWithStatus{Model: *m.Model.clone(), Info: m.Info}
}

type actionKind int

const (
	load actionKind = iota
	unload
	update
)

func (k actionKind) String() string {
	switch k {
	case load:
		return "load"
	case unload:
		return "unload"
	case update:
		return "update"
	default:
		return "invalid"
	}
}

// action represents an administrative method the model server can perform on a model.
type action struct {
	kind     actionKind
	ctx      context.Context
	fullName naming.ModelFullName
	model    *Model
	waiter   *waitable.Waitable
}

// State mirrors and manages the state of a remote model server.
//
// All methods on State are thread-safe.
type State struct {
	// Immutable server attributes.
	Addr      string
	DebugAddr string
	DataAddr  string
	Specs     *protobuf.ModelServer

	// Connection to the server.
	client mgrpc.ModeletClient
	conn   *grpc.ClientConn

	mu sync.RWMutex
	// The server state reported by the most recent GetStatus call.
	seen map[naming.ModelFullName]*ModelWithStatus
	// Eventually loaded models when all pending actions finish.
	wanted map[naming.ModelFullName]*Model

	// Requested actions that haven't been sent to the server yet but already reflected in wanted.
	queue     chan *action
	queueStop chan bool

	// Ticker for calling refresh periodically.
	ticker     *time.Ticker
	tickerStop chan bool

	// Last successful refresh time.
	muLastPing sync.Mutex
	lastPing   time.Time

	// Logger to log events such as publish, unpublish, etc.
	eventLogger eventlog.Logger
}

// SeenModels returns a copy of the reported server state.
func (s *State) SeenModels() map[naming.ModelFullName]*ModelWithStatus {
	s.mu.RLock()
	defer s.mu.RUnlock()

	seen := make(map[naming.ModelFullName]*ModelWithStatus, len(s.seen))
	for fullName, model := range s.seen {
		seen[fullName] = model.clone()
	}
	return seen
}

// WantedModels returns a copy of the desired server state.
func (s *State) WantedModels() map[naming.ModelFullName]*Model {
	s.mu.RLock()
	defer s.mu.RUnlock()

	wanted := make(map[naming.ModelFullName]*Model, len(s.wanted))
	for fullName, model := range s.wanted {
		wanted[fullName] = model.clone()
	}
	return wanted
}

// Load asynchronously loads a model.
func (s *State) Load(ctx context.Context, fullName naming.ModelFullName, spec *apb.Model, waiter *waitable.Waitable) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for existing := range s.wanted {
		// Don't check generation values because multiple models with the same full name can't coexist
		// even if they have different generation values.
		if existing == fullName {
			return fmt.Errorf("model %v is already loaded: %w", fullName, errors.ErrAlreadyExists)
		}
	}

	log.V(2).Infof("Loading model %v with %v", fullName, spec)
	model := newModel(spec)
	s.wanted[fullName] = model
	log.Infof("Enqueuing action %v of model %v on state %p, queue size %d", load, fullName, s, len(s.queue))
	s.queue <- &action{load, ctx, fullName, model.clone(), waiter}
	return nil
}

// Update asynchronously the spec on the loaded model replica.
func (s *State) Update(fullName naming.ModelFullName, spec *apb.Model) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.wanted[fullName]; !ok {
		return fmt.Errorf("model %v is not loaded: %w", fullName, errors.ErrNotFound)
	}
	model := newModel(spec)
	s.wanted[fullName] = model
	log.Infof("Enqueuing action %v of model %v on state %p, queue size %d", update, fullName, s, len(s.queue))
	s.queue <- &action{update, context.Background(), fullName, model.clone(), nil}
	return nil
}

// Unload asynchronously unloads a model.
func (s *State) Unload(ctx context.Context, fullName naming.ModelFullName, waiter *waitable.Waitable) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for existing, model := range s.wanted {
		if existing == fullName {
			log.V(2).Infof("Unloading model %v", fullName)
			delete(s.wanted, fullName)
			log.Infof("Enqueuing action %v of model %v on state %p, queue size %d", unload, fullName, s, len(s.queue))
			s.queue <- &action{unload, ctx, fullName, model, waiter}
			return nil
		}
	}

	return fmt.Errorf("model %v is never loaded: %w", fullName, errors.ErrNotFound)
}

// act takes an action.
func (s *State) act(a *action) {
	switch a.kind {
	case load:
		log.V(0).Infof("Loading model %v onto server %v with overrides %v", a.fullName, s.Addr, a.model.Overrides)
		req := &mpb.LoadRequest{
			ModelKey:       a.fullName.ModelFullName(),
			ModelPath:      a.model.Path,
			CheckpointPath: a.model.Checkpoint,
			Acls: &cpb.AccessControlLists{
				Items: a.model.Acls,
			},
			Overrides: a.model.Overrides,
		}
		if _, err := s.client.Load(a.ctx, req); err == nil {
			if a.waiter != nil {
				a.waiter.Add(1)
			}
			s.eventLogger.Log(eventlog.ServingStart, &apb.Model{
				ModelId:        a.fullName.ModelFullName(),
				ModelPath:      a.model.Path,
				CheckpointPath: a.model.Checkpoint,
				Uuid:           a.model.UUID,
			}, s.Addr)
		} else {
			log.Warningf("Failed to load model %v onto server %v", a.fullName, s.Addr)
			// On failure, we don't remove a.fullName from s.wanted, so we can show the failed status in
			// GetStatus responses to the user.
		}
	case update:
		log.V(0).Infof("Updating model %v onto server %v", a.fullName, s.Addr)
		req := &mpb.UpdateLoadedRequest{
			ModelKey: a.fullName.ModelFullName(),
			Acls: &cpb.AccessControlLists{
				Items: a.model.Acls,
			},
		}
		if _, err := s.client.UpdateLoaded(a.ctx, req); err != nil {
			log.Warningf("Failed to update model %v onto server %v (%v)", a.fullName, s.Addr, err)
		}
	case unload:
		log.V(0).Infof("Unloading model %v from server %v", a.fullName, s.Addr)
		req := &mpb.UnloadRequest{
			ModelKey: a.fullName.ModelFullName(),
		}
		if _, err := s.client.Unload(a.ctx, req); err == nil {
			if a.waiter != nil {
				a.waiter.Add(-1)
			}
			s.eventLogger.Log(eventlog.ServingStop, &apb.Model{
				ModelId:        a.fullName.ModelFullName(),
				ModelPath:      a.model.Path,
				CheckpointPath: a.model.Checkpoint,
				Uuid:           a.model.UUID,
			}, s.Addr)
		} else {
			log.Warningf("Failed to unload model %v from server %v (%v)", a.fullName, s.Addr, err)
			// On failure, we put a.fullName back into s.wanted, so the unload failure shows up as a
			// failed status to the user.
			s.mu.Lock()
			s.wanted[a.fullName] = a.model
			s.mu.Unlock()
		}
	default:
		log.Warningf("Unknown action type %T", a)
	}
}

// GetStatus returns information about a server.
func (s *State) GetStatus(ctx context.Context, full bool) (*mpb.GetStatusResponse, error) {
	if s.client == nil {
		return nil, fmt.Errorf("no model server client: %w", errors.ErrFailedPrecondition)
	}
	ctx, cancel := context.WithTimeout(ctx, getStatusTimeout)
	defer cancel()
	return s.client.GetStatus(ctx, &mpb.GetStatusRequest{IncludeFailureReasons: full})
}

// getStatus calls GetStatus on the server and returns the response in an internal format.
func (s *State) getStatus(ctx context.Context) (map[naming.ModelFullName]*ModelInfo, error) {
	if s.client == nil {
		return nil, fmt.Errorf("no model server client: %w", errors.ErrFailedPrecondition)
	}

	ctx, cancel := context.WithTimeout(ctx, getStatusTimeout)
	defer cancel()
	res, err := s.client.GetStatus(ctx, &mpb.GetStatusRequest{IncludeMethodStats: true})
	if err != nil {
		return nil, fmt.Errorf("getStatus RPC error: %w", err)
	}

	seen := make(map[naming.ModelFullName]*ModelInfo)
	for _, model := range res.GetModels() {
		fullName, err := naming.NewModelFullName(model.GetModelKey())
		if err != nil {
			return nil, fmt.Errorf("getStatus got invalid model key: %w", err)
		}
		status, err := protobuf.NewModelStatus(cpb.ModelStatus(model.GetModelStatus().Number()))
		if err != nil {
			return nil, fmt.Errorf("getStatus got invalid model status: %w", err)
		}

		methodStats := make(map[string]MethodStats)
		for _, stats := range model.GetMethodStats() {
			methodStats[stats.GetMethod()] = MethodStats{
				SuccessesPerSecond:   stats.GetSuccessesPerSecond(),
				ErrorsPerSecond:      stats.GetErrorsPerSecond(),
				MeanLatencyInSeconds: stats.GetMeanLatencyOnSuccessPerSecond(),
			}
		}
		seen[fullName] = &ModelInfo{Status: status, Stats: methodStats}
	}
	return seen, nil
}

// initialize sets wanted and seen models of a just created State instance from a running server.
func (s *State) initialize(ctx context.Context, modelFinder ModelFinder) error {
	seen, err := s.getStatus(ctx)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	log.V(3).Infof("The server sees %v", seen)

	for fullName, info := range seen {
		status := info.Status
		modelPb := modelFinder.FindModel(fullName)
		if modelPb == nil {
			log.Infof("Ignoring an unknown model %v with status %v on server %s", fullName, status, s.Addr)
			continue
		}
		log.Infof("Found a model %v with status %v on server %s", fullName, status, s.Addr)
		model := newModel(modelPb)
		if status == protobuf.Loading || status == protobuf.Loaded || status == protobuf.Failed {
			s.wanted[fullName] = model
			// Possibly need to update the model metadata such as ACLs.
			s.queue <- &action{update, context.Background(), fullName, model.clone(), nil}
		}
		s.seen[fullName] = &ModelWithStatus{Model: *model, Info: *info}
	}

	s.muLastPing.Lock()
	s.lastPing = time.Now()
	s.muLastPing.Unlock()
	return nil
}

// Refresh updates seen models to what's reported by the server.
func (s *State) Refresh(ctx context.Context) error {
	seen, err := s.getStatus(ctx)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// s.seen contains previously seen models, including model paths, status, etc.
	// seen contains the most up-to-date view, including only status.
	// Merge seen into s.seen.
	log.V(3).Infof("The server wants %v", s.wanted)
	log.V(3).Infof("The server saw %v", s.seen)
	log.V(3).Infof("The server sees %v", seen)

	// Prune unloaded models.
	for fullName := range s.seen {
		if _, ok := seen[fullName]; !ok {
			log.V(3).Infof("Removing unloaded model %v on server %v", fullName, s.Addr)
			delete(s.seen, fullName)
		}
	}

	// Now, s.seen is a subset of or equal to seen.
	for fullName, info := range seen {
		if previouslySeen, ok := s.seen[fullName]; ok {
			log.V(3).Infof("Updating model %v status from %v to %v on server %v", fullName, previouslySeen.Info.Status, info.Status, s.Addr)
			previouslySeen.Info = *info
			continue
		}
		expected, ok := s.wanted[fullName]
		if !ok {
			log.V(3).Infof("Ignoring unloaded or unrecognized model %v with status %v on server %v", fullName, info.Status, s.Addr)
			continue
		}
		log.V(3).Infof("Seeing expected model %v with status %v for the first time on server %v", fullName, info.Status, s.Addr)
		newlySeen := &ModelWithStatus{Model: *expected.clone(), Info: *info}
		s.seen[fullName] = newlySeen
	}

	s.muLastPing.Lock()
	s.lastPing = time.Now()
	s.muLastPing.Unlock()
	return nil
}

// LastPing returns when the last successful initialize or refresh call happened.
func (s *State) LastPing() time.Time {
	s.muLastPing.Lock()
	defer s.muLastPing.Unlock()
	return s.lastPing
}

// Start starts background state synchronization with the model server.
func (s *State) Start(ctx context.Context, modelFinder ModelFinder) error {
	// The server has just joined. Create a client for calling GetStatus.
	ctx, cancel := context.WithTimeout(ctx, dialTimeout)
	defer cancel()
	var err error
	s.conn, err = env.Get().DialContext(ctx, s.Addr)
	if err != nil {
		return fmt.Errorf("Start failed to create a client for model server %v: %w", s.Addr, err)
	}
	s.client = mgrpc.NewModeletClient(s.conn)

	// Start a goroutine that drains the action queue, stopping when s.Close is called.
	log.Info("Starting a queue that drains pending model server actions")
	// Golang channels need to allocate memory upfront, so use a large but reasonable capacity.
	s.queue = make(chan *action, 1<<20)
	s.queueStop = make(chan bool)
	go func() {
		for {
			select {
			case <-s.queueStop:
				close(s.queueStop)
				return
			case a := <-s.queue:
				log.Infof("Taking action %v of model %v on state %p, queue size %d", a.kind, a.fullName, s, len(s.queue))
				s.act(a)
			}
		}
	}()

	// Make a GetStatus call and use the result to initialize state.
	log.Infof("Initializing state from model server %v", s.Addr)
	if err := s.initialize(ctx, modelFinder); err != nil {
		return fmt.Errorf("Start failed to initialize state: %w", err)
	}

	// Start a goroutine that calls refresh periodically, stopping when s.Close is called.
	log.Infof("Refreshing model server state every %v", refreshPeriod)
	s.ticker = time.NewTicker(refreshPeriod)
	s.tickerStop = make(chan bool)
	go func() {
		for {
			select {
			case <-s.tickerStop:
				close(s.tickerStop)
				return
			case t := <-s.ticker.C:
				if err := s.Refresh(context.TODO()); err != nil {
					log.Warningf("Failed to refresh model server (%s) state: %v", s.Addr, err)
				} else {
					log.V(3).Infof("Refreshed model server state at %v", t)
				}
			}
		}
	}()

	return nil
}

// Close stops synchronization and closes the connection to the model server.
func (s *State) Close() {
	// Don't close the channel here, to prevent the goroutine from seeing an empty action.
	s.queueStop <- true
	<-s.queueStop

	s.ticker.Stop()
	s.tickerStop <- true
	<-s.tickerStop

	s.conn.Close()
}

// New creates a new State instance from a running model server.
func New(addr, debugAddr, dataAddr string, specs *protobuf.ModelServer, eventLogger eventlog.Logger) *State {
	if specs == nil {
		specs = &protobuf.ModelServer{}
	}
	// If dataAddr is empty, set it to be same as addr
	if dataAddr == "" {
		dataAddr = addr
	}
	return &State{
		Addr:        addr,
		DebugAddr:   debugAddr,
		DataAddr:    dataAddr,
		Specs:       specs,
		seen:        make(map[naming.ModelFullName]*ModelWithStatus),
		wanted:      make(map[naming.ModelFullName]*Model),
		eventLogger: eventLogger,
	}
}
