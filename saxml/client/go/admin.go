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

// Package saxadmin provides a library to interact with sax admin.
package saxadmin

import (
	"context"
	"encoding/binary"
	"fmt"
	"hash/maphash"
	"sync"
	"time"

	log "github.com/golang/glog"
	"google.golang.org/grpc"
	"saxml/client/go/skiplist"
	"saxml/common/addr"
	"saxml/common/errors"
	"saxml/common/platform/env"
	"saxml/common/retrier"
	"saxml/common/watchable"

	pb "saxml/protobuf/admin_go_proto_grpc"
	pbgrpc "saxml/protobuf/admin_go_proto_grpc"
)

const (
	// RPC timeout, only intended for admin methods. Data methods have
	// no timeout in general.
	timeout = 10 * time.Second
	// Remember the fact a model is unpublished for this much time
	// before asking the admin server again.
	delayForgetModel = 10 * time.Minute
	// Inserts these many points into the consistent hash ring for
	// each server address.
	numVirtualReplicas = 8
)

// Create Admin server connection.
func establishAdminConn(address string) (*grpc.ClientConn, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	conn, err := env.Get().DialContext(ctx, address)
	if errors.IsDeadlineExceeded(err) {
		err = fmt.Errorf("Dial to admin failed: %w", errors.ErrUnavailable)
	}
	return conn, err
}

// Admin represents a connection to the admin server.
type Admin struct {
	saxCell string // E.g., /sax/bar

	mu sync.Mutex
	// dialing is true if and only if there is an attempt ongoing to open a
	// network connection to the admin server.
	dialing bool
	// conn is the grpc connection to the admin server. Can be nil if
	// the admin is unreachable.
	conn *grpc.ClientConn
	// client is the admin service client.
	client pbgrpc.AdminClient

	// addrs maintains an addrReplica for every model seen by this
	// admin through FindAdddress(). Each addrReplica is the set of
	// model server addresses for the model. The set is lazily
	// replicated from the admin server through WatchAddresses().
	addrs map[string]*addrReplica
}

// TODO(zhifengc): consider abstracting out module providing a
// resettable sync.Once interface, which can be tested separatedly.
func (a *Admin) getAdminClient(ctx context.Context) (pbgrpc.AdminClient, error) {
	a.mu.Lock()
	// A quick check if a.client is established already.
	if a.client != nil {
		defer a.mu.Unlock()
		return a.client, nil
	}
	// Makes sure there is only one thread attempting to dial to the admin server.
	if a.dialing {
		defer a.mu.Unlock()
		return nil, fmt.Errorf("Dialing to admin: %w", errors.ErrResourceExhausted)
	}
	a.dialing = true
	a.mu.Unlock()

	// Ensures a.dialing is set to false when this function ends.
	defer func() {
		a.mu.Lock()
		defer a.mu.Unlock()
		a.dialing = false
	}()

	addr, err := addr.FetchAddr(ctx, a.saxCell)
	if err != nil {
		return nil, err
	}

	conn, err := establishAdminConn(addr)
	if err != nil {
		return nil, err
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	a.conn, a.client = conn, pbgrpc.NewAdminClient(conn)
	return a.client, nil
}

func (a *Admin) poison() {
	a.mu.Lock()
	conn := a.conn
	a.conn = nil
	a.client = nil
	a.mu.Unlock()

	if conn != nil {
		conn.Close()
	}
}

func (a *Admin) retry(ctx context.Context, callback func(client pbgrpc.AdminClient) error) error {
	action := func() error {
		client, err := a.getAdminClient(ctx)
		if err == nil {
			err = callback(client)
		}
		if errors.AdminShouldPoison(err) {
			a.poison()
		}
		return err
	}
	return retrier.Do(ctx, action, errors.AdminShouldRetry)
}

// Publish publishes a model.
func (a *Admin) Publish(ctx context.Context, modelID, modelPath, checkpointPath string, numReplicas int, overrides map[string]string) error {
	req := &pb.PublishRequest{
		Model: &pb.Model{
			ModelId:              modelID,
			ModelPath:            modelPath,
			CheckpointPath:       checkpointPath,
			RequestedNumReplicas: int32(numReplicas),
			Overrides:            overrides,
		},
	}

	return a.retry(ctx, func(client pbgrpc.AdminClient) error {
		_, err := client.Publish(ctx, req)
		return err
	})
}

// Update updates the model definition of a published model.
func (a *Admin) Update(ctx context.Context, model *pb.Model) error {
	req := &pb.UpdateRequest{Model: model}
	return a.retry(ctx, func(client pbgrpc.AdminClient) error {
		_, err := client.Update(ctx, req)
		return err
	})
}

// Unpublish unpublishes a model.
func (a *Admin) Unpublish(ctx context.Context, modelID string) error {
	req := &pb.UnpublishRequest{
		ModelId: modelID,
	}
	return a.retry(ctx, func(client pbgrpc.AdminClient) error {
		var err error
		_, err = client.Unpublish(ctx, req)
		return err
	})
}

// List lists the status of a published model.
func (a *Admin) List(ctx context.Context, modelID string) (*pb.PublishedModel, error) {
	req := &pb.ListRequest{
		ModelId: modelID,
	}
	var res *pb.ListResponse
	err := a.retry(ctx, func(client pbgrpc.AdminClient) error {
		var err error
		res, err = client.List(ctx, req)
		return err
	})
	if err != nil {
		return nil, err
	}
	if len(res.GetPublishedModels()) != 1 {
		return nil, fmt.Errorf("one model expected for %s but found %d %w", modelID, len(res.GetPublishedModels()), errors.ErrNotFound)
	}
	return res.GetPublishedModels()[0], nil
}

// ListAll lists the status of all published models.
func (a *Admin) ListAll(ctx context.Context) (*pb.ListResponse, error) {
	req := &pb.ListRequest{}
	var res *pb.ListResponse
	err := a.retry(ctx, func(client pbgrpc.AdminClient) error {
		var err error
		res, err = client.List(ctx, req)
		return err
	})
	if err != nil {
		return nil, err
	}
	return res, nil
}

// Stats returns the status of the cell
func (a *Admin) Stats(ctx context.Context, modelID string) (*pb.StatsResponse, error) {
	req := &pb.StatsRequest{
		ModelId: modelID,
	}
	var res *pb.StatsResponse
	err := a.retry(ctx, func(client pbgrpc.AdminClient) error {
		var err error
		res, err = client.Stats(ctx, req)
		return err
	})
	if err != nil {
		return nil, err
	}
	return res, nil
}

// addrReplica maintains a set of server addresses for a model.
type addrReplica struct {
	modelID  string
	hashSeed maphash.Seed

	mu  sync.Mutex
	err error

	// All replica addresses (strings) are hashed uniformly into [0,
	// uint64max]. These hashes are kept in order in 'hash'.  For each
	// hash value h in 'hash', addr[h] maps it back to the address.
	addr map[uint64]string
	hash *skiplist.T[uint64]
}

func intcmp(a *uint64, b *uint64) (cmp int) {
	if *a > *b {
		cmp = 1
	} else if *a < *b {
		cmp = -1
	}
	return cmp
}

func newAddrReplica(model string) *addrReplica {
	a := &addrReplica{
		modelID:  model,
		hashSeed: maphash.MakeSeed(),
	}
	a.reset(nil)
	return a
}

func (a *addrReplica) reset(addrs []string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.err = nil
	a.addr = make(map[uint64]string)
	a.hash = skiplist.New[uint64](intcmp)
	if addrs != nil {
		for _, addr := range addrs {
			a.addLocked(addr)
		}
	}
}

func (a *addrReplica) hashAddr(addr string, index uint64) uint64 {
	var h maphash.Hash
	h.SetSeed(a.hashSeed)
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, index)
	h.Write(b)
	h.WriteString(addr)
	return h.Sum64()
}

func (a *addrReplica) hashUint64(value uint64) uint64 {
	var h maphash.Hash
	h.SetSeed(a.hashSeed)
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, value)
	h.Write(b)
	return h.Sum64()
}

func (a *addrReplica) setError(err error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.err = err
}

func (a *addrReplica) add(addr string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.addLocked(addr)
}

func (a *addrReplica) addLocked(addr string) {
	for i := uint64(0); i < numVirtualReplicas; i++ {
		h := a.hashAddr(addr, i)
		if a.hash.Insert(&h, false /* dup not ok*/) {
			a.addr[h] = addr
		}
	}
}

func (a *addrReplica) del(addr string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for i := uint64(0); i < numVirtualReplicas; i++ {
		h := a.hashAddr(addr, i)
		if a.hash.Remove(&h) {
			delete(a.addr, h)
		}
	}
}

// Update updates the set of server addresses according to the
// incremental updates sent back from the admin server through
// chanWatchResult.
func (a *addrReplica) Update(chanWatchResult chan *WatchResult) error {
	for wr := range chanWatchResult {
		log.Infof("addrReplica.Update(%s) %v", a.modelID, wr)
		if wr.Err != nil {
			a.setError(wr.Err)
			return wr.Err
		}
		if wr.Result.Data != nil {
			// After a long network partition or the first time using
			// the model, the client may get a full set from the admin
			// server. It should happen rarely.
			log.Infof("Receive a full set for %s: %v", a.modelID, wr)
			a.reset(wr.Result.Data.ToList())
		}
		for _, m := range wr.Result.Log {
			switch m.Kind {
			case watchable.Add:
				a.add(m.Val)
			case watchable.Del:
				a.del(m.Val)
			default:
				log.Warningf("Unexpected Kind: %v", m.Kind)
			}
		}
	}
	return nil
}

// Pick picks one address using the basic consistent hashing
// algorithm.
func (a *addrReplica) Pick(seed uint64) (string, error) {
	a.mu.Lock()
	defer a.mu.Unlock()
	addr, err := "", a.err
	if err == nil && a.hash.Count() == 0 {
		err = errors.ErrUnavailable
	}
	if err == nil {
		h := a.hashUint64(seed)
		it := a.hash.LowerBound(&h)
		var val uint64
		if it.IsNil() {
			val = *a.hash.First().Value()
		} else {
			val = *it.Value()
		}
		addr = a.addr[val]
	}
	return addr, err
}

// FindAddress queries the local replica of the server address set to
// get one server address randomly. Seed specifies the random seed.
func (a *Admin) FindAddress(ctx context.Context, model string, seed uint64) (string, error) {
	a.mu.Lock()
	ar, ok := a.addrs[model]
	if !ok {
		// First time to access the model, setup the addrReplica and
		// arrange a background go routine to keep it updated.
		ar = newAddrReplica(model)
		a.addrs[model] = ar
		chanWatchResult := make(chan *WatchResult)
		go a.WatchAddresses(context.Background(), model, chanWatchResult)
		go func() {
			err := ar.Update(chanWatchResult)
			if err == nil {
				log.Fatalf("addrReplica.Update for %s exited ok unexpectedly.", model)
			} else {
				// There is an error (e.g., the model is unpublished).
				log.Infof("addrReplica.Update for %s got error %v", model, err)
			}
			time.AfterFunc(delayForgetModel, func() {
				a.mu.Lock()
				defer a.mu.Unlock()
				if newAr, ok := a.addrs[model]; ok && ar != newAr {
					log.Infof("remove addrReplica unexpectedly: %s ", model)
				}
				delete(a.addrs, model)
			})
		}()
	}
	a.mu.Unlock()

	return ar.Pick(seed)
}

// WatchResult encapsulates the changes to the server addresses for a
// model.
type WatchResult struct {
	// Err is the error returned by the WatchAddresses().
	Err error
	// Result represents the changes to the server addresses if Err is nil.
	Result *watchable.WatchResult
}

// WatchAddresses replicates the changes to the model's server addresses.
//
// The caller of WatchAddresses() receives all the changes though the
// chanWatchResult.  WatchAddresses intentionally never stops until
// the model is unpublished.
func (a *Admin) WatchAddresses(ctx context.Context, model string, chanWatchResult chan *WatchResult) {
	var serverID string
	var seqno int32
	for {
		req := &pb.WatchLocRequest{
			ModelId:       model,
			AdminServerId: serverID,
			Seqno:         seqno,
		}
		var resp *pb.WatchLocResponse
		err := a.retry(ctx, func(client pbgrpc.AdminClient) error {
			var err error
			resp, err = client.WatchLoc(ctx, req)
			return err
		})
		if err != nil {
			chanWatchResult <- &WatchResult{Err: err}
			if errors.IsNotFound(err) {
				return
			}
			// For other errors, we reset the process.
			log.Errorf("Unexpected WatchLoc rpc call error: %v", err)
			serverID, seqno = "", 0
			time.Sleep(time.Second)
			continue
		}
		serverID = resp.GetAdminServerId()
		w := watchable.FromProto(resp.GetResult())
		chanWatchResult <- &WatchResult{Result: w}
		seqno = w.Next
	}
}

// WaitForReady blocks until at least numReplicas replicas are ready.
func (a *Admin) WaitForReady(ctx context.Context, modelID string, numReplicas int) error {
	req := &pb.WaitForReadyRequest{
		ModelId:     modelID,
		NumReplicas: int32(numReplicas),
	}
	err := a.retry(ctx, func(client pbgrpc.AdminClient) error {
		_, err := client.WaitForReady(ctx, req)
		return err
	})
	return err
}

type openedAdmin struct {
	mu     sync.Mutex
	admins map[string]*Admin
}

func (o *openedAdmin) Get(saxCell string) *Admin {
	o.mu.Lock()
	defer o.mu.Unlock()
	if found, ok := o.admins[saxCell]; ok {
		return found
	}
	ret := &Admin{
		saxCell: saxCell,
		addrs:   make(map[string]*addrReplica),
	}
	o.admins[saxCell] = ret
	return ret
}

var adminCache *openedAdmin = &openedAdmin{admins: make(map[string]*Admin)}

// Open returns an admin interface for users to query system state, such as listing all models.
func Open(saxCell string) *Admin {
	return adminCache.Get(saxCell)
}
