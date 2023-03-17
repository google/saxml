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

// Package watchable provides an abstraction Watchable.
//
// Watchable keeps track of a set of strings, whose initial state is
// just an empty set. It allows producers to add or remove
// strings. These mutations can be replicated asynchronously to other
// consumers. Each mutation is assigned with a unique, ascending
// sequence number. Consumers can call Watchable.Watch to receive all
// mutations lazily after a given sequence number.
//
// E.g.,
//
//		w := watchable.New()
//
//		// Producer
//		... many w.Add() or w.Del()
//		w.Add("ip:portA")
//		w.Add("ip:PortB")
//
//	   // Shutdown the watcher and resets the watcher.
//	   w.Close()
//
//		// Consumer
//		// The consumer uses a map to keep track of the set of strings.
//		state := NewDataSet()
//		next := 0
//		for {
//		  err, result := w.Watch(ctx, next)
//		  if err != nil {
//		    ...
//		  }
//		  if result.Data != nil {
//		    // Watch() tells us a full state.
//		    state = result.Data
//		  }
//		  // Watch() tells us the mutations needed to
//		  // update our local state.
//		  state.Apply(result.Log)
//		  // Now we are caught up the sequence number.
//		  next = result.Next
//		}
//
// In the sax admin/client protocol, we expect to use one Watchable on
// the admin server to maintain the live set of servers serving a
// model. The client will call the admin server remotely to watch the
// mutation to the set and maintain its local copy of the set.
package watchable

import (
	"context"
	"sync"

	log "github.com/golang/glog"

	pb "saxml/protobuf/admin_go_proto_grpc"
)

// DataSet is an unordered set of strings.
type DataSet struct {
	set map[string]struct{}
}

// NewDataSet returns a new DataSet instance.
func NewDataSet() *DataSet {
	return &DataSet{set: make(map[string]struct{})}
}

// Size returns the number of elements in the set.
func (d *DataSet) Size() int {
	return len(d.set)
}

// Add adds val into the set.
func (d *DataSet) Add(val string) {
	d.set[val] = struct{}{}
}

// Del removes val from the set.
func (d *DataSet) Del(val string) {
	delete(d.set, val)
}

// Exist returns true iff val exists in the set.
func (d *DataSet) Exist(val string) bool {
	_, ok := d.set[val]
	return ok
}

// Copy returns a copy of a dataset.
func (d *DataSet) Copy() *DataSet {
	ret := NewDataSet()
	for k := range d.set {
		ret.Add(k)
	}
	return ret
}

// ToList returns strings in the set in a list.
func (d *DataSet) ToList() []string {
	var ret []string
	for k := range d.set {
		ret = append(ret, k)
	}
	return ret
}

// MutationKind indicates either an addition (add) or a removal (del).
type MutationKind int

const (
	// Add inserts a string to the set.
	Add MutationKind = iota
	// Del removes a string from the set.
	Del
)

// Mutation is a mutation on DataSet.
type Mutation struct {
	Kind MutationKind
	Val  string
}

// ChangeLog is a sequence of Mutations.
type ChangeLog []Mutation

// Apply applies the change log muts to the dataset d in the given
// order.
func (d *DataSet) Apply(muts ChangeLog) {
	for _, mut := range muts {
		switch mut.Kind {
		case Add:
			d.Add(mut.Val)
		case Del:
			d.Del(mut.Val)
		default:
			log.Warningf("Unexpected Kind: %v", mut.Kind)
		}
	}
}

// Watchable keeps track of a set of strings and allows its clients to
// watch mutations to the set.
type Watchable struct {
	chDone chan bool
	mu     sync.Mutex
	cond   *sync.Cond

	// Protected by mu.
	//
	// True iff Close() is called.
	done bool

	// Protected by mu.
	//
	// The sequence number assigned to the next mutation.
	nextSeqno int32

	// Protected by mu.
	//
	// data is the result of application of mutations [0 .. nextSeqno - len(mutation)).
	data DataSet

	// Protected by mu.
	//
	// mutation contain all mutations with sequnece numbers [nextSeqno - len(mutation), seqno).
	mutation []Mutation
}

// New returns a Watchable with an initial empty set.
func New() *Watchable {
	m := &Watchable{
		data: *NewDataSet(),
	}
	m.chDone = make(chan bool)
	m.cond = sync.NewCond(&m.mu)
	return m
}

// packLocked applies a prefix of mutation to data so that mutation do not
// grow too large.
func (w *Watchable) packLocked() {
	if len(w.mutation) >= 2 && len(w.mutation) > 2*w.data.Size() {
		// We keeps len(mutation) not twice larger than w.data.Size().
		num := len(w.mutation) / 2
		w.data.Apply(w.mutation[:num])
		n := copy(w.mutation, w.mutation[num:])
		w.mutation = w.mutation[:n]
	}
}

// Add adds a new val into the set.
func (w *Watchable) Add(val string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.mutation = append(w.mutation, Mutation{Kind: Add, Val: val})
	w.nextSeqno++
	w.packLocked()
	w.cond.Broadcast()
}

// Del removes a string from the set.
func (w *Watchable) Del(val string) {
	w.cond.L.Lock()
	defer w.cond.L.Unlock()
	w.mutation = append(w.mutation, Mutation{Kind: Del, Val: val})
	w.nextSeqno++
	w.packLocked()
	w.cond.Broadcast()
}

// WatchResult is the result from a Watch call.
type WatchResult struct {
	// If Data is not nil, Data.Apply(Log) represents the final state
	// of mutations[0 .. Token).
	//
	// If Data is nil, Log contains mutation[startSeqno .. Token).
	Data *DataSet
	Log  ChangeLog
	Next int32
}

// ToProto encodes a WatchResult struct into a WatchResult proto.
func (r *WatchResult) ToProto() *pb.WatchResult {
	var mutations []*pb.WatchResult_Mutation
	for _, m := range r.Log {
		switch m.Kind {
		case Add:
			mutations = append(mutations,
				&pb.WatchResult_Mutation{Kind: &pb.WatchResult_Mutation_Addition{Addition: m.Val}})
		case Del:
			mutations = append(mutations,
				&pb.WatchResult_Mutation{Kind: &pb.WatchResult_Mutation_Deletion{Deletion: m.Val}})
		default:
			log.Warningf("Unexpected Kind: %v", m.Kind)
		}
	}
	ret := &pb.WatchResult{
		NextSeqno:  int32(r.Next),
		HasFullset: r.Data != nil,
		Changelog:  mutations,
	}
	if ret.GetHasFullset() {
		ret.Values = r.Data.ToList()
	}
	return ret
}

// FromProto converts a WatchResult proto into a WatchResult struct.
func FromProto(p *pb.WatchResult) *WatchResult {
	var data *DataSet
	if !p.GetHasFullset() {
		data = nil
	} else {
		data = NewDataSet()
		for _, val := range p.GetValues() {
			data.Add(val)
		}
	}
	var changes ChangeLog
	for _, m := range p.GetChangelog() {
		switch m.GetKind().(type) {
		case *pb.WatchResult_Mutation_Addition:
			changes = append(changes, Mutation{Kind: Add, Val: m.GetAddition()})
		case *pb.WatchResult_Mutation_Deletion:
			changes = append(changes, Mutation{Kind: Del, Val: m.GetDeletion()})
		default:
			log.Warningf("Missing cases: %v", m.GetKind())
		}
	}
	return &WatchResult{Next: p.GetNextSeqno(), Data: data, Log: changes}
}

// Close shutdowns this Watchable.
func (w *Watchable) Close() {
	close(w.chDone)
	w.cond.L.Lock()
	defer w.cond.L.Unlock()
	w.done = true
	w.nextSeqno = 0
	w.data = *NewDataSet()
	w.mutation = nil
	w.cond.Broadcast()
}

// Watch returns all mutations[startSeqno:] if the change log
// mutation[...]  still contain all mutations[startSeqno:]. Otherwise,
// returns a string set (result.Data), which is the result of
// mutations[0, some seqNo), and all mutations after the seqNo.
func (w *Watchable) Watch(ctx context.Context, startSeqno int32) (*WatchResult, error) {
	ch := make(chan *WatchResult, 1)
	go func() {
		var result *WatchResult

		w.cond.L.Lock()
	Loop:
		for {
			seqno := w.nextSeqno - int32(len(w.mutation))
			switch {
			case w.done:
				break Loop
			case startSeqno < seqno:
				// mutation[] no longer contains all mutations after
				// startSeqno. We returns a full set (the result of
				// mutations[0, seqno) and all mutations >= seqno.
				retDataSet := w.data.Copy()
				retChangeLog := make(ChangeLog, len(w.mutation))
				copy(retChangeLog, w.mutation)
				result = &WatchResult{retDataSet, retChangeLog, w.nextSeqno}
				break Loop
			case startSeqno < w.nextSeqno:
				// mutation[] contains all mutations afer startSeqno. We
				// only need to return mutation[startSeqno:].
				prefixLen := startSeqno - seqno
				suffixLen := int32(len(w.mutation)) - prefixLen
				retChangeLog := make(ChangeLog, suffixLen)
				copy(retChangeLog, w.mutation[prefixLen:])
				result = &WatchResult{nil, retChangeLog, w.nextSeqno}
				break Loop
			case startSeqno > w.nextSeqno:
				// startSeqno is ahead of us. We simply return
				// nextSeqno.  If the client is strictly following the
				// protocol, this should not happen.
				result = &WatchResult{nil, nil, w.nextSeqno}
				break Loop
			case startSeqno == w.nextSeqno:
				// Blocks until there is the next mutation.
				w.cond.Wait()
			}
		}
		w.cond.L.Unlock()

		if result != nil {
			// Send result without holding the lock.
			ch <- result
		}
	}()

	select {
	case <-ctx.Done():
		// Cancelled or timed out.
		return nil, ctx.Err()
	case ret := <-ch:
		return ret, nil
	case <-w.chDone:
		// Tells the consumer to reset its state.
		return &WatchResult{NewDataSet(), nil, 0}, nil
	}
}
