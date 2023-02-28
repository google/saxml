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

// Package location fixes the modelet server address for a model in sax client.
package location

import (
	"context"
	"fmt"
	"math/rand"
	"sync"

	log "github.com/golang/glog"
	"saxml/client/go/saxadmin"
	"saxml/common/errors"
)

// Table holds the address information for a given model.
type Table struct {
	model             string
	preferredNumConns int
	admin             *saxadmin.Admin

	mu            sync.RWMutex
	lastAddrIndex int
	addrList      []string
	addrIndex     map[string]int // Mapping between address and its index in the addrList.
}

// add adds a list of addresses.
//
// It dedups internally and can take empty list as input.
func (t *Table) add(addrs []string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	for _, idx := range rand.Perm(len(addrs)) {
		addr := addrs[idx]
		if _, found := t.addrIndex[addr]; !found {
			t.addrIndex[addr] = len(t.addrList)
			t.addrList = append(t.addrList, addr)
		}
	}
}

// shouldTryRefill checks if we should retry and if yes, how many to ask.
func (t *Table) shouldTryRefill() (bool, int) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	has := len(t.addrList)
	ask := t.preferredNumConns - has

	if ask <= 0 {
		// No need to refill if there are plenty.
		return false, 0
	}

	return true, ask
}

// refill refills addresses when necessary.
func (t *Table) refill(ctx context.Context) {
	shouldRetry, ask := t.shouldTryRefill()
	if !shouldRetry {
		log.V(2).Infof("No need to refill")
		return
	}

	if t.admin == nil {
		log.Warningf("Refill fired but failed for model %s, unable to get admin\n", t.model)
		return
	}

	addresses, err := t.admin.FindAddresses(ctx, t.model, ask)
	if err != nil {
		log.V(2).Infof("Refill fired but FindAddresses failed for model %s because %v\n", t.model, err)
		return
	}
	if len(addresses) == 0 {
		log.Infof("No addresses found for %v", t.model)
	} else {
		log.V(2).Infof("Got addresses %v for model %v", addresses, t.model)
	}
	t.add(addresses)
}

// Poison marks a model address invalid.
// Note:
//   - by design, Poison does not notify Admin.
//   - we do not keep a denylist: all addresses returned by admin are considered valid.
func (t *Table) Poison(addr string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	idx, found := t.addrIndex[addr]
	if !found {
		return
	}
	// Remove address `addr` from both addrList and addrIndex if it exists.
	delete(t.addrIndex, addr)
	currLen := len(t.addrList)
	if idx < 0 || idx >= currLen {
		log.V(2).Infof("Poison(%s) for model %s has unexpected index. Found index %d for a array of size %d\n", addr, t.model, idx, currLen)
		return
	}
	// Remove the address at index.
	if idx != currLen-1 {
		// If address is not at last, swap with the last.
		t.addrList[idx] = t.addrList[currLen-1]
		t.addrIndex[t.addrList[idx]] = idx
	}
	// reduce list size by 1.
	t.addrList = t.addrList[:currLen-1]
	log.V(2).Infof("Poison(%s) for model %s succeeded. Now model has %d addresses\n", addr, t.model, len(t.addrList))
}

// Pick picks a random server address for a model.
func (t *Table) Pick(ctx context.Context) (string, error) {
	t.refill(ctx)

	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.addrList) == 0 {
		// This error is retriable.
		return "", fmt.Errorf("pick for model %s unable to find server %w", t.model, errors.ErrUnavailable)
	}
	// Round-robin.
	t.lastAddrIndex = (t.lastAddrIndex + 1) % len(t.addrList)
	return t.addrList[t.lastAddrIndex], nil
}

// NewLocationTable create a new Table for a model.
func NewLocationTable(admin *saxadmin.Admin, name string, numConn int) *Table {
	return &Table{
		model:             name,
		preferredNumConns: numConn,
		admin:             admin,
		lastAddrIndex:     -1,
		addrList:          []string{},
		addrIndex:         make(map[string]int),
	}
}
