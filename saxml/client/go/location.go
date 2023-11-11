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

// Package location picks server addresses randomly.
package location

import (
	"context"
	"math/rand"
	"sync"

	"saxml/client/go/saxadmin"
)

// Table holds the address information for a given model.
type Table struct {
	model             string
	preferredNumConns uint64
	admin             *saxadmin.Admin

	mu       sync.RWMutex
	nextSeed uint64
}

// Pick picks a random server address for a model.
func (t *Table) Pick(ctx context.Context) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	addr, err := t.admin.FindAddress(ctx, t.model, t.nextSeed)
	t.nextSeed = (t.nextSeed + 1) % t.preferredNumConns // Round-robin.
	return addr, err
}

// NewLocationTable create a new Table for a model.
func NewLocationTable(admin *saxadmin.Admin, name string, numConn int) *Table {
	return &Table{
		model:             name,
		preferredNumConns: uint64(numConn),
		admin:             admin,
		nextSeed:          rand.Uint64() % uint64(numConn),
	}
}
