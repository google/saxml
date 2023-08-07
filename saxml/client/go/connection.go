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

// Package connection provides utilities to get connection for address.
package connection

import (
	"context"
	"fmt"
	"sync"
	"time"

	log "github.com/golang/glog"
	"google.golang.org/grpc"
	"saxml/client/go/location"
	"saxml/common/errors"
	"saxml/common/platform/env"
)

const (
	sleepTime   = 10 * time.Second
	purgeTime   = 1 * time.Hour
	dialTimeout = 2 * time.Second
)

type conn struct {
	client      *grpc.ClientConn
	lastAccTime time.Time
}

type connTable struct {
	mu sync.RWMutex
	// Mapping between modelet address to modelet connection and last access time.
	table map[string]*conn
}

func newConnTable() *connTable {
	c := &connTable{table: make(map[string]*conn)}

	go func() {
		// A loop that clears the connections based on last access time.
		// This is intendented to live as long as the client library (client/go/...) lives.
		// TODO(jianlijianli): consider passing in a context to close/cancel this routine.
		for {
			cutoff := time.Now().Add(-purgeTime)
			c.mu.Lock()
			log.V(2).Infof("clearing connTable with %d connections using cutoff time = %v\n", len(c.table), cutoff)
			for addr, conn := range c.table {
				if conn.lastAccTime.Before(cutoff) {
					conn.client.Close()
					delete(c.table, addr) // It's safe to delete and traverse.
					log.V(3).Infof("conneTable removed addr %s\n", addr)
				}
			}
			log.V(2).Infof("after clearing connTable initiated at %v, there are %d connections\n", cutoff, len(c.table))
			c.mu.Unlock()
			time.Sleep(sleepTime)
		}
	}()

	return c
}

// checkAndGet checks the existence of connecton for an addrress and returns connection.
// The returned boolean indicates if connection is found.
func (t *connTable) checkAndGet(addr string) (*grpc.ClientConn, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	connection, found := t.table[addr]
	if found && connection.client != nil {
		connection.lastAccTime = time.Now()
		return connection.client, true
	}
	return nil, false
}

func (t *connTable) getOrCreate(ctx context.Context, addr string) (*grpc.ClientConn, error) {
	existingClient, found := t.checkAndGet(addr)
	if found && existingClient != nil {
		return existingClient, nil
	}

	// Couldn't find connection. Create a new one.
	var newClient *grpc.ClientConn
	var err error
	ctx, cancel := context.WithTimeout(ctx, dialTimeout)
	defer cancel()
	newClient, err = env.Get().DialContext(ctx, addr)
	if err != nil || newClient == nil {
		log.V(3).Infof("getOrCreate create connection for %s failed due to %v\n", addr, err)
		if errors.IsDeadlineExceeded(err) {
			err = fmt.Errorf("Dial to %s exceeded deadline: %w", addr, errors.ErrUnavailable)
		}
		return nil, err
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	existingConn, found := t.table[addr] // Re-check since Dial was not locked.
	if found && existingConn.client != nil {
		newClient.Close()
		existingConn.lastAccTime = time.Now()
		return existingConn.client, nil
	}

	newConn := &conn{client: newClient, lastAccTime: time.Now()}
	t.table[addr] = newConn
	return newClient, nil
}

var globalConnTable *connTable = newConnTable()

// Factory manages connections to a given model.
type Factory interface {
	GetOrCreate(ctx context.Context) (*grpc.ClientConn, string, error)
	Poison(address string)
}

// SaxConnectionFactory resolves backends via SAX admin server and connects to them in a round-robin fashion.
type SaxConnectionFactory struct {
	Location *location.Table // Keeps track a list of addresses for this model.
}

// GetOrCreate selects a server and returns a connection and address to it.
func (f SaxConnectionFactory) GetOrCreate(ctx context.Context) (*grpc.ClientConn, string, error) {
	address, err := f.Location.Pick(ctx)
	if err != nil {
		return nil, address, err
	}
	connection, err := globalConnTable.getOrCreate(ctx, address)
	return connection, address, err

}

// Poison marks a model address invalid.
func (f SaxConnectionFactory) Poison(address string) {
	f.Location.Poison(address)
}

// DirectConnectionFactory connects to the given address directly.
type DirectConnectionFactory struct {
	Address    string
	mu         sync.Mutex
	connection *grpc.ClientConn
}

// GetOrCreate returns a connection and address of the model server.
func (f *DirectConnectionFactory) GetOrCreate(ctx context.Context) (*grpc.ClientConn, string, error) {
	// WithDefaultServiceConfig is required for MBNS. It will be ignored if the server is backed by GRPC.
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.connection != nil {
		return f.connection, f.Address, nil
	}
	connection, err := env.Get().DialContext(ctx, f.Address, grpc.WithDefaultServiceConfig(`{"loadBalancingConfig": [{"round_robin":{}}]}`))
	f.connection = connection
	return f.connection, f.Address, err
}

// Poison marks a model address invalid.
func (f *DirectConnectionFactory) Poison(address string) {
	// No-op.
}
