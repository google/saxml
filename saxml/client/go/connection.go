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
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc"
	"saxml/client/go/location"
	"saxml/common/errors"
	"saxml/common/platform/env"
)

const (
	sleepTime     = 5 * time.Second
	fastPurgeTime = 10 * time.Second
	purgeTime     = 10 * time.Minute
	dialTimeout   = 2 * time.Second
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
			now := time.Now()
			c.mu.Lock()
			log.V(2).Infof("clearing connTable with %d connections: %v %v\n", len(c.table), purgeTime, fastPurgeTime)
			for addr, conn := range c.table {
				shouldClose := false
				if conn.client.GetState() == connectivity.Ready {
					// If the grpc connection is ready but has not been used for quite a while, close it.
					if conn.lastAccTime.Before(now.Add(-purgeTime)) {
						shouldClose = true
						log.V(3).Infof("conneTable removed idle connection to addr %s\n", addr)
					}
				} else if conn.lastAccTime.Before(now.Add(-fastPurgeTime)) {
					// If the grpc connection is _not_ ready and has not been used recently, close it.
					shouldClose = true
					log.V(3).Infof("conneTable removed addr %s\n", addr)
				}
				if shouldClose {
					log.Infof("connTable close %s", addr)
					conn.client.Close()
					delete(c.table, addr) // It's safe to delete and traverse.
				}
			}
			log.V(2).Infof("after clearing connTable with %v/%v, there are %d connections\n", purgeTime, fastPurgeTime, len(c.table))
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
	GetOrCreate(ctx context.Context) (*grpc.ClientConn, error)
}

// SaxConnectionFactory resolves backends via SAX admin server and connects to them in a round-robin fashion.
type SaxConnectionFactory struct {
	Location *location.Table // Keeps track a list of addresses for this model.
}

// GetOrCreate selects a server and returns a connection to it.
func (f SaxConnectionFactory) GetOrCreate(ctx context.Context) (conn *grpc.ClientConn, err error) {
	addr, err := f.Location.Pick(ctx)
	if err == nil {
		conn, err = globalConnTable.getOrCreate(ctx, addr)
	}
	return conn, err
}

// DirectConnectionFactory connects to the given address directly.
type DirectConnectionFactory struct {
	Address    string
	mu         sync.Mutex
	connection *grpc.ClientConn
}

// GetOrCreate returns a connection and address of the model server.
func (f *DirectConnectionFactory) GetOrCreate(ctx context.Context) (conn *grpc.ClientConn, err error) {
	// WithDefaultServiceConfig is required for MBNS. It will be ignored if the server is backed by GRPC.
	f.mu.Lock()
	defer f.mu.Unlock()
	conn = f.connection
	if conn == nil {
		conn, err = env.Get().DialContext(ctx, f.Address,
			grpc.WithDefaultServiceConfig(`{"loadBalancingConfig": [{"round_robin":{}}]}`))
		f.connection = conn
	}
	return conn, err
}
