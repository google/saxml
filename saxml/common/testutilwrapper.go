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

// testutilwrapper exports functions from the testutil package for Cgo.
package main

import (
	"context"
	"os"
	"sync"

	log "github.com/golang/glog"
	_ "saxml/common/platform/register" // registers a platform
	"saxml/common/testutil"
)

import "C"

const (
	fsRootPattern = "sax-test-fsroot-"
)

var (
	mu         sync.Mutex
	allClosers map[string][]chan struct{} = make(map[string][]chan struct{}) // saxCell as key
	fsRoots    map[string]string          = make(map[string]string)          // saxCell as key
)

func startLocalTestCluster(saxCell string, modelType testutil.ModelType, adminPort int) {
	fsRoot, err := os.MkdirTemp("", fsRootPattern)
	if err != nil {
		log.Fatalf("startLocalTestCluster failed: %v", err)
	}

	cluster := testutil.NewCluster(saxCell).SetFsRoot(fsRoot)
	cluster.SetModelType(modelType).SetAdminPort(adminPort)
	ctx := context.Background()
	closers, err := cluster.StartInternal(ctx)
	if err != nil {
		log.Fatalf("startLocalTestCluster failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if _, ok := fsRoots[saxCell]; ok {
		log.Fatalf("startLocalTestCluster failed: %v already started", saxCell)
	}
	fsRoots[saxCell] = fsRoot
	if _, ok := allClosers[saxCell]; ok {
		log.Fatalf("startLocalTestCluster failed: %v already started", saxCell)
	}
	allClosers[saxCell] = closers
}

func stopLocalTestCluster(saxCell string) {
	mu.Lock()
	if _, ok := allClosers[saxCell]; !ok {
		log.Fatalf("stopLocalTestCluster failed: %v not started", saxCell)
	}
	closers := allClosers[saxCell]
	delete(allClosers, saxCell)
	if _, ok := fsRoots[saxCell]; !ok {
		log.Fatalf("stopLocalTestCluster failed: %v not started", saxCell)
	}
	fsRoot := fsRoots[saxCell]
	delete(fsRoots, saxCell)
	mu.Unlock()

	for _, closer := range closers {
		close(closer)
	}

	if err := os.RemoveAll(fsRoot); err != nil {
		log.Fatalf("stopLocalTestCluster failed: %v", err)
	}
}

//export sax_set_up
func sax_set_up(saxCellStr *C.char, saxCellSize C.int) {
	saxCell := C.GoStringN(saxCellStr, saxCellSize)
	fsRoot, err := os.MkdirTemp("", fsRootPattern)
	if err != nil {
		log.Fatalf("sax_set_up failed: %v", err)
	}
	ctx := context.Background()
	if err := testutil.SetUpInternal(ctx, saxCell, fsRoot); err != nil {
		log.Fatalf("sax_set_up failed: %v", err)
	}
}

//export sax_start_local_test_cluster
func sax_start_local_test_cluster(saxCellStr *C.char, saxCellSize C.int, modelType C.int, adminPort C.int) {
	saxCell := C.GoStringN(saxCellStr, saxCellSize)
	startLocalTestCluster(saxCell, testutil.ModelType(modelType), int(adminPort))
}

//export sax_stop_local_test_cluster
func sax_stop_local_test_cluster(saxCellStr *C.char, saxCellSize C.int) {
	saxCell := C.GoStringN(saxCellStr, saxCellSize)
	stopLocalTestCluster(saxCell)
}

func main() {}
