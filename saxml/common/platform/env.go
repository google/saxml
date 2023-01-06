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

// Package env centralizes and redirects platform-specific API calls.
package env

import (
	"context"
	"net"
	"net/http"

	log "github.com/golang/glog"
	"google.golang.org/grpc"

	pb "saxml/protobuf/admin_go_proto_grpc"
)

var globalEnv Env

// StatusPageKind defines the kinds of server status pages.
type StatusPageKind int

// StatusPageKind value definitions.
const (
	RootStatusPage StatusPageKind = iota
	ModelStatusPage
	ServerStatusPage
)

// StatusPageData contains data needed for generating a server status page.
type StatusPageData struct {
	Kind    StatusPageKind
	SaxCell string
	Models  []*pb.PublishedModel
	Servers []*pb.JoinedModelServer
}

// Server defines methods every platform's server type must support.
type Server interface {
	// GRPCServer returns the underlying gRPC server. It should be only used for service registration.
	GRPCServer() *grpc.Server
	// CheckACLs returns nil iff the principal extracted from a request context passes an ACL check.
	CheckACLs(ctx context.Context, acls []string) error

	// WriteStatusPage writes the status page of an admin server.
	WriteStatusPage(w http.ResponseWriter, data *StatusPageData) error

	// Serve starts serving.
	Serve(lis net.Listener) error
	// Stop stops serving.
	Stop()
}

// Env defines methods every platform must support.
type Env interface {
	// Init initializes the platform, such as parsing command line flags, in non-test binaries.
	Init(ctx context.Context)

	// ReadFile reads the content of a file.
	ReadFile(ctx context.Context, path string) ([]byte, error)
	// ReadFile reads the content of a file, caching the result on repeated reads if possible.
	ReadCachedFile(ctx context.Context, path string) ([]byte, error)
	// WriteFile writes the content of a file.
	WriteFile(ctx context.Context, path string, data []byte) error
	// WriteFileAtomically writes the content of a file to file systems without versioning support.
	WriteFileAtomically(ctx context.Context, path string, data []byte) error
	// FileExists checks the existence of a file.
	FileExists(ctx context.Context, path string) (bool, error)

	// RootDir returns the directory path where all Sax cells store their metadata.
	RootDir(ctx context.Context) string
	// CreateDir creates a directory.
	CreateDir(ctx context.Context, path, acl string) error
	// ListSubdirs lists subdirectories in a directory.
	ListSubdirs(ctx context.Context, path string) ([]string, error)
	// DirExists checks the existence of a directory.
	DirExists(ctx context.Context, path string) (bool, error)

	// CheckACLs returns nil iff the given principal passes an ACL check.
	CheckACLs(principal string, acls []string) error

	// Watch watches for content changes in a file and sends the new content on the returned channel.
	Watch(ctx context.Context, path string) (<-chan []byte, error)
	// Lead blocks until it acquires exclusive access to a file. The caller should arrange calling
	// close() on the returned channel to release the exclusive lock.
	Lead(ctx context.Context, path string) (chan<- struct{}, error)

	// PickUnusedPort picks an unused port.
	PickUnusedPort() (port int, err error)
	// DialContext establishes a connection to the target.
	DialContext(ctx context.Context, target string) (*grpc.ClientConn, error)
	// RequiredACLNamePrefix returns the string required to prefix all ACL names.
	RequiredACLNamePrefix() string
	// NewServer creates a server.
	NewServer() (Server, error)
}

// Register lets a platform register its Env implementation.
//
// This should only get called in init functions, so there is no need for mutex protection.
func Register(env Env) { globalEnv = env }

// Get returns the registered Env implementation.
func Get() Env {
	if globalEnv == nil {
		log.Fatal("No platform environment is registered.")
	}
	return globalEnv
}
