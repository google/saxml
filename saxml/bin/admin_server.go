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

// admin_server starts a Sax admin server.
package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"flag"
	log "github.com/golang/glog"
	"saxml/admin/admin"
	"saxml/common/platform/env"
	_ "saxml/common/platform/register" // registers a platform
)

var (
	saxCell = flag.String("sax_cell", "", "Sax cell, e.g., /sax/test")
	port    = flag.Int("port", 10000, "server port")
)

func main() {
	ctx := context.Background()
	env.Get().Init(ctx)

	if *saxCell == "" {
		log.Fatal("The sax_cell flag must be set")
	}
	log.Info("Starting the server")

	// If the running server receives a Ctrl+C during interactive local runs or an eviction SIGTERM
	// on Borg, send the signal to this channel to gracefully shut down the server.
	done := make(chan os.Signal, 1)
	signal.Notify(done, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)

	adminServer := admin.NewServer(*saxCell, *port)
	adminServer.EnableStatusPages()
	if err := adminServer.Start(ctx); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
	defer adminServer.Close()

	// Wait for an interrupt or terminate signal.
	<-done
}
