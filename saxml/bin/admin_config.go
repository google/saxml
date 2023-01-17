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

// admin_config writes a config.proto to a directory.
package main

import (
	"context"

	"flag"
	log "github.com/golang/glog"
	"saxml/common/cell"
	"saxml/common/config"
	"saxml/common/naming"
	"saxml/common/platform/env"
	_ "saxml/common/platform/register" // registers a platform
)

var (
	saxCell = flag.String("sax_cell", "", "Sax cell, e.g., /sax/test")
	fsRoot  = flag.String("fs_root", "", "FS root, e.g., /tmp/sax-fs-root")
)

func main() {
	ctx := context.Background()
	env.Get().Init(ctx)

	if *saxCell == "" || *fsRoot == "" {
		log.Fatal("Usage: admin_config --sax_cell=/sax/<cell> --sax_root=... --fs_root=...")
	}
	if _, err := naming.SaxCellToCell(*saxCell); err != nil {
		log.Fatalf("Invalid sax_cell flag %s: %v", *saxCell, err)
	}

	// Create the Sax root directory if it doesn't exist.
	path, err := cell.Path(ctx, *saxCell)
	if err != nil {
		log.Fatalf("Failed to get path for Sax cell %s: %v", *saxCell, err)
	}
	if exists, err := env.Get().DirExists(ctx, path); err != nil {
		log.Fatalf("Failed to check path %s existence: %v", path, err)
	} else if !exists {
		if err := env.Get().CreateDir(ctx, path, ""); err != nil {
			log.Fatalf("Failed to create path %s: %v", path, err)
		}
	}

	// Write a Sax config.proto file into the Sax cell subdirectory in the Sax root directory.
	if err := config.Create(ctx, *saxCell, *fsRoot, ""); err != nil {
		log.Fatalf("Failed to create config: %v", err)
	}
}
