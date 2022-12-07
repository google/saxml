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

// Package cell provides functions to create and query Sax cells.
//
// Sax cells store their metadata in a shared root directory:
//
//	Sax cell path := <root path>/sax/<cell name>
//
// The env.Get().RootDir function returns the root path. It can be on any file
// system that supports file and directory methods defined in the env package.
package cell

import (
	"context"
	"fmt"
	"path/filepath"
	"time"

	log "github.com/golang/glog"
	"saxml/common/errors"
	"saxml/common/naming"
	"saxml/common/platform/env"
)

// Sax returns the directory path containing all Sax cells.
func Sax(ctx context.Context) string {
	return filepath.Join(env.Get().RootDir(ctx), naming.Prefix)
}

// Path returns the directory path to a Sax cell.
func Path(ctx context.Context, saxCell string) (string, error) {
	if _, err := naming.SaxCellToCell(saxCell); err != nil {
		return "", err
	}
	return filepath.Join(env.Get().RootDir(ctx), saxCell), nil
}

// Exists returns nil if and only if saxCell is a valid Sax cell that exists.
func Exists(ctx context.Context, saxCell string) error {
	path, err := Path(ctx, saxCell)
	if err != nil {
		return err
	}

	exist, err := env.Get().DirExists(ctx, path)
	if err != nil {
		return err
	}
	if !exist {
		return fmt.Errorf("%q is not a directory: %w", path, errors.ErrFailedPrecondition)
	}

	return nil
}

// ListAll returns the names of all Sax cells.
func ListAll(ctx context.Context) ([]string, error) {
	return env.Get().ListSubdirs(ctx, Sax(ctx))
}

// Create creates a Sax cell.
func Create(ctx context.Context, saxCell, adminACL string) error {
	path, err := Path(ctx, saxCell)
	if err != nil {
		return err
	}

	log.Infof("Creating directory %q", path)
	if err := env.Get().CreateDir(ctx, path, adminACL); err != nil {
		return err
	}

	// Double-check the invariant.
	for {
		if err := Exists(ctx, saxCell); err == nil {
			break
		}
		time.Sleep(time.Second)
		log.Info("Creating...")
	}
	log.Infof("Created directory %q", path)

	return nil
}
