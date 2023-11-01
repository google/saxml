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

package cloud_test

import (
	"bytes"
	"context"
	"path/filepath"
	"testing"
	"time"

	"saxml/common/platform/cloud" // registers a platform
	"saxml/common/platform/env"
)

func TestFileOps(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	fname := "foo"
	path := filepath.Join(dir, fname)

	if err := env.Get().WriteFile(ctx, path, "", nil); err != nil {
		t.Fatalf("WriteFile got error %v, expect no error", err)
	}
	if exist, err := env.Get().FileExists(ctx, path); err != nil {
		t.Fatalf("FileExists got error %v, expect no error", err)
	} else if !exist {
		t.Fatal("FileExists got false, expect true")
	}
	if bytes, err := env.Get().ReadFile(ctx, path); err != nil {
		t.Fatalf("ReadFile got error %v, expect no error", err)
	} else if len(bytes) != 0 {
		t.Errorf("ReadFile got %v, expect empty content", bytes)
	}
}

func TestWatch(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	fname := "foo"
	path := filepath.Join(dir, fname)

	// Faster watch updates but still long enough for the file modification to finish.
	cloud.SetOptionsForTesting(200 * time.Millisecond)

	// Write an empty file.
	has := []byte("")
	if err := env.Get().WriteFile(ctx, path, "", has); err != nil {
		t.Fatalf("WriteFile got error %v, expect no error", err)
	}

	// Start a watching channel. The first read should be empty.
	ch, err := env.Get().Watch(ctx, path)
	if err != nil {
		t.Fatalf("Watch got error %v, expect no error", err)
	}
	got := <-ch
	if !bytes.Equal(got, has) {
		t.Errorf("Watch got %v, expect %v", got, has)
	}

	// Modify the file.
	has = []byte("bar")
	if err := env.Get().WriteFile(ctx, path, "", has); err != nil {
		t.Fatalf("WriteFile got error %v, expect no error", err)
	}

	// The next read should have the file update.
	got = <-ch
	if !bytes.Equal(got, has) {
		t.Errorf("Watch got %v, expect %v", got, has)
	}
}
