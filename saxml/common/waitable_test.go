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

package waitable_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"saxml/common/errors"
	"saxml/common/waitable"
)

func TestBasic(t *testing.T) {
	w := waitable.New()
	go func() {
		w.Set(100)
	}()
	w.Wait(context.Background(), 100)
}

func TestManyWaiters(t *testing.T) {
	n := 100
	w := waitable.New()
	w.Set(n / 4)

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(threshold int) {
			defer wg.Done()
			err := w.Wait(context.Background(), threshold)
			if err != nil {
				t.Errorf("err = %v, want nil", err)
			}
			t.Logf("threshold %d %v", threshold, err)
		}(i)
	}
	for i := n / 4; i < n; i++ {
		t.Logf("value = %v", w.Value())
		w.Add(1)
		time.Sleep(time.Microsecond)
	}
	wg.Wait()
}

func TestManyWaitersCancelled(t *testing.T) {
	n := 100
	w := waitable.New()
	w.Set(n / 4)
	ctx, cancel := context.WithCancel(context.Background())

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(threshold int) {
			defer wg.Done()
			err := w.Wait(ctx, threshold)
			if threshold <= n/2 && err != nil {
				t.Errorf("threshold = %d err = %v, want nil", threshold, err)
			}
			if threshold > n/2 && err != context.Canceled {
				t.Errorf("threshold = %d err = %v, want %v", threshold, err, context.Canceled)
			}
			t.Logf("threshold %d err %v", threshold, err)
		}(i)
	}
	for i := n / 4; i < n/2; i++ {
		w.Add(1)
		time.Sleep(time.Microsecond)
	}
	time.Sleep(time.Second)
	// If cancel() is not called, wg.Wait() would hang.
	cancel()
	wg.Wait()
}

func TestClose(t *testing.T) {
	n := 100
	w := waitable.New()
	for i := 0; i < n; i++ {
		go func(i int) {
			err := w.Wait(context.Background(), n)
			if err != errors.ErrCanceled {
				t.Errorf("error = %v, want %v", err, errors.ErrCanceled)
			}
		}(i)
	}
	w.Close()
}
