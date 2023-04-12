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

// Package waitable provides an abstraction Waitable.
//
// Waitable maintains an integral counter. Multiple threads can set,
// increment or decrement the counter concurrently. A client can
// also wait for the counter to be larger than a given threshold.
//
// E.g.,
//
//		w := w.New()
//		w.Set(100)
//		w.Add(100)
//		w.Add(-100)
//
//		// Expect the value to be 100
//		assert w.Value() == 100
//		go func() {
//		   time.Sleep(time.Second)
//		   w.Add(100)
//		 }
//
//		// Block until the counter becomes larger than 150.
//		err := w.Wait(ctx, 150)
//
//	        w.Close()
package waitable

import (
	"context"
	"sync"

	"saxml/common/errors"
)

// Waitable keeps track of an integral counter and allows the client
// to wait for its value exceeding a threshold.
type Waitable struct {
	mu      sync.Mutex
	counter int
	waiter  map[*waiter]struct{}
}

// New creates a Waitable instance.
func New() *Waitable {
	return &Waitable{waiter: make(map[*waiter]struct{})}
}

type waiter struct {
	threshold int
	ready     chan error
}

// Value returns a Waitable's current counter value.
func (w *Waitable) Value() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.counter
}

// Set sets a Waitable's counter value.
func (w *Waitable) Set(val int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.counter = val
	w.updateWaitersLocked()
}

// Add increases, or decreases if negative, the counter value by delta.
func (w *Waitable) Add(delta int) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.counter += delta
	w.updateWaitersLocked()
}

func (w *Waitable) updateWaitersLocked() {
	for r := range w.waiter {
		if w.counter >= r.threshold {
			r.ready <- nil
			delete(w.waiter, r)
		}
	}
}

// Wait blocks the caller until either the ctx is Done or the
// waitable's counter is larger than threshold.
func (w *Waitable) Wait(ctx context.Context, threshold int) error {
	w.mu.Lock()
	if w.counter >= threshold {
		w.mu.Unlock()
		return nil
	}
	r := &waiter{threshold: threshold, ready: make(chan error)}
	w.waiter[r] = struct{}{}
	w.mu.Unlock()

	var err error
	select {
	case err = <-r.ready:
	case <-ctx.Done():
		err = ctx.Err()
		w.mu.Lock()
		delete(w.waiter, r)
		w.mu.Unlock()
	}

	return err
}

// Close shutdowns the waitable properly by cancelling all pending
// Wait() calls.
func (w *Waitable) Close() {
	w.mu.Lock()
	defer w.mu.Unlock()
	for r := range w.waiter {
		r.ready <- errors.ErrCanceled
	}
}
