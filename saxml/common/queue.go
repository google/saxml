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

// Package queue is a generic FIFO queue.
package queue

// Usage
//
// Create a queue:
//   q := queue.New[int]()
//
// Adds an element
//   v := 100
//   q.Add(&v)
//
// Consumes an element, blocks until non-empty.
//   v := q.Get()

import (
	"sync"
)

// T is a queue containing elements of type V.
type T[V any] struct {
	mu       sync.Mutex
	notEmpty *sync.Cond

	// head and tail forms q deque.
	head []V
	tail []V
}

// New creates a pointer to a new FIFO queue.
func New[V any]() *T[V] {
	q := &T[V]{}
	q.notEmpty = sync.NewCond(&q.mu)
	return q
}

// Put appends one element to the tail of the queue.
func (q *T[V]) Put(v V) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.tail = append(q.tail, v)
	q.notEmpty.Broadcast()
}

// Size returns the number of elements in the queue.
func (q *T[V]) Size() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.head) + len(q.tail)
}

// Get returns the element at the head of the queue. Blocks until the
// queue is not empty.
func (q *T[V]) Get() V {
	q.mu.Lock()
	defer q.mu.Unlock()

	// Blocks until the queue is not empty.
	for len(q.head)+len(q.tail) == 0 {
		q.notEmpty.Wait()
	}

	headSize := len(q.head)
	if headSize == 0 {
		// If head is empty, moves items from tail to head.
		for i := len(q.tail) - 1; i >= 0; i-- {
			q.head = append(q.head, q.tail[i])
		}
		q.tail = q.tail[:0]
		headSize = len(q.head)
	}

	// Pops one item from thead.
	item := q.head[headSize-1]
	q.head = q.head[:headSize-1]
	return item
}
