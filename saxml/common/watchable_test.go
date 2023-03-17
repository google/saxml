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

package watchable_test

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"saxml/common/watchable"
)

func applyNextState(ctx context.Context, w *watchable.Watchable, start int32, state *watchable.DataSet) (int32, *watchable.DataSet, error) {
	result, err := w.Watch(ctx, start)
	if err != nil {
		return 0, watchable.NewDataSet(), err
	}
	if result.Data != nil {
		state = result.Data
	}
	state.Apply(result.Log)
	return result.Next, state, nil
}

func itoa(i int) string {
	return fmt.Sprintf("%08d", i)
}

func create(num int) []string {
	var ret []string
	for i := 0; i < num; i++ {
		ret = append(ret, itoa(i))
	}
	return ret
}

func equal(num int, data *watchable.DataSet) bool {
	if data.Size() != num {
		return false
	}
	for i := 0; i < num; i++ {
		if !data.Exist(itoa(i)) {
			return false
		}
	}
	return true
}

// Tests basic operations of a watchable.
func TestBasic(t *testing.T) {
	// Setup a watchable state with 'num' entries added.
	w := watchable.New()
	num := 1000
	for _, val := range create(num) {
		w.Add(val)
	}

	// Assumes the caller has caught up the mutation up to start-1 and
	// asks for mutations [start, ...).
	ctx := context.Background()
	for start := 0; start < num; start++ {
		state := watchable.NewDataSet() // Empty
		for i := 0; i < start; i++ {
			state.Add(itoa(i))
		}
		next, state, err := applyNextState(ctx, w, int32(start), state)
		if err != nil {
			t.Errorf("applyNextState(%d) = %v, want nil", start, err)
		}
		if !equal(num, state) {
			t.Errorf("applyNextState(%d).state = %v, want %v", start, state, create(num))
		}
		if next != int32(num) {
			t.Errorf("applyNextState(%d) next = %v, want %v", start, next, num)
		}
	}

	// If the caller pass in a seqno in the future, returns the seqno
	// frontier immediately.
	result, err := w.Watch(ctx, int32(num+1))
	if err != nil {
		t.Errorf("w.Watch() got an error(%v), want no err", err)
	}
	if result.Next != int32(num) {
		t.Errorf("result.Next = %v, want %v", result.Next, num)
	}

	ctx, cancel := context.WithTimeout(ctx, 10*time.Millisecond)
	defer cancel()
	_, err = w.Watch(ctx, int32(num))
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("Watch(%d) err %v, wanted %v", num, err, context.DeadlineExceeded)
	}

	w.Close()
	ctx = context.Background()
	result, err = w.Watch(ctx, int32(num))
	if err != nil {
		t.Errorf("w.Watch() got an error(%v), want no err", err)
	}
	if result.Data == nil || result.Data.Size() != 0 {
		t.Errorf("result.Data = %v, want empty", result.Data)
	}
	if result.Log != nil {
		t.Errorf("result.Log = %v, want nil", result.Log)
	}
	if result.Next != 0 {
		t.Errorf("result.Next = %v, want 0", result.Next)
	}
}

func TestCancelWatch(t *testing.T) {
	w := watchable.New()
	ctx, cancel := context.WithCancel(context.Background())
	go w.Watch(ctx, 0)
	cancel()
	w.Add("foo")
	time.Sleep(time.Millisecond)
	w.Add("bar")
}

func TestMultipleReplicas(t *testing.T) {
	w := watchable.New()
	ch := make(chan *watchable.DataSet)

	// Setup a few replicas (consumers) watching mutations of w.  They
	// stop when see a special string "DONE".
	ctx := context.Background()
	numCopies := 100
	for i := 0; i < numCopies; i++ {
		go func() {
			data := watchable.NewDataSet()
			var token int32 = 0
			for {
				result, err := w.Watch(ctx, token)
				if err != nil {
					t.Errorf("w.Watch(%d) get err %v, want nil", token, err)
					break
				}
				if result.Data != nil {
					data = result.Data
				}
				data.Apply(result.Log)
				token = result.Next
				if data.Exist("DONE") {
					break
				}
			}
			ch <- data
		}()
	}

	// A single producer generates changes to w.
	wantData := watchable.NewDataSet()
	numMuts := 2000
	for i := 0; i < numMuts; i++ {
		w.Add(itoa(i))
		wantData.Add(itoa(i))
		if i%10 == 0 {
			w.Del(itoa(i - i/10))
			wantData.Del(itoa(i - i/10))
		}
	}

	// Sends DONE.
	w.Add("DONE")
	wantData.Add("DONE")

	want := wantData.ToList()
	sort.Strings(want)

	// Waits for all replicas sending back their states.
	for i := 0; i < numCopies; i++ {
		data := <-ch
		got := data.ToList()
		sort.Strings(got)
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("Watched client state mismatch (-want +got)\n%s", diff)
		}
	}
}

func TestMultipleReplicasOnClose(t *testing.T) {
	w := watchable.New()

	// Setup a few replicas (consumers) watching mutations of w.
	// After seeing a special string "DONE", they expect that w is
	// closed and The dataset is reset.
	ctx := context.Background()
	var allSeenDone sync.WaitGroup
	var allSeenClose sync.WaitGroup
	numCopies := 100
	for i := 0; i < numCopies; i++ {
		allSeenDone.Add(1)
		allSeenClose.Add(1)
		go func(idx int) {
			defer allSeenClose.Done()
			seenDone := false
			data := watchable.NewDataSet()
			var token int32 = 0
			for {
				result, err := w.Watch(ctx, token)
				if err != nil {
					t.Errorf("w.Watch(%d) get err %v, want nil", token, err)
					break
				}
				if result.Data != nil {
					data = result.Data
				}
				data.Apply(result.Log)
				token = result.Next
				if !seenDone {
					if data.Exist("DONE") {
						seenDone = true
						allSeenDone.Done()
					}
				} else if data.Size() > 0 {
					t.Errorf("w.Watch(%d) is closed, but not reset %v", token, data.ToList())
				} else {
					// We are done
					break
				}
			}
		}(i)
	}

	// A single producer generates changes to w.
	numMuts := 100
	for i := 0; i < numMuts; i++ {
		w.Add(itoa(i))
		if i%10 == 0 {
			w.Del(itoa(i - i/10))
		}
	}

	// Sends DONE.
	w.Add("DONE")
	allSeenDone.Wait()

	// All consumers should exit.
	w.Close()
	allSeenClose.Wait()
}
