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

package queue

import (
	"math/rand"
	"testing"
	"time"
)

func TestBasic(t *testing.T) {
	n := 1000
	q := New[*int]()
	done := make(chan bool)

	go func() {
		for i := 0; ; i++ {
			v := q.Get()
			if v == nil {
				break
			}
			t.Logf("Get %v", *v)
			if *v != i {
				t.Errorf("Get() = %v, want %v", *v, i)
			}
		}
		done <- true
	}()

	for i := 0; i < n; i++ {
		t.Logf("Put %v", i)
		q.Put(&i)
		time.Sleep(time.Duration(rand.Intn(10)+1) * time.Microsecond)
	}

	q.Put(nil)
	<-done
}
