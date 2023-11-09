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

package skiplist

import (
	"fmt"
	"math/rand"
	"testing"
)

func intcmp(a *int, b *int) (cmp int) {
	if *a > *b {
		cmp = 1
	} else if *a < *b {
		cmp = -1
	}
	return cmp
}

func TestBasic(t *testing.T) {
	var sl *T[int] = New[int](intcmp)
	var a [1024]int

	rand.Seed(1)

	for i := 0; i != 1000000; i++ {
		if i%10000 == 0 {
			fmt.Printf("%d %d\n", i, sl.Count())
		}
		var op int = rand.Intn(8)
		var value int = rand.Intn(len(a))
		if op == 0 || op == 1 { // insert, no_dups
			var inserted bool = sl.Insert(&value, false)
			if inserted != (a[value] == 0) {
				t.Fatalf("bad no dup insert")
			}
			if inserted {
				a[value]++
			}
		} else if op == 2 { // insert, dups
			var inserted bool = sl.Insert(&value, true)
			if !inserted {
				t.Fatalf("bad dup insert")
			}
			a[value]++
		} else if op == 3 || op == 4 || op == 5 { // remove
			var removed bool = sl.Remove(&value)
			if removed != (a[value] != 0) {
				t.Fatalf("bad remove")
			}
			if removed {
				a[value]--
			}
		} else if op == 6 { // lookup
			var result Iterator[int] = sl.Lookup(&value)
			if result.IsNil() != (a[value] == 0) {
				t.Fatalf("bad lookup")
			}
			if !result.IsNil() && *result.Value() != value {
				t.Fatalf("bad lookup2")
			}
		} else if op == 7 { // lower bound
			var result Iterator[int] = sl.LowerBound(&value)
			if result.IsNil() {
				for i := value; i < len(a); i++ {
					if a[i] != 0 {
						t.Fatalf("%d exists and is at least %d", i, value)
					}
				}
			} else {
				for i := value; i < *result.Value(); i++ {
					if a[i] != 0 {
						t.Fatalf("%d exists and is at least %d but smaller than %d", i, value, result.Value())
					}
				}
			}

		}
		it := sl.First()
		var totalCount int
		for j := 0; j != len(a); j++ {
			totalCount += a[j]
			if !it.IsNil() && *it.Value() < j {
				t.Fatalf("extra elements")
			}
			var count int
			for count = a[j]; count != 0 && !it.IsNil() && *it.Value() == j; count-- {
				it = it.Next()
			}
			if count != 0 {
				t.Fatalf("not enough elements")
			} else if !it.IsNil() && *it.Value() == j {
				t.Fatalf("too many elements")
			}
		}
		if totalCount != sl.Count() {
			t.Fatalf("bad count")
		}
	}
}
