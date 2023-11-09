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

// A generic skiplist.
//
// Usage, for a skiplist of "int".
//
// Define a comparison routine:
//
//  It should return -ve, 0, or +ve according to whether *a < *b, *a
//  == *b, or *a > *b.
//
// func intcmp(a *int, b *int) (cmp int) {
//        if *a > *b {
//                cmp = 1
//        } else if *a < *b {
//                cmp = -1
//        }
//        return cmp
// }
//
// Create a skiplist:
//
//     sl := skiplist.New[int](intcmp)
//
// Insert an element: (Element values are passed by pointer so reduce
// costs with large elements, but the value (rather than the pointer)
// is copied into the list.)
//
//    value := 17
//    var inserted bool = sl.Insert(&value, false /* don't allow duplicates */)
//
// If the second parameter were "true", the value would always be
// inserted, and Insert() would always return true.
//
// Remove an element:
//
//    value := 17
//    var removed bool = sl.Remove(&value)
//
// The return value will be true if a matching element exists.
//
// Get the number of elements in the list:
//
//   var count int = sl.Count()
//
// An Iterator can walk forward through the list.  Get an iterator
// pointing to the first element, if any:
//
//    var it skiplist.Iterator[int] = sl.First()
//
// Or an iterator pointing to the first item with a given value:
//
//    value := 17
//    var it skiplist.Iterator[int] = sl.Lookup(&value)
//
// Test whether an iterator is nil (which means "end of list") with
// IsNil().  Get an iterator to the next element (if any) with
// Next(). Get a pointer to the value a non-nil iterator points to
// with Value().  So to print all the values in the list:
//
//    for it := sl.First(); !it.IsNil(); it = it.Next() {
//        fmt.PrintLn(*it.Value)
//    }
//
// The client may not use the pointer returned by Value() to change
// the ordering of the value within the list.

package skiplist

import "math/rand"

const max_level = 32 // Maximum number of levels in the skip list.

// Each node in the list has a value of type V, and up to max_level
// pointers to subsequent nodes.  next[0] is the next node in the
// list, or nil.
type node[V interface{}] struct {
	value V
	next  []*node[V]
}

// A skiplist.T[V] is a skiplist containing elements of type V.
type T[V interface{}] struct {
	max_node_size int                  // number of levels in use; [1..max_level]
	cmp           func(a *V, b *V) int // client's comparison routine
	count         int                  // number of element in the list
	head          node[V]              // the "head" of the list; length is always max_level
}

// Create a pointer to a new skiplist.
func New[V interface{}](cmp func(a *V, b *V) int) (sl *T[V]) {
	return &T[V]{
		max_node_size: 1,
		cmp:           cmp,
		count:         0,
		head:          node[V]{next: make([]*node[V], max_level)},
	}
}

// Find the first element in *sl that is at least *pval, and if parents!=nil,
// set the first sl.max_node_size elements of parents[] to point to the
// elements that precede that point at the corresponding level in *sl.
func (sl *T[V]) search(pval *V, parents *[32]*node[V]) *node[V] {
	var n int = sl.max_node_size
	var p *node[V] = &sl.head
	for i := n - 1; i != -1; i-- {
		for pnext := p.next[i]; pnext != nil && sl.cmp(pval, &pnext.value) > 0; pnext = p.next[i] {
			p = pnext
		}
		if parents != nil {
			(*parents)[i] = p
		}
	}
	return p.next[0]
}

// Return an integer in [1, max_level], biased so that the probability of
// returning n is 1/(2**n).
func new_size() (size int) {
	for size = 1; size != max_level && rand.Intn(2) == 0; size++ {
	}
	return size
}

// If allow_dup, insert an element containing a copy of *pval into
// *sl, and return true.  If !allow_dup, determine whether an element
// containing *pval already exists in *sl, and if it does not, insert
// one; return whether an insertion was performed.
func (sl *T[V]) Insert(pval *V, allow_dup bool) (inserted bool) {
	var parents [max_level]*node[V]
	var p *node[V] = sl.search(pval, &parents)

	if allow_dup || p == nil || sl.cmp(pval, &p.value) != 0 {
		var new_node_size int = new_size()
		var new_node *node[V] = &node[V]{value: *pval, next: make([]*node[V], new_node_size)}
		for sl.max_node_size < new_node_size {
			parents[sl.max_node_size] = &sl.head
			sl.max_node_size++
		}
		for i := 0; i != new_node_size; i++ {
			new_node.next[i] = parents[i].next[i]
			parents[i].next[i] = new_node
		}
		sl.count++
		inserted = true
	}
	return inserted
}

// Remove from *sl the first element found that conatins *pval, if any; return
// whether an element was removed.
func (sl *T[V]) Remove(pval *V) (removed bool) {
	var parents [max_level]*node[V]
	var p *node[V] = sl.search(pval, &parents)
	if p != nil && sl.cmp(pval, &p.value) == 0 {
		for i := 0; i != len(p.next); i++ {
			if parents[i].next[i] == p {
				parents[i].next[i] = p.next[i]
			}
			p.next[i] = nil
		}
		for sl.max_node_size != 1 && sl.head.next[sl.max_node_size-1] == nil {
			sl.max_node_size--
		}
		sl.count--
		removed = true
	}
	return removed
}

// Return the number of elements in *sl.
func (sl *T[V]) Count() int {
	return sl.count
}

// An iterator is conceptually just a pointer to a list eleemnt.
type Iterator[V interface{}] struct {
	ptr *node[V]
}

// Return an iterator pointing to the first element in *sl that is
// equal to *pval, or a nil iterator if no such element exists.
func (sl *T[V]) Lookup(pval *V) (it Iterator[V]) {
	it = sl.LowerBound(pval)
	if it.ptr != nil && sl.cmp(pval, &it.ptr.value) != 0 {
		it.ptr = nil
	}
	return it
}

// Return an iterator pointing to the first element in *sl that is at least
// *pval, or a nil iterator if no such element exists.
func (sl *T[V]) LowerBound(pval *V) (it Iterator[V]) {
	it.ptr = sl.search(pval, nil)
	return it
}

// Return an iterator pointing to the first element in *sl,
// or a nil iterator if no such element exists.
func (sl *T[V]) First() (it Iterator[V]) {
	it.ptr = sl.head.next[0]
	return it
}

// Return an iterator pointing to the element after "it", or nil if there is no
// such element.
func (it Iterator[V]) Next() Iterator[V] {
	if it.ptr != nil {
		it.ptr = it.ptr.next[0]
	}
	return it
}

// Return a pointer to the value contained in the element indicated by "it", or
// nil if there is no such element.  The client may not use the returned
// pointer to change the order of the element within the list.
func (it Iterator[V]) Value() (result *V) {
	if it.ptr != nil {
		result = &it.ptr.value
	}
	return result
}

// Return whether the iternator "it" is nil, which indicates "end of list".
func (it Iterator[V]) IsNil() bool {
	return it.ptr == nil
}
