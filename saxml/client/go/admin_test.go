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

package saxadmin

import (
	"fmt"
	"math"
	"testing"

	"saxml/common/errors"
)

func TestEmpty(t *testing.T) {
	ar := newAddrReplica("/sax/foo/bar")
	if _, err := ar.Pick(0); !errors.ServerShouldRetry(err) {
		t.Errorf("Pick(0) = %v, want %v", err, errors.ErrUnavailable)
	}
}

func TestErr(t *testing.T) {
	ar := newAddrReplica("/sax/foo/bar")
	ar.add("1.2.3.4:5555")
	ar.setError(errors.ErrNotFound)
	if _, err := ar.Pick(0); err != errors.ErrNotFound {
		t.Errorf("Pick(0) = %v, want %v", err, errors.ErrNotFound)
	}
}

func TestHash(t *testing.T) {
	ar0 := newAddrReplica("/sax/foo/bar")
	ar1 := newAddrReplica("/sax/foo/bar")

	h0 := ar0.hashUint64(0)

	// Hash should be deterministic.
	if h1 := ar0.hashUint64(0); h0 != h1 {
		t.Errorf("Expect %x == %x", h0, h1)
	}

	// Hash should be randomly different among clients.
	if h1 := ar1.hashUint64(0); h0 == h1 {
		t.Errorf("Expect %x != %x", h0, h1)
	}
}

func TestLoadBalancing(t *testing.T) {
	// Assume there are n servers and m clients.
	// Each client has affinity of l.
	m, n, l := 72, 8192, 7

	addrs := make([]string, m)
	counts := make(map[string]int)
	for i := 0; i < m; i++ {
		addr := fmt.Sprintf("%08d", i)
		addrs[i] = addr
		counts[addr] = 0
	}

	for i := 0; i < n; i++ {
		ar := newAddrReplica("/sax/foo/bar")

		// Half of the addrs are added in bulk. The other half are
		// added one-by-one.
		ar.reset(addrs[:m/2])
		for _, addr := range addrs[m/2:] {
			ar.add(addr)
		}

		for j := 0; j < l; j++ {
			if addr, err := ar.Pick(uint64(j)); err != nil {
				t.Errorf("Pick error %v", err)
			} else {
				counts[addr]++
			}
		}

		// Makes sure all addrs are deleted.
		for _, addr := range addrs {
			ar.del(addr)
		}
		if ar.hash.Count() != 0 || len(ar.addr) != 0 {
			t.Errorf("Unexpected: ar is not empty")
		}
	}

	// Computes the standard deviation of server load.
	mean := float64(n*l) / float64(m)
	var sumD2 float64 = 0.
	for _, addr := range addrs {
		t.Logf("%010s %4d\n", addr, counts[addr])
		d := float64(counts[addr]) - mean
		sumD2 += d * d
	}
	std := math.Sqrt(sumD2 / float64(m))
	t.Logf("load mean %6.2f std: %6.2f\n", mean, std)
	if std > mean*0.05 {
		t.Errorf("Too big varaince of the load")
	}
}
