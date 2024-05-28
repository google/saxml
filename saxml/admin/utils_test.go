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

package utils

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
	"testing"

	"saxml/admin/protobuf"
	apb "saxml/protobuf/admin_go_proto_grpc"
)

func TestGetServerMemoryCapacity(t *testing.T) {
	tests := []struct {
		chipType      apb.ModelServer_ChipType
		chipTopo      apb.ModelServer_ChipTopology
		expectedGB    int64
		expectedCores int64
	}{
		{
			apb.ModelServer_CHIP_TYPE_TPU_V5I,
			apb.ModelServer_CHIP_TOPOLOGY_1,
			16, 1,
		},
		{
			apb.ModelServer_CHIP_TYPE_TPU_V5,
			apb.ModelServer_CHIP_TOPOLOGY_4X4X8,
			48,
			128,
		},
	}
	for _, tc := range tests {
		t.Run("server mem", func(t *testing.T) {
			ms := protobuf.ModelServer{
				ChipType:     protobuf.ChipType(tc.chipType),
				ChipTopology: protobuf.ChipTopology(tc.chipTopo),
			}
			actual := GetServerMemoryInfo(&ms)
			if actual.BytesPerCore != tc.expectedGB*(1<<30) || actual.Cores != tc.expectedCores {
				t.Errorf("GetServerMemoryInfo(%v) err got %d x %d, want %d x %d",
					ms, actual.BytesPerCore, actual.Cores, tc.expectedGB, tc.expectedCores)
			}
		})
	}
}

func str(ramPerCores map[int64]int64) string {
	var cores []int64
	for k := range ramPerCores {
		cores = append(cores, k)
	}
	sort.Slice(cores, func(i, j int) bool { return cores[i] < cores[j] })

	b := new(strings.Builder)
	for _, k := range cores {
		fmt.Fprintf(b, "%d:%d,", k, ramPerCores[k])
	}
	return b.String()
}

func TestGetMemoryRequired(t *testing.T) {
	tests := []struct {
		ram      map[string]string
		expected map[int64]int64
	}{
		{nil, nil},
		{map[string]string{"ram": "random blah"}, nil},
		{map[string]string{"ram": "12345678"}, map[int64]int64{1: 12345678}},
		{map[string]string{"ram": "12345678", "ram8": "5678"}, map[int64]int64{1: 12345678, 8: 5678}},
		{map[string]string{"ram": "12345678", "ram8": "random blah"}, map[int64]int64{1: 12345678}},
	}
	for _, tc := range tests {
		spec := &apb.Model{}
		spec.Overrides = tc.ram
		actual := str(GetMemoryRequired(spec))
		expected := str(tc.expected)
		if actual != expected {
			t.Errorf("GetMemoryRequired(%v) err got %s, want %s", spec, actual, expected)
		}
	}
}

func TestGetConstraints(t *testing.T) {
	tests := []struct {
		constraints string
		expected    []string
	}{
		{"", nil},
		{"run=abc,topo=2x2", []string{"run=abc", "topo=2x2"}},
	}
	for _, tc := range tests {
		spec := &apb.Model{}
		if tc.constraints != "" {
			spec.Overrides = map[string]string{"constraints": tc.constraints}
		}
		actual := GetConstraints(spec)
		if !reflect.DeepEqual(actual, tc.expected) {
			t.Errorf("GetConstraints(%v) err got %v, want %v", spec, actual, tc.expected)
		}
	}
}
