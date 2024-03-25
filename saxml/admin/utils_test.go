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
	"reflect"
	"testing"

	"saxml/admin/protobuf"
	apb "saxml/protobuf/admin_go_proto_grpc"
)

func TestGetServerMemoryCapacity(t *testing.T) {
	tests := []struct {
		chipType   apb.ModelServer_ChipType
		chipTopo   apb.ModelServer_ChipTopology
		expectedGB int64
	}{
		{
			apb.ModelServer_CHIP_TYPE_TPU_V5I,
			apb.ModelServer_CHIP_TOPOLOGY_1,
			16,
		},
		{
			apb.ModelServer_CHIP_TYPE_TPU_V5,
			apb.ModelServer_CHIP_TOPOLOGY_4X4X8,
			48 * 128,
		},
	}
	for _, tc := range tests {
		t.Run("server mem", func(t *testing.T) {
			ms := protobuf.ModelServer{
				ChipType:     protobuf.ChipType(tc.chipType),
				ChipTopology: protobuf.ChipTopology(tc.chipTopo),
			}
			actual := GetServerMemoryCapacity(&ms)
			if actual != tc.expectedGB*(1<<30) {
				t.Errorf("GetServerMemoryCapacity(%v) err got %d, want %d", ms, actual, tc.expectedGB)
			}
		})
	}
}

func TestGetMemoryRequired(t *testing.T) {
	tests := []struct {
		ram      string
		expected int64
	}{
		{"", -1},
		{"12345678", 12345678},
		{"random blah", -1},
	}
	for _, tc := range tests {
		spec := &apb.Model{}
		if tc.ram != "" {
			spec.Overrides = map[string]string{"ram": tc.ram}
		}
		actual := GetMemoryRequired(spec)
		if actual != tc.expected {
			t.Errorf("GetMemoryRequired(%v) err got %d, want %d", spec, actual, tc.expected)
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
