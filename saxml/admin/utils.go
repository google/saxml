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

// Package utils provides helper routines for the admin server implementation.
package utils

import (
	"strconv"
	"strings"

	log "github.com/golang/glog"
	"saxml/admin/protobuf"
	apb "saxml/protobuf/admin_go_proto_grpc"
)

// GetServerMemoryCapacity estimates the model server's memory capacity (in bytes).
func GetServerMemoryCapacity(spec *protobuf.ModelServer) int64 {
	var memGBPerCore int64
	switch apb.ModelServer_ChipType(spec.ChipType) {
	case apb.ModelServer_CHIP_TYPE_UNKNOWN:
		memGBPerCore = 8 // Assumption
	case apb.ModelServer_CHIP_TYPE_TPU_V2:
		memGBPerCore = 8
	case apb.ModelServer_CHIP_TYPE_TPU_V3:
		memGBPerCore = 16
	case apb.ModelServer_CHIP_TYPE_TPU_V4:
		memGBPerCore = 16
	case apb.ModelServer_CHIP_TYPE_TPU_V4I:
		memGBPerCore = 8
   // some unhandled cases
	case apb.ModelServer_CHIP_TYPE_TPU_V5E:
		memGBPerCore = 16
	case apb.ModelServer_CHIP_TYPE_GPU_P100:
		memGBPerCore = 16
	case apb.ModelServer_CHIP_TYPE_GPU_V100:
		memGBPerCore = 16
	case apb.ModelServer_CHIP_TYPE_GPU_T4:
		memGBPerCore = 16
	case apb.ModelServer_CHIP_TYPE_GPU_A100:
		memGBPerCore = 40
	case apb.ModelServer_CHIP_TYPE_GPU_H100:
		memGBPerCore = 80
	case apb.ModelServer_CHIP_TYPE_GPU_L4:
		memGBPerCore = 24
	default:
		memGBPerCore = 64 // Assumption
	}
	var numCores int64
	switch apb.ModelServer_ChipTopology(spec.ChipTopology) {
	case apb.ModelServer_CHIP_TOPOLOGY_1:
		numCores = 1
	case apb.ModelServer_CHIP_TOPOLOGY_2:
		numCores = 2
	case apb.ModelServer_CHIP_TOPOLOGY_4:
		numCores = 4
	case apb.ModelServer_CHIP_TOPOLOGY_8:
		numCores = 8
	case apb.ModelServer_CHIP_TOPOLOGY_16:
		numCores = 16
	case apb.ModelServer_CHIP_TOPOLOGY_1X1:
		numCores = 1
	case apb.ModelServer_CHIP_TOPOLOGY_2X2:
		numCores = 4
	case apb.ModelServer_CHIP_TOPOLOGY_2X4:
		numCores = 8
	case apb.ModelServer_CHIP_TOPOLOGY_4X2:
		numCores = 8
	case apb.ModelServer_CHIP_TOPOLOGY_4X4:
		numCores = 16
	case apb.ModelServer_CHIP_TOPOLOGY_4X8:
		numCores = 32
	case apb.ModelServer_CHIP_TOPOLOGY_8X8:
		numCores = 64
	case apb.ModelServer_CHIP_TOPOLOGY_8X16:
		numCores = 128
	case apb.ModelServer_CHIP_TOPOLOGY_16X16:
		numCores = 256
	case apb.ModelServer_CHIP_TOPOLOGY_16X32:
		numCores = 512
	case apb.ModelServer_CHIP_TOPOLOGY_32X32:
		numCores = 1024
	case apb.ModelServer_CHIP_TOPOLOGY_1X1X1:
		numCores = 1
	case apb.ModelServer_CHIP_TOPOLOGY_1X2X1:
		numCores = 2
	case apb.ModelServer_CHIP_TOPOLOGY_2X2X1:
		numCores = 4
	case apb.ModelServer_CHIP_TOPOLOGY_2X2X2:
		numCores = 8
	case apb.ModelServer_CHIP_TOPOLOGY_2X2X4:
		numCores = 16
	case apb.ModelServer_CHIP_TOPOLOGY_2X4X4:
		numCores = 32
	case apb.ModelServer_CHIP_TOPOLOGY_4X4X4:
		numCores = 64
	case apb.ModelServer_CHIP_TOPOLOGY_4X4X8:
		numCores = 128
	case apb.ModelServer_CHIP_TOPOLOGY_4X4X16:
		numCores = 256
	case apb.ModelServer_CHIP_TOPOLOGY_4X8X8:
		numCores = 256
	case apb.ModelServer_CHIP_TOPOLOGY_8X8X12:
		numCores = 768
	case apb.ModelServer_CHIP_TOPOLOGY_8X8X8:
		numCores = 512
	default:
		numCores = 1 // Assumption
	}
	return memGBPerCore * numCores * (1 << 30)
}

// GetMemoryRequired estimates the model's memory usage based on
// spec.overrided["ram"]. Returns <0 if the estimation is not
// feasible.
func GetMemoryRequired(spec *apb.Model) int64 {
	// TODO: Maybe define a field in apb.Model. But for now, we expect
	// a field in the override provides a hint.
	found, ok := spec.GetOverrides()["ram"]
	if !ok {
		return -1
	}
	value, err := strconv.ParseInt(found, 10, 64)
	if err != nil {
		log.Errorf("Invalid number found in \"ram\": %s", found)
		return -1
	}
	return value
}

// GetConstraints extracts constraints from the model specification's overrides.
func GetConstraints(spec *apb.Model) []string {
	found, ok := spec.GetOverrides()["constraints"]
	if !ok {
		return nil
	}
	return strings.Split(found, ",")
}
