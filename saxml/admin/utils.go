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

// ServerMemoryInfo describes the memory capabilities of a server.
type ServerMemoryInfo struct {
	BytesPerCore int64
	Cores        int64
}

// GetServerMemoryInfo estimates the model server's memory capacity (in bytes),
// and provides the number of accelerator cores on the server.
func GetServerMemoryInfo(spec *protobuf.ModelServer) ServerMemoryInfo {
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
	case apb.ModelServer_CHIP_TYPE_TPU_V6E:
		memGBPerCore = 32
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
	var cores int64
	switch apb.ModelServer_ChipTopology(spec.ChipTopology) {
	case apb.ModelServer_CHIP_TOPOLOGY_UNKNOWN:
		cores = 1 // Assumption
	case apb.ModelServer_CHIP_TOPOLOGY_1:
		cores = 1
	case apb.ModelServer_CHIP_TOPOLOGY_2:
		cores = 2
	case apb.ModelServer_CHIP_TOPOLOGY_4:
		cores = 4
	case apb.ModelServer_CHIP_TOPOLOGY_8:
		cores = 8
	case apb.ModelServer_CHIP_TOPOLOGY_16:
		cores = 16
	case apb.ModelServer_CHIP_TOPOLOGY_1X1:
		cores = 1
	case apb.ModelServer_CHIP_TOPOLOGY_2X2:
		cores = 4
	case apb.ModelServer_CHIP_TOPOLOGY_2X4:
		cores = 8
	case apb.ModelServer_CHIP_TOPOLOGY_4X2:
		cores = 8
	case apb.ModelServer_CHIP_TOPOLOGY_4X4:
		cores = 16
	case apb.ModelServer_CHIP_TOPOLOGY_4X8:
		cores = 32
	case apb.ModelServer_CHIP_TOPOLOGY_8X8:
		cores = 64
	case apb.ModelServer_CHIP_TOPOLOGY_8X16:
		cores = 128
	case apb.ModelServer_CHIP_TOPOLOGY_16X16:
		cores = 256
	case apb.ModelServer_CHIP_TOPOLOGY_16X32:
		cores = 512
	case apb.ModelServer_CHIP_TOPOLOGY_32X32:
		cores = 1024
	case apb.ModelServer_CHIP_TOPOLOGY_1X1X1:
		cores = 1
	case apb.ModelServer_CHIP_TOPOLOGY_1X2X1:
		cores = 2
	case apb.ModelServer_CHIP_TOPOLOGY_2X2X1:
		cores = 4
	case apb.ModelServer_CHIP_TOPOLOGY_2X2X2:
		cores = 8
	case apb.ModelServer_CHIP_TOPOLOGY_2X2X4:
		cores = 16
	case apb.ModelServer_CHIP_TOPOLOGY_2X4X4:
		cores = 32
	case apb.ModelServer_CHIP_TOPOLOGY_4X4X4:
		cores = 64
	case apb.ModelServer_CHIP_TOPOLOGY_4X4X8:
		cores = 128
	case apb.ModelServer_CHIP_TOPOLOGY_4X4X16:
		cores = 256
	case apb.ModelServer_CHIP_TOPOLOGY_4X8X8:
		cores = 256
	case apb.ModelServer_CHIP_TOPOLOGY_8X8X12:
		cores = 768
	case apb.ModelServer_CHIP_TOPOLOGY_8X8X8:
		cores = 512
	default:
		cores = 1 // Assumption
	}
	return ServerMemoryInfo{BytesPerCore: memGBPerCore * (1 << 30), Cores: cores}
}

// GetMemoryRequired estimates the model's memory usage based on
// spec.overrided["ramN"]. Returns nil if the estimation is not
// feasible.
// Memory usage is given as map[cores]bytesRequired.
func GetMemoryRequired(spec *apb.Model) map[int64]int64 {
	// TODO: Maybe define a field in apb.Model. But for now, we expect
	// a field in the override provides a hint.

	var bytesPerCores = make(map[int64]int64)
	for key, value := range spec.GetOverrides() {
		if !strings.HasPrefix(key, "ram") {
			continue
		}

		cores := int64(1)
		if len(key) > 3 {
			var err error
			cores, err = strconv.ParseInt(key[3:], 10, 64)
			if err != nil {
				log.Errorf("Failed to parse number of cores from key %s: %v", key, err)
				continue
			}
		}

		bytes, err := strconv.ParseInt(value, 10, 64)
		if err != nil {
			log.Errorf("Failed to parse bytes from value %s: %v", value, err)
			continue
		}

		bytesPerCores[cores] = bytes
	}

	return bytesPerCores
}

// GetConstraints extracts constraints from the model specification's overrides.
func GetConstraints(spec *apb.Model) []string {
	found, ok := spec.GetOverrides()["constraints"]
	if !ok {
		return nil
	}
	return strings.Split(found, ",")
}
