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

// Package protobuf provides Go definitions for protos used by the admin server.
package protobuf

import (
	"fmt"
	"strings"

	"saxml/common/errors"

	apb "saxml/protobuf/admin_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
)

// ModelStatus represents the status of a loaded model on a model server.
type ModelStatus int

// ModelStatus value definitions.
const (
	None      ModelStatus = iota // Unused: unloaded models are simply removed from responses.
	Loading                      // This model is being loaded and can't serve yet.
	Loaded                       // This model is loaded and ready to serve.
	Failed                       // This model failed to load or unload.
	Unloading                    // This model is being unloaded and can't serve anymore.
)

// NewModelStatus converts a proto value to a ModelStatus value.
func NewModelStatus(s cpb.ModelStatus) (ModelStatus, error) {
	switch s {
	case cpb.ModelStatus_NONE:
		return None, nil
	case cpb.ModelStatus_LOADING:
		return Loading, nil
	case cpb.ModelStatus_LOADED:
		return Loaded, nil
	case cpb.ModelStatus_FAILED:
		return Failed, nil
	case cpb.ModelStatus_UNLOADING:
		return Unloading, nil
	default:
		return None, fmt.Errorf("invalid ModelStatus proto value %v: %w", s, errors.ErrInvalidArgument)
	}
}

// ToProto converts a ModelStatus value to a proto value.
func (s ModelStatus) ToProto() (cpb.ModelStatus, error) {
	switch s {
	case None:
		return cpb.ModelStatus_NONE, nil
	case Loading:
		return cpb.ModelStatus_LOADING, nil
	case Loaded:
		return cpb.ModelStatus_LOADED, nil
	case Failed:
		return cpb.ModelStatus_FAILED, nil
	case Unloading:
		return cpb.ModelStatus_UNLOADING, nil
	default:
		return cpb.ModelStatus_NONE, fmt.Errorf("invalid ModelStatus value %v: %w", s, errors.ErrInvalidArgument)
	}
}

// String returns a human-readable string for debugging.
func (s ModelStatus) String() string {
	switch s {
	case None:
		return "None"
	case Loading:
		return "Loading"
	case Loaded:
		return "Loaded"
	case Failed:
		return "Failed"
	case Unloading:
		return "Unloading"
	default:
		return "Invalid"
	}
}

// ChipType represents the type of chips in the model server.
type ChipType apb.ModelServer_ChipType

// ChipTopology represents the topology of chips in the model server.
type ChipTopology apb.ModelServer_ChipTopology

// ModelServer represents the specifications of a model server.
type ModelServer struct {
	ChipType           ChipType
	ChipTopology       ChipTopology
	ServableModelPaths []string
}

// NewModelServer converts a proto value to a ModelServer value.
func NewModelServer(m *apb.ModelServer) *ModelServer {
	paths := make([]string, len(m.GetServableModelPaths()))
	copy(paths, m.GetServableModelPaths())
	return &ModelServer{
		ChipType:           ChipType(m.GetChipType()),
		ChipTopology:       ChipTopology(m.GetChipTopology()),
		ServableModelPaths: paths,
	}
}

// ToProto converts a ModelServer value to a proto value.
func (m *ModelServer) ToProto() *apb.ModelServer {
	paths := make([]string, len(m.ServableModelPaths))
	copy(paths, m.ServableModelPaths)
	return &apb.ModelServer{
		ChipType:           apb.ModelServer_ChipType(m.ChipType),
		ChipTopology:       apb.ModelServer_ChipTopology(m.ChipTopology),
		ServableModelPaths: paths,
	}
}

// Equal compares a ModelServer value with a proto value.
func (m *ModelServer) Equal(other *apb.ModelServer) bool {
	if m.ChipType != ChipType(other.GetChipType()) {
		return false
	}
	if m.ChipTopology != ChipTopology(other.GetChipTopology()) {
		return false
	}
	if len(m.ServableModelPaths) != len(other.GetServableModelPaths()) {
		return false
	}
	for i, p := range m.ServableModelPaths {
		if p != other.GetServableModelPaths()[i] {
			return false
		}
	}
	return true
}

// String returns a human-readable string for debugging.
func (m *ModelServer) String() string {
	return fmt.Sprintf("Model server with chip type: %v, chip topology: %v, servable model paths: %v",
		m.ChipType, m.ChipTopology, strings.Join(m.ServableModelPaths, ", "))
}
