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

// Package validator groups together gRPC request validation functions.
//
// Name and ID validation functions are in the naming package.
package validator

import (
	"fmt"
	"regexp"
	"strings"

	"saxml/common/errors"
	"saxml/common/naming"
	"saxml/common/platform/env"

	pb "saxml/protobuf/admin_go_proto_grpc"
)

const (
	// Valid FS root pattern.
	fsRootPattern = "^[a-zA-Z0-9_./-]+$"

	// Valid model path pattern.
	modelPathPattern = "^[a-zA-Z]+[a-zA-Z0-9_.-]*$"

	// Valid checkpoint pattern.
	checkpointPattern = `^(/|\d|\w|_|\.|-|%|:|=|\|)+$`
)

var (
	validFSRoot     = regexp.MustCompile(fsRootPattern)
	validModelPath  = regexp.MustCompile(modelPathPattern)
	validCheckpoint = regexp.MustCompile(checkpointPattern)
	validMethodName = map[string]struct{}{
		"am.recognize":       {},
		"lm.score":           {},
		"lm.generate":        {},
		"lm.generate_stream": {},
		"lm.embed":           {},
		"lm.gradient":        {},
		"vm.classify":        {},
		"vm.generate":        {},
		"vm.embed":           {},
		"vm.detect":          {},
		"vm.image_to_text":   {},
		"vm.video_to_text":   {},
	}
)

// ValidateConfigProto checks whether a Config proto message is valid.
func ValidateConfigProto(cfg *pb.Config) error {
	if fsRoot := cfg.GetFsRoot(); !validFSRoot.MatchString(fsRoot) {
		return fmt.Errorf("FS root %q must match %q: %w", fsRoot, fsRootPattern, errors.ErrInvalidArgument)
	}
	if acl := cfg.GetAdminAcl(); acl != "" {
		validACLNamePrefix := env.Get().RequiredACLNamePrefix()
		if !strings.HasPrefix(acl, validACLNamePrefix) {
			return fmt.Errorf("%s is not a valid ACL name (must start with %s): %w", acl, validACLNamePrefix, errors.ErrInvalidArgument)
		}
	}
	return nil
}

// ValidateConfigUpdate checks whether an update of a Config proto message is valid.
func ValidateConfigUpdate(previous, change *pb.Config) error {
	if err := ValidateConfigProto(change); err != nil {
		return err
	}
	if previous.GetFsRoot() != change.GetFsRoot() {
		return fmt.Errorf("FS root can't be changed: %w", errors.ErrInvalidArgument)
	}
	return nil
}

// ValidateModelFullName checks whether a model full name string is valid within saxCell.
func ValidateModelFullName(modelFullName string, saxCell string) error {
	fullName, err := naming.NewModelFullName(modelFullName)
	if err != nil {
		return err
	}
	if fullName.CellFullName() != saxCell {
		return fmt.Errorf("model %s must be within SAX cell %q: %w", modelFullName, saxCell, errors.ErrInvalidArgument)
	}
	return nil
}

// ValidateModelProto checks whether a Model proto message is valid within saxCell.
func ValidateModelProto(model *pb.Model, saxCell string) error {
	if err := ValidateModelFullName(model.GetModelId(), saxCell); err != nil {
		return err
	}
	if !validModelPath.MatchString(model.GetModelPath()) {
		return fmt.Errorf("model path %q must match %q: %w", model.GetModelPath(), modelPathPattern, errors.ErrInvalidArgument)
	}
	if !validCheckpoint.MatchString(model.GetCheckpointPath()) {
		return fmt.Errorf("checkpoint path %q must match %q: %w", model.GetCheckpointPath(), checkpointPattern, errors.ErrInvalidArgument)
	}
	if model.GetRequestedNumReplicas() < 0 {
		return fmt.Errorf("number of replicas %d must be non-negative: %w", model.GetRequestedNumReplicas(), errors.ErrInvalidArgument)
	}
	validACLNamePrefix := env.Get().RequiredACLNamePrefix()
	if aclname := model.GetAdminAcl(); aclname != "" {
		if !strings.HasPrefix(aclname, validACLNamePrefix) {
			return fmt.Errorf("%s is not a valid ACL name (must start with %s): %w", aclname, validACLNamePrefix, errors.ErrInvalidArgument)
		}
	}
	if model.GetAcls() != nil && model.GetAcls().GetItems() != nil {
		for method, aclname := range model.GetAcls().GetItems() {
			if _, ok := validMethodName[method]; !ok {
				return fmt.Errorf("%s is not a valid method name (%v): %w", method, validMethodName, errors.ErrInvalidArgument)
			}
			if !strings.HasPrefix(aclname, validACLNamePrefix) {
				return fmt.Errorf("%s is not a valid ACL name (must start with %s): %w", aclname, validACLNamePrefix, errors.ErrInvalidArgument)
			}
		}
	}
	return nil
}

// ValidateModelUpdate checks whether an update of a Model proto message is valid within saxCell.
func ValidateModelUpdate(previous, change *pb.Model, saxCell string) error {
	if err := ValidateModelProto(change, saxCell); err != nil {
		return err
	}
	if previous.GetModelId() != change.GetModelId() {
		return fmt.Errorf("model full name can't be changed: %w", errors.ErrInvalidArgument)
	}
	if previous.GetModelPath() != change.GetModelPath() {
		return fmt.Errorf("model path can't be changed: %w", errors.ErrInvalidArgument)
	}
	if previous.GetCheckpointPath() != change.GetCheckpointPath() {
		return fmt.Errorf("checkpoint path can't be changed: %w", errors.ErrInvalidArgument)
	}
	return nil
}

// ValidateWatchLocRequest checks whether a WatchLocRequest instance is valid.
func ValidateWatchLocRequest(req *pb.WatchLocRequest) error {
	if err := naming.ValidateModelFullName(req.GetModelId()); err != nil {
		return err
	}
	if seqno := req.GetSeqno(); seqno < 0 {
		return fmt.Errorf("WatchLoc seqno %d should be non-negative %w", seqno, errors.ErrInvalidArgument)
	}
	return nil
}

// ValidateJoinRequest checks whether a JoinRequest instance is valid.
func ValidateJoinRequest(req *pb.JoinRequest) error {
	addr := req.GetAddress()
	if addr == "" {
		return fmt.Errorf("JoinRequest.modelet_address cannot be empty: %w", errors.ErrInvalidArgument)
	}

	modelet := req.GetModelServer()
	if modelet == nil {
		return fmt.Errorf("JoinRequest.modelet missing: %w", errors.ErrInvalidArgument)
	}

	chipType := modelet.GetChipType()
	if chipType == pb.ModelServer_CHIP_TYPE_UNKNOWN {
		return fmt.Errorf("Modelet.chip_type cannot be unknown: %w", errors.ErrInvalidArgument)
	}

	chipTopology := modelet.GetChipTopology()
	if chipTopology == pb.ModelServer_CHIP_TOPOLOGY_UNKNOWN {
		return fmt.Errorf("Modelet.chip_topology cannot be unknown: %w", errors.ErrInvalidArgument)
	}

	return nil
}
