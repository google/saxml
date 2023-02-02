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

package validator_test

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"saxml/admin/validator"
	"saxml/common/platform/env"
	_ "saxml/common/platform/register" // registers a platform

	apb "saxml/protobuf/admin_go_proto_grpc"
	cpb "saxml/protobuf/common_go_proto"
)

type testConfig struct {
	config *apb.Config
}

func validConfig() *testConfig {
	return &testConfig{
		config: &apb.Config{
			FsRoot:   "/path/to/root/dir",
			AdminAcl: env.Get().RequiredACLNamePrefix() + "dev-team",
		},
	}
}

func (c *testConfig) withFsRoot(fsRoot string) *testConfig {
	c.config.FsRoot = fsRoot
	return c
}

func (c *testConfig) withAdminACL(adminACL string) *testConfig {
	c.config.AdminAcl = adminACL
	return c
}

func TestCheckConfigProto(t *testing.T) {
	tests := []struct {
		desc    string
		input   *testConfig
		wantErr error
	}{
		{
			"ok",
			validConfig(),
			nil,
		},
		{
			"FS root empty not ok",
			validConfig().withFsRoot(""),
			cmpopts.AnyError,
		},
		{
			"FS root root ok",
			validConfig().withFsRoot("/"),
			nil,
		},
		{
			"FS root full path ok",
			validConfig().withFsRoot("/some/path"),
			nil,
		},
		{
			"FS root relative path ok",
			validConfig().withFsRoot("some/path"),
			nil,
		},
		{
			"FS root ilegal character not ok",
			validConfig().withFsRoot("/some/path$"),
			cmpopts.AnyError,
		},
		{
			"FS root vertical bar not ok",
			validConfig().withFsRoot("/some/path|with|vertical|bars"),
			cmpopts.AnyError,
		},
		{
			"FS root hypyen ok",
			validConfig().withFsRoot("/some/path-with-hyphens"),
			nil,
		},
		{
			"admin ACL empty ok",
			validConfig().withAdminACL(""),
			nil,
		},
		{
			"admin ACL invalid not ok",
			validConfig().withAdminACL("dev-group"),
			cmpopts.AnyError,
		},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			err := validator.ValidateConfigProto(tc.input.config)
			if !cmp.Equal(tc.wantErr, err, cmpopts.EquateErrors()) {
				t.Errorf("ValidateConfigProto(%v) err got %v, want %v", tc.input, err, tc.wantErr)
			}
		})
	}
}

func TestCheckConfigUpdate(t *testing.T) {
	tests := []struct {
		desc    string
		before  *testConfig
		after   *testConfig
		wantErr error
	}{
		{
			"no change ok",
			validConfig(),
			validConfig(),
			nil,
		},
		{
			"FS root changed not ok",
			validConfig(),
			validConfig().withFsRoot("/other/path"),
			cmpopts.AnyError,
		},
		{
			"admin ACL changed ok",
			validConfig(),
			validConfig().withAdminACL(""),
			nil,
		},
		{
			"admin ACL changed not ok",
			validConfig(),
			validConfig().withAdminACL("other-group"),
			cmpopts.AnyError,
		},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			err := validator.ValidateConfigUpdate(tc.before.config, tc.after.config)
			if !cmp.Equal(tc.wantErr, err, cmpopts.EquateErrors()) {
				t.Errorf("ValidateConfigUpdate(%v, %v) err got %v, want %v", tc.before.config, tc.after.config, err, tc.wantErr)
			}
		})
	}
}

type testModel struct {
	model   *apb.Model
	saxCell string
}

func validModel() *testModel {
	return &testModel{
		model: &apb.Model{
			ModelId:              "/sax/bar/foo",
			ModelPath:            "google3.learning.multipod.sax.lm.params.ulm.Base",
			CheckpointPath:       "/cns/od-d/home/blah/blah/blah/darn-0000",
			RequestedNumReplicas: 10,
			Acls: &cpb.AccessControlLists{
				Items: map[string]string{
					"lm.score":    env.Get().RequiredACLNamePrefix() + "all",
					"lm.generate": env.Get().RequiredACLNamePrefix() + "sax-dev",
				},
			},
		},
		saxCell: "/sax/bar",
	}
}

func (m *testModel) withModelID(id string) *testModel {
	m.model.ModelId = id
	return m
}

func (m *testModel) withModelPath(path string) *testModel {
	m.model.ModelPath = path
	return m
}

func (m *testModel) withCheckpointPath(path string) *testModel {
	m.model.CheckpointPath = path
	return m
}

func (m *testModel) withRequestedNumReplicas(num int32) *testModel {
	m.model.RequestedNumReplicas = num
	return m
}

func (m *testModel) withSaxCell(saxCell string) *testModel {
	m.saxCell = saxCell
	return m
}

func (m *testModel) withACL(method, aclname string) *testModel {
	m.model.GetAcls().GetItems()[method] = aclname
	return m
}

func TestCheckModelProto(t *testing.T) {
	tests := []struct {
		desc    string
		input   *testModel
		wantErr error
	}{
		{
			"ok",
			validModel(),
			nil,
		},
		{
			"invalid model id",
			validModel().withModelID(""),
			cmpopts.AnyError,
		},
		{
			"invalid model id",
			validModel().withModelID("/sax"),
			cmpopts.AnyError,
		},
		{
			"invalid model id",
			validModel().withModelID("/sax/bar"),
			cmpopts.AnyError,
		},
		{
			"invalid model id",
			validModel().withModelID("foo"),
			cmpopts.AnyError,
		},
		{
			"invalid model id",
			validModel().withModelID("!@#"),
			cmpopts.AnyError,
		},
		{
			"invalid model path",
			validModel().withModelPath(""),
			cmpopts.AnyError,
		},
		{
			"invalid model path",
			validModel().withModelPath("google3/foo/bar"),
			cmpopts.AnyError,
		},
		{
			"invalid model path",
			validModel().withModelPath(" a b c "),
			cmpopts.AnyError,
		},
		{
			"invalid checkpoint path",
			validModel().withCheckpointPath(""),
			cmpopts.AnyError,
		},
		{
			"invalid checkpoint path",
			validModel().withCheckpointPath("/cns "),
			cmpopts.AnyError,
		},
		{
			"invalid checkpoint path",
			validModel().withCheckpointPath("/cns /flag"),
			cmpopts.AnyError,
		},
		{
			"invalid checkpoint path",
			validModel().withCheckpointPath("/cns$"),
			cmpopts.AnyError,
		},
		{
			"invalid checkpoint path",
			validModel().withCheckpointPath("/!@$"),
			cmpopts.AnyError,
		},
		{
			"invalid requested num replicas",
			validModel().withRequestedNumReplicas(-1),
			cmpopts.AnyError,
		},
		{
			"invalid sax cell",
			validModel().withSaxCell("/sax/baz"),
			cmpopts.AnyError,
		},
		{
			"invalid method name",
			validModel().withACL("never.method", env.Get().RequiredACLNamePrefix()+"all"),
			cmpopts.AnyError,
		},
		{
			"invalid acl name",
			validModel().withACL("lm.score", "never_an_aclname"),
			cmpopts.AnyError,
		},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			err := validator.ValidateModelProto(tc.input.model, tc.input.saxCell)
			if !cmp.Equal(tc.wantErr, err, cmpopts.EquateErrors()) {
				t.Errorf("ValidateModelProto(%v) err got %v, want %v", tc.input, err, tc.wantErr)
			}
		})
	}
}

func TestCheckModelUpdate(t *testing.T) {
	tests := []struct {
		desc    string
		before  *testModel
		after   *testModel
		wantErr error
	}{
		{
			"no change",
			validModel(),
			validModel(),
			nil,
		},
		{
			"model id changed",
			validModel(),
			validModel().withModelID("/sax/bar/foo2"),
			cmpopts.AnyError,
		},
		{
			"model path changed",
			validModel(),
			validModel().withModelID("google3.learning.multipod.sax.lm.params.ulm2.Base"),
			cmpopts.AnyError,
		},
		{
			"model checkpoint changed",
			validModel(),
			validModel().withCheckpointPath("google3.learning.multipod.sax.lm.params.ulm2.Base"),
			cmpopts.AnyError,
		},
		{
			"num replicas changed",
			validModel(),
			validModel().withRequestedNumReplicas(100),
			nil,
		},
		{
			"num replicas changed, not ok",
			validModel(),
			validModel().withRequestedNumReplicas(-100),
			cmpopts.AnyError,
		},
	}
	for _, tc := range tests {
		t.Run(tc.desc, func(t *testing.T) {
			err := validator.ValidateModelUpdate(tc.before.model, tc.after.model, tc.after.saxCell)
			if !cmp.Equal(tc.wantErr, err, cmpopts.EquateErrors()) {
				t.Errorf("ValidateModelUpdate(%v, %v) err got %v, want %v", tc.before.model, tc.after.model, err, tc.wantErr)
			}
		})
	}
}

func TestValidateJoinRequest(t *testing.T) {
	req := &apb.JoinRequest{}
	if err := validator.ValidateJoinRequest(req); err == nil {
		t.Errorf("Join(%v) no error, want some error for empty Address", req)
	}

	req.Address = "localhost:20000"
	req.ModelServer = &apb.ModelServer{}
	if err := validator.ValidateJoinRequest(req); err == nil {
		t.Errorf("Join(%v) no error, want some error for unknown ChipType", req)
	}

	req.GetModelServer().ChipType = apb.ModelServer_CHIP_TYPE_TPU_V2
	if err := validator.ValidateJoinRequest(req); err == nil {
		t.Errorf("Join(%v) no error, want some error for unknown ChipTopology", req)
	}

	req.GetModelServer().ChipTopology = apb.ModelServer_CHIP_TOPOLOGY_1X1
	if err := validator.ValidateJoinRequest(req); err != nil {
		t.Errorf("Join(%v) err %v, want no error even with empty ServableModelPaths", req, err)
	}

	req.GetModelServer().ServableModelPaths = []string{"lingvo.lm.glam_8b"}
	if err := validator.ValidateJoinRequest(req); err != nil {
		t.Errorf("Join(%v) err %v, want no error", req, err)
	}
}
