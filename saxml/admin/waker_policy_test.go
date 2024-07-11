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

package wakerpolicy

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"saxml/admin/state"
	"saxml/common/naming"
)

type fakeState struct {
	Models       map[naming.ModelFullName]*state.ModelWithStatus
	ServerStatus state.ServerStatus
}

type modelsAndState struct {
	models []string
	state  state.ServerStatus
}

func (fs *fakeState) SeenModels() map[naming.ModelFullName]*state.ModelWithStatus {
	return fs.Models
}

func (fs *fakeState) LastReportedServerStatus() state.ServerStatus {
	return fs.ServerStatus
}

func createFakeState(t *testing.T, server modelsAndState) *fakeState {
	fs := &fakeState{
		Models:       make(map[naming.ModelFullName]*state.ModelWithStatus),
		ServerStatus: state.ServerStatus{},
	}
	for _, model := range server.models {
		name := naming.NewModelFullNameT(t, "test", model)
		fs.Models[name] = &state.ModelWithStatus{}
	}
	fs.ServerStatus = server.state
	return fs
}

func createDormantServerReport(latest, previous float32) state.ServerStatus {
	return state.ServerStatus{
		IsDormant:                     true,
		EarlyRejectionErrorsPerSecond: [2]float32{latest, previous},
	}
}

func createActiveServerReport() state.ServerStatus {
	return state.ServerStatus{
		IsDormant:                     false,
		EarlyRejectionErrorsPerSecond: [2]float32{0.0, 0.0},
	}
}

func TestDormantServerWakeUpPolicy(t *testing.T) {
	tests := []struct {
		desc              string
		servers           map[string]modelsAndState
		expectedCandiates []string
	}{
		{
			desc:              "No servers results in no-ops",
			servers:           map[string]modelsAndState{},
			expectedCandiates: []string{},
		},
		{
			desc: "No dormant servers results in no-ops",
			servers: map[string]modelsAndState{
				"server1": modelsAndState{
					models: []string{"model1"},
					state:  createActiveServerReport(),
				},
				"server2": modelsAndState{
					models: []string{"model1"},
					state:  createActiveServerReport(),
				},
			},
			expectedCandiates: []string{},
		}, {
			desc: "Don't wake-up dormant server if no early rejection load",
			servers: map[string]modelsAndState{
				"server2": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(0.0, 0.0),
				},
				"server1": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(0.0, 0.0),
				},
			},
			expectedCandiates: []string{},
		}, {
			desc: "Wake-up server with max address if all servers are dormant and seeing early rejection load",
			servers: map[string]modelsAndState{
				"server2": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(46.0, 0.0),
				},
				"server1": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(47.0, 1.0),
				},
			},
			expectedCandiates: []string{"server2"},
		}, {
			desc: "Wake-up more dormant server if seeing avg of early rejection load increasing greater than active servers count",
			servers: map[string]modelsAndState{
				"server2": modelsAndState{
					models: []string{"model1"},
					state:  createActiveServerReport(),
				},
				"server1": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(99.0, 24.0),
				},
				"server3": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(100.0, 27.0),
				},
			},
			// The avg early rejection load change for model1 is ((99 - 24) + (100 - 27)) / 2 = 74, and
			// it's greater than count of active servers (i.e. 1), so wake-up server3.
			expectedCandiates: []string{"server3"},
		}, {
			desc: "Don't wake-up dormant server if seeing avg of early rejection load increasing no greater than active servers count",
			servers: map[string]modelsAndState{
				"server2": modelsAndState{
					models: []string{"model1"},
					state:  createActiveServerReport(),
				},
				"server1": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(100.0, 99.0),
				},
				"server3": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(103.0, 103.0),
				},
			},
			// The avg early rejection load change for model1 is ((100 - 99) + (103 - 103)) / 2 = 0.5
			// which is smaller than count of active servers (i.e. 1), so no dormant server will wake-up.
			expectedCandiates: []string{},
		}, {
			desc: "Don't wake-up dormant server if seeing avg of early rejection load decreasing",
			servers: map[string]modelsAndState{
				"server2": modelsAndState{
					models: []string{"model1"},
					state:  createActiveServerReport(),
				},
				"server1": modelsAndState{
					models: []string{"model1"},
					state:  createDormantServerReport(24.0, 99.0),
				},
			},
			// The early rejection load on server1 is decreasing as 24 - 99 = -75, and it's less than
			// count of active servers (i.e. 1), so no dormant server will wake-up.
			expectedCandiates: []string{},
		}, {
			desc: "Don't wake-up same dormant server twice in the same policy cycle",
			servers: map[string]modelsAndState{
				"server1": modelsAndState{
					models: []string{"model1", "model2"},
					state:  createDormantServerReport(180.0, 50.0),
				},
			},
			// The early rejection load on server1 is significantly increasing as the server is seeing
			// load for both model1, model2. However, we wake-up server1 at most once.
			expectedCandiates: []string{"server1"},
		},
	}

	for _, tc := range tests {
		policy := NewWakerPolicy()
		for addr, modelAndState := range tc.servers {
			policy.AddServerStatus(addr, createFakeState(t, modelAndState))
		}
		candidates := policy.Decide()
		actual := []string{}
		for _, candidate := range candidates {
			actual = append(actual, string(candidate))
		}
		if !cmp.Equal(actual, tc.expectedCandiates) {
			t.Errorf("Policy test (%s): got %v, want %v", tc.desc, actual, tc.expectedCandiates)
		}
	}
}
