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

package assigner

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"saxml/admin/protobuf"
	"saxml/common/naming"
)

func reportActions(acts []Action) string {
	sort.Slice(acts, func(i, j int) bool {
		cmp := strings.Compare(string(acts[i].Addr), string(acts[j].Addr))
		if cmp != 0 {
			return cmp < 0
		}
		return strings.Compare(acts[i].Model.ModelName(), acts[j].Model.ModelName()) < 0
	})
	var ret string
	for _, act := range acts {
		ret += fmt.Sprintf("%s: %s\n", string(act.Addr), act.Model.ModelName())
	}
	return ret
}

func report(a *Assigner) string {
	var ret string

	assignment := a.GetAssignment()
	var names []naming.ModelFullName
	for name := range assignment {
		names = append(names, name)
	}
	sort.Slice(names, func(i, j int) bool {
		return strings.Compare(names[i].ModelName(), names[j].ModelName()) < 0
	})
	ret += "\n========\nAssignment:\n"
	for _, fullName := range names {
		addrs := assignment[fullName]
		ret += fmt.Sprintf("%s: [", fullName.ModelName())
		sort.Slice(addrs, func(i, j int) bool {
			return strings.Compare(string(addrs[i]), string(addrs[j])) < 0
		})
		for i, addr := range addrs {
			if i > 0 {
				ret += " "
			}
			ret += fmt.Sprintf("%s", addr)
		}
		ret += "]\n"
	}
	ret += "========\nToUnload\n"
	ret += reportActions(a.GetToUnload())
	ret += "========\nToLoad\n"
	ret += reportActions(a.GetToLoad())
	return ret
}

type serverCase struct {
	addr    string
	capGB   int64
	params  []string
	loaded  []string
	loading []string
}

func addServer(t *testing.T, a *Assigner, c *serverCase) {
	s := &ServerInfo{
		memoryCapacity: c.capGB << 30,
		loadedModel:    map[naming.ModelFullName]protobuf.ModelStatus{},
	}
	for _, p := range c.params {
		s.servableModelPath = append(s.servableModelPath, ParamPath(p))
	}
	for _, name := range c.loaded {
		s.loadedModel[naming.NewModelFullNameT(t, "test", name)] = protobuf.Loaded
	}
	for _, name := range c.loading {
		s.loadedModel[naming.NewModelFullNameT(t, "test", name)] = protobuf.Loading
	}
	a.AddServer(ServerAddr(c.addr), s)
}

type modelCase struct {
	name       string
	path       string
	replicas   int
	requiredGB int64
}

func addModel(t *testing.T, a *Assigner, c *modelCase) {
	m := &ModelInfo{
		modelPath:      ParamPath(c.path),
		neededReplicas: c.replicas,
		memoryRequired: c.requiredGB << 30,
	}
	a.AddModel(naming.NewModelFullNameT(t, "test", c.name), m)
}

type testCase struct {
	desc           string
	servers        []serverCase
	models         []modelCase
	expectedReport string
}

func setupCase(t *testing.T, a *Assigner, c *testCase) {
	for _, s := range c.servers {
		addServer(t, a, &s)
	}
	for _, m := range c.models {
		addModel(t, a, &m)
	}
}

func TestBasicAssignment(t *testing.T) {
	testCases := []testCase{
		{
			desc: "1 server, n models, clean slate",
			servers: []serverCase{
				{"s0", 16, []string{"p0", "p1"}, []string{}, []string{}},
			},
			models: []modelCase{
				{"m0", "p0", 1, 1},
				{"m1", "p1", 1, 20},
				{"m2", "p2", 1, 8},
				{"m3", "p1", 1, 10},
			},
			expectedReport: `
========
Assignment:
m0: [s0]
m3: [s0]
========
ToUnload
========
ToLoad
s0: m0
s0: m3
`,
		},
		{
			desc: "n servers, n models, clean slate, no model memory info",
			servers: []serverCase{
				{"s0", 16, []string{"p0"}, []string{}, []string{}},
				{"s1", 16, []string{"p1"}, []string{}, []string{}},
				{"s2", 16, []string{"p2"}, []string{}, []string{}},
			},
			models: []modelCase{
				{"m0", "p0", 1, -1},
				{"m1", "p1", 1, -1},
				{"m2", "p2", 1, -1},
			},
			expectedReport: `
========
Assignment:
m0: [s0]
m1: [s1]
m2: [s2]
========
ToUnload
========
ToLoad
s0: m0
s1: m1
s2: m2
`,
		},
		{
			desc: "n servers, 1 model, clean slate",
			servers: []serverCase{
				{"s0", 16, []string{"p0"}, []string{}, []string{}},
				{"s1", 16, []string{"p0"}, []string{}, []string{}},
				{"s2", 16, []string{"p0"}, []string{}, []string{}},
			},
			models: []modelCase{
				{"m0", "p0", 5, 4},
			},
			expectedReport: `
========
Assignment:
m0: [s0 s1 s2]
========
ToUnload
========
ToLoad
s0: m0
s1: m0
s2: m0
`,
		},
		{
			desc: "n servers, 1 model, loaded, but unpublished",
			servers: []serverCase{
				{"s0", 16, []string{"p0"}, []string{"m0"}, []string{}},
				{"s1", 16, []string{"p0"}, []string{"m0"}, []string{}},
				{"s2", 16, []string{"p0"}, []string{"m0"}, []string{}},
			},
			models: []modelCase{},
			expectedReport: `
========
Assignment:
========
ToUnload
s0: m0
s1: m0
s2: m0
========
ToLoad
`,
		},
		{
			desc: "3 servers, 1 model, 1 replica already in loading, need 2 replicas",
			servers: []serverCase{
				{"s0", 16, []string{"p0"}, []string{}, []string{}},
				{"s1", 16, []string{"p0"}, []string{}, []string{"m0"}},
				{"s2", 16, []string{"p0"}, []string{}, []string{}},
			},
			models: []modelCase{
				{"m0", "p0", 2, 4},
			},
			expectedReport: `
========
Assignment:
m0: [s0 s1]
========
ToUnload
========
ToLoad
s0: m0
`,
		},
		{
			desc: "3 servers, 1 model, 2 replicas already (1 loading, 1 loaded), only need 1 replica, should unload the loading replica",
			servers: []serverCase{
				{"s0", 16, []string{"p0"}, []string{}, []string{}},
				{"s1", 16, []string{"p0"}, []string{}, []string{"m0"}},
				{"s2", 16, []string{"p0"}, []string{"m0"}, []string{}},
			},
			models: []modelCase{
				{"m0", "p0", 1, 4},
			},
			expectedReport: `
========
Assignment:
m0: [s2]
========
ToUnload
s1: m0
========
ToLoad
`,
		},
	}
	for _, tc := range testCases {
		a := New()
		setupCase(t, a, &tc)
		a.Assign()
		actual := report(a)
		if actual != tc.expectedReport {
			t.Errorf("Assignment(%s) err got %s, want %s", tc.desc, actual, tc.expectedReport)
		}
	}
}
