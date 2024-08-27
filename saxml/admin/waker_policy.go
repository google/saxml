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

// Package wakerpolicy decides which dormant servers should be woken up based on the current model server status.
package wakerpolicy

import (
	"fmt"

	log "github.com/golang/glog"
	"saxml/admin/state"
	"saxml/common/naming"
)

// serverAddr represents a model server address. E.g., 1.2.3.4:14001.
type serverAddr string

// modelFullName identifies a model in the form of /sax/<cell>/<model>.
type modelName = naming.ModelFullName

// ServerStatusFeeder provides the current model server status.
type ServerStatusFeeder interface {
	SeenModels() map[naming.ModelFullName]*state.ModelWithStatus
	LastReportedServerStatus() state.ServerStatus
}

type dormantStats struct {
	totalServerCount    int
	dormantServerCount  int
	candidateToWakeUp   serverAddr
	sumErrorsRateChange float32
}

// WakerPolicy models per-model load stats and decides which dormant servers should be woken up.
type WakerPolicy struct {
	perModelDormantServerStats map[modelName]*dormantStats
}

// NewWakerPolicy creates a new WakerPolicy instance.
func NewWakerPolicy() *WakerPolicy {
	return &WakerPolicy{
		perModelDormantServerStats: make(map[modelName]*dormantStats),
	}
}

// AddServerStatus adds a model server status to the policy module for consideration.
func (w *WakerPolicy) AddServerStatus(server string, statusFeeder ServerStatusFeeder) {
	addr := serverAddr(server)
	lastReported := statusFeeder.LastReportedServerStatus()
	for name := range statusFeeder.SeenModels() {
		if w.perModelDormantServerStats[name] == nil {
			w.perModelDormantServerStats[name] = &dormantStats{}
		}
		stats := w.perModelDormantServerStats[name]
		stats.totalServerCount++
		if lastReported.IsDormant {
			stats.dormantServerCount++
			stats.candidateToWakeUp = max(stats.candidateToWakeUp, addr)
			hist := lastReported.EarlyRejectionErrorsPerSecond
			stats.sumErrorsRateChange = hist[0] - hist[1]
		}
	}
}

// Decide returns a list of server addresses that should be woken up.
func (w *WakerPolicy) Decide() []string {
	// Policy for wake-up a candidate server for a model: "the first derivative of rejection rate on
	// dormant server > the number of active servers."
	//
	// On the one hand, this policy is based on a simple intuition "if we see increasing rejection
	// error rate per-dormant-server, this suggests we could use one more server." Therefore on the
	// opposite side, if we see decreasing rejection error rate, this suggests we have sufficient
	// active servers and it's no-ops.
	//
	// On the other hand, the "# active servers" is based on the fact that: "when k client queries and
	// there is k active servers, each model server should see ≤k rejection error because of retry."
	// So if we see error rate change >k per dormant server, we know there is more traffic than
	// current active servers may handle.
	//
	// The goal is to save resources by minimizing amount of dormant server waking-up, for example,
	// - No-ops when there is no early rejection error to a model.
	// - No-ops if previous server is not finished waking-up yet.
	// - No-ops if early rejection error rate is not increasing.
	candidates := map[serverAddr]bool{}
	for mname, stats := range w.perModelDormantServerStats {
		logMsg := fmt.Sprintf("Running server wake policy against model: %v with stats: %v \n", mname, stats)
		if stats.dormantServerCount > 0 {
			averageErrorRateChange := stats.sumErrorsRateChange / float32(stats.dormantServerCount)
			activeServersCount := float32(stats.totalServerCount - stats.dormantServerCount)
			shouldWakeUp := averageErrorRateChange > activeServersCount
			if shouldWakeUp {
				addr := stats.candidateToWakeUp
				candidates[addr] = true
			}
			logMsg += fmt.Sprintf("Decision: %v as seeing avg_error_rate_delta: %v, active_server_count: %v",
				shouldWakeUp, averageErrorRateChange, activeServersCount)
		}
		log.V(1).Infof(logMsg)
	}
	servers := []string{}
	for cand := range candidates {
		servers = append(servers, string(cand))
	}
	return servers
}
