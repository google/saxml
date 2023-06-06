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

// Package basiceventlogger implements env.EventLogger and just logs events on debug level.
package basiceventlogger

import (
	log "github.com/golang/glog"
	"saxml/common/eventlog"
)

// Logger is implementation of eventlog.Logger which just writes to basic log.
type Logger struct {
}

// New returns an instance of Logger.
func New() *Logger {
	return &Logger{}
}

// Log writes event type and additional arguments to base/go/log log.
func (*Logger) Log(eventType eventlog.Type, args ...any) {
	log.V(4).Infof("Logging event %s: %v", eventType, args)
}

// Close does nothing for basiceventlogger and needed only to satisfy eventlog.Logger.
func (*Logger) Close() {}
