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

// Package eventlog provides interface for event logging in SAX.
package eventlog

// Type of the event.
type Type int

const (
	// Deploy is happening when model is published.
	Deploy Type = iota
	// ServingStart indicates that at least one model server serving the model.
	ServingStart
	// ServingStop indicates that client is initiated model unpublish.
	ServingStop
)

func (t Type) String() string {
	switch t {
	case Deploy:
		return "Deploy"
	case ServingStart:
		return "ServingStart"
	case ServingStop:
		return "ServingStop"
	}
	return "Unknown"
}

// Logger is the interface for logging events.
type Logger interface {
	// Log logs specific event type with generic args.
	// It's safe to call Log() concurrently from multiple threads.
	Log(eventType Type, args ...any)
	// Close notifies logger that it should stop any activity and release resources.
	Close()
}
