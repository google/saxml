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

package saxcommand

import (
	"context"

	"flag"
	log "github.com/golang/glog"
	"github.com/google/subcommands"
	"saxml/client/go/sax"
)

// SaveCmd is the command for save checkpoint.
type SaveCmd struct {
}

// Name returns the name of SaveCmd.
func (*SaveCmd) Name() string { return "save" }

// Synopsis returns the synopsis of SaveCmd.
func (*SaveCmd) Synopsis() string { return "Save a model checkpoint." }

// SetFlags sets flags for SaveCmd.
func (c *SaveCmd) SetFlags(f *flag.FlagSet) {
}

// Usage returns the full usage of SaveCmd.
func (*SaveCmd) Usage() string {
	return `save <model ID>  <CNS path>:
    Save given model checkpoint to the CNS path.
`
}

// Execute SaveCmd.
func (c *SaveCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 {
		log.Errorf("Provide a model ID and a CNS path.")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}

	ctx, cancel := context.WithTimeout(ctx, *cmdTimeout)
	defer cancel()
	if err := m.Saver().Save(ctx, f.Args()[1]); err != nil {
		log.Errorf("Failed to save model: %v", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}
