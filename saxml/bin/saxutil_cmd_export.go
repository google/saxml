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
	"strings"

	"flag"
	log "github.com/golang/glog"
	"github.com/google/subcommands"
	"saxml/client/go/sax"
)

// ExportCmd is the command for WatchAddresses. Useful for debugging.
type ExportCmd struct {
	rngSeedMode string
	signatures  []string
}

// Name returns the name of ExportCmd.
func (*ExportCmd) Name() string { return "export" }

// Synopsis returns the synopsis of ExportCmd.
func (*ExportCmd) Synopsis() string { return "Export a model to a TF SavedModel." }

// Usage returns the full usage of ExportCmd.
func (*ExportCmd) Usage() string {
	return `export [--rng_seed_mode=<stateful|stateless>] <model ID> <method name> <CNS path>:
    Export the given method of the given model to the CNS path.
`
}

// SetFlags sets flags for ExportCmd.
func (c *ExportCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.rngSeedMode, "rng_seed_mode", "stateful", "RNG seed mode. Either 'stateful' or 'stateless'.")
	f.StringListVar(&c.signatures, "signatures", nil,
		"A list of signatures to export to. Must equal to the length of Methods to export and ordering.")
}

// Execute executes ExportCmd.
func (c *ExportCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 3 {
		log.Errorf("Provide a model ID, a method name and a CNS path.")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}

	if err := m.Exporter().Export(ctx, strings.Split(f.Args()[1], ","), f.Args()[2], c.rngSeedMode, c.signatures); err != nil {
		log.Errorf("Failed to export model: %v", err)
		return subcommands.ExitFailure
	}
	return subcommands.ExitSuccess
}
