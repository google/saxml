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
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"flag"
	log "github.com/golang/glog"
	"github.com/google/subcommands"
	"github.com/olekukonko/tablewriter"
	"saxml/client/go/sax"
)

// GenerateCmd is the command for generate
type GenerateCmd struct {
	extra  string
	stream bool
	raw    bool
}

// Name returns the name of GenerateCmd.
func (*GenerateCmd) Name() string { return "lm.generate" }

// Synopsis returns the synopsis of GenerateCmd.
func (*GenerateCmd) Synopsis() string { return "generate a text against a given model" }

// Usage returns the full usage of GenerateCmd.
func (*GenerateCmd) Usage() string {
	return `generate ModelID Query:
	generates a text input using a published language model. Support temperature through: -extra="temperature:0.2"
`
}

// SetFlags sets the flags for GenerateCmd.
func (c *GenerateCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Generate().")
	f.BoolVar(&c.stream, "stream", false, "stream responses")
}

func (c *GenerateCmd) streamingGenerate(ctx context.Context, query string, lm *sax.LanguageModel) subcommands.ExitStatus {
	chanStreamResults := lm.GenerateStream(ctx, query, ExtraInputs(c.extra)...)
	var fullText, prev string
	var end bool
	for {
		select {
		case <-ctx.Done():
			fmt.Printf("Command canceled: %v\n", ctx.Err())
			return subcommands.ExitFailure
		case streamResult := <-chanStreamResults:
			switch streamResult.Err {
			case nil:
				// Every streamResult contains all tokens decoded so far represented as separate texts.
				var texts []string
				for _, generateResult := range streamResult.Result {
					if generateResult.Score == 0 {
						// All but the last streaming response have zero scores.
						texts = append(texts, generateResult.Text)
					} else {
						// The last streaming response contains the full text decoded from all tokens and score.
						if end {
							fmt.Printf("Got two full texts unexpectedly: %q and %q\n", fullText, generateResult.Text)
							return subcommands.ExitFailure
						}
						end = true
						fullText = generateResult.Text
					}
				}
				if end {
					// If this is the last streaming response, skip to the io.EOF case in the next iteration.
					continue
				}

				// For all but the last streaming responses, print incrementally added text compared to the
				// last streaming response.
				curr := strings.Join(texts, " ")
				if !strings.HasPrefix(curr, prev) {
					fmt.Printf("Current response %q not prefixed by previous response %q\n", curr, prev)
					return subcommands.ExitFailure
				}
				diff := strings.TrimPrefix(curr, prev)
				fmt.Print(diff)
				prev = curr
			case io.EOF:
				// Print the final full text on a newline.
				fmt.Println()
				fmt.Println("Full text:", fullText)
				return subcommands.ExitSuccess
			default:
				fmt.Printf("Command failed: %v\n", streamResult.Err)
				return subcommands.ExitFailure
			}
		}
	}
}

// Executes GenerateCmd.
func (c *GenerateCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 {
		log.Errorf("Provide model and query for generate")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	lm := m.LM()
	if lm == nil {
		log.Errorf("Failed to create language model: %v", err)
		return subcommands.ExitFailure
	}

	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()

	// Streaming generate.
	if c.stream {
		return c.streamingGenerate(ctx, f.Args()[1], lm)
	}

	// Non-streaming generate.
	generates, err := lm.Generate(ctx, f.Args()[1], ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to generate query: %v", err)
		return subcommands.ExitFailure
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"generate", "Score"})
	for _, generate := range generates {
		table.Append([]string{generate.Text, strconv.FormatFloat(generate.Score, 'G', 8, 64)})
	}
	table.Render()
	return subcommands.ExitSuccess
}

// ScoreCmd is the command for Score.
type ScoreCmd struct {
	extra string
}

// Name returns the name of the ScoreCmd.
func (*ScoreCmd) Name() string { return "lm.score" }

// Synopsis returns the synopsis of ScoreCmd.
func (*ScoreCmd) Synopsis() string { return "score a prefix and suffix against a given model" }

// Usage returns the full usage of ScoreCmd.
func (*ScoreCmd) Usage() string {
	return `score ModelID prefix and suffix:
	Scores a prefix and suffix using a published language model.
`
}

// SetFlags sets flags for ScoreCmd.
func (c *ScoreCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Score().")
}

// Execute executes ScoreCmd.
func (c *ScoreCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) < 3 {
		log.Errorf("Provide model and prefix/suffix for score")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	lm := m.LM()
	if lm == nil {
		log.Errorf("Failed to create language model: %v", err)
		return subcommands.ExitFailure
	}

	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()

	suffixes := f.Args()[2:]
	logPs, err := lm.Score(ctx, f.Args()[1], suffixes, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to score prefix/suffix: %v", err)
		return subcommands.ExitFailure
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Suffix", "Score"})
	for index, suffix := range suffixes {
		table.Append([]string{suffix, strconv.FormatFloat(logPs[index], 'G', 8, 64)})
	}
	table.Render()
	return subcommands.ExitSuccess
}

// EmbedTextCmd is the command for embedding text.
type EmbedTextCmd struct {
	extra string
}

// Name returns the name of EmbedTextCmd.
func (*EmbedTextCmd) Name() string { return "lm.embed" }

// Synopsis returns the synopsis of EmbedTextCmd.
func (*EmbedTextCmd) Synopsis() string { return "embed a text against a given model" }

// Usage returns the full usage of EmbedTextCmd.
func (*EmbedTextCmd) Usage() string {
	return ` Embeds a text input using a published language model.
`
}

// SetFlags sets flags for EmbedTextCmd.
func (c *EmbedTextCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Embed().")
}

// Execute executes EmbedTextCmd.
func (c *EmbedTextCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 {
		log.Errorf("Provide model and text for embed")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	lm := m.LM()
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	text := f.Args()[1]

	results, err := lm.Embed(ctx, text, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to embed text (%s) due to %v", text, err)
		return subcommands.ExitFailure
	}

	resultStr := arrayToString(results)
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Text", "Embedding"})
	table.Append([]string{text, resultStr})
	table.Render()
	return subcommands.ExitSuccess
}
