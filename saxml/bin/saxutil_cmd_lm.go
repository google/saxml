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
	"os/exec"
	"sort"
	"strings"

	"flag"
	log "github.com/golang/glog"
	"github.com/google/subcommands"
	"github.com/olekukonko/tablewriter"
	"saxml/client/go/sax"
)

var clear = func() {
	cmd := exec.Command("clear")
	cmd.Stdout = os.Stdout
	cmd.Run()
}

// GenerateCmd is the command for generate
type GenerateCmd struct {
	extra      string
	proxy      string
	stream     bool
	terse      bool
	maxOutputs int
	raw        bool
}

// Name returns the name of GenerateCmd.
func (*GenerateCmd) Name() string { return "lm.generate" }

// Synopsis returns the synopsis of GenerateCmd.
func (*GenerateCmd) Synopsis() string { return "Generate a text against a given model." }

// Usage returns the full usage of GenerateCmd.
func (*GenerateCmd) Usage() string {
	return `Generates a text input using a published language model.
	* Support extra inputs such as temperature through: -extra="temperature:0.2".
	* Support streaming through -stream.
	* Set maximum number of outputs with -n (not compatible with -stream).
	* Output only generated text(s) with -terse (non-tabular).
    * If <Query> is '-', reads the query from the stdin.

Usage: saxutil lm.generate [-n=<N>] [-stream] [-terse] [-extra=<extra>] <ModelID> <Query>
`
}

// SetFlags sets the flags for GenerateCmd.
func (c *GenerateCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Generate().")
	f.StringVar(&c.proxy, "proxy", "", "SAX Proxy address, e.g., sax.server.lm.lmservice-prod.blade.gslb.googleprod.com")
	f.BoolVar(&c.stream, "stream", false, "stream responses")
	f.BoolVar(&c.terse, "terse", false, "print generated texts one line per result, descending by score")
	f.IntVar(&c.maxOutputs, "n", 0, "maximum number of generated texts to output or zero for all")
}

func (c *GenerateCmd) streamingGenerate(ctx context.Context, query string, lm *sax.LanguageModel) subcommands.ExitStatus {
	chanStreamResults := lm.GenerateStream(ctx, query, ExtraInputs(c.extra)...)
	var accumulatedResults []string
	var allScores [][]float64
	var lastScore [][]float64
	for {
		select {
		case <-ctx.Done():
			fmt.Printf("Command canceled: %v\n", ctx.Err())
			return subcommands.ExitFailure
		case streamResult := <-chanStreamResults:
			switch streamResult.Err {
			case nil:
				// Grow accumulatedResults and scores to accommodate all returned suffixes.
				for len(accumulatedResults) < len(streamResult.Items) {
					accumulatedResults = append(accumulatedResults, "")
				}
				for len(allScores) < len(streamResult.Items) {
					s := make([]float64, len(streamResult.Items[0].Scores))
					allScores = append(allScores, s)
					lastScore = append(lastScore, s)
				}
				clear()
				for i, item := range streamResult.Items {
					if len(accumulatedResults[i]) < item.PrefixLen {
						fmt.Printf("PrefixLen %v exceeds the current result %v\n", item.PrefixLen, accumulatedResults[i])
						return subcommands.ExitFailure
					}
					accumulatedResults[i] = accumulatedResults[i][:item.PrefixLen] + item.Text
					// Accumulate the scores.
					for j := range item.Scores {
						allScores[i][j] += item.Scores[j]
					}
					lastScore[i] = item.Scores

					// Print all suffixes separated by one blank line.
					fmt.Println(accumulatedResults[i])
					fmt.Println()
				}
			case io.EOF:
				if len(allScores) == 0 {
					continue
				}
				for i := range allScores[0] {
					var strs []string
					for _, scores := range allScores {
						if len(scores) > 0 {
							strs = append(strs, fmt.Sprintf("%v", scores[i]))
						}
					}
					if len(allScores[0]) == 1 {
						fmt.Print("Scores: ")
					} else {
						fmt.Printf("Total Scores[%v]: ", i)
					}
					fmt.Println(strings.Join(strs, ", "))
				}
				// In some cases, users may want the last score instead.
				if len(lastScore[0]) > 1 {
					fmt.Println()
					for i := range lastScore[0] {
						var strs []string
						for _, scores := range lastScore {
							if len(scores) > 0 {
								strs = append(strs, fmt.Sprintf("%v", scores[i]))
							}
						}
						fmt.Printf("Last Score[%v]: ", i)
						fmt.Println(strings.Join(strs, ", "))
					}
				}
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

	m, err := sax.Open(f.Args()[0], sax.WithProxy(c.proxy))
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	lm := m.LM()
	if lm == nil {
		log.Errorf("Failed to create language model: %v", err)
		return subcommands.ExitFailure
	}

	query := f.Args()[1]
	if query == "-" {
		query = string(readStdin())
	}

	ctx, cancel := context.WithTimeout(ctx, *cmdTimeout)
	defer cancel()

	// Streaming generate.
	if c.stream {
		if c.terse {
			log.Error("Terse output incompatible with streaming (-stream).")
			return subcommands.ExitFailure
		}
		if c.maxOutputs > 0 {
			log.Error("Max outputs (-n) incompatible with streaming (-stream).")
			return subcommands.ExitFailure
		}
		return c.streamingGenerate(ctx, query, lm)
	}

	// Non-streaming generate.
	generates, err := lm.Generate(ctx, query, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to generate query: %v", err)
		return subcommands.ExitFailure
	}

	if c.terse {
		sort.Slice(generates, func(i, j int) bool {
			return generates[i].Score > generates[j].Score
		})
		for i, generate := range generates {
			if c.maxOutputs > 0 && i == c.maxOutputs {
				break
			}
			fmt.Println(generate.Text)
		}
		return subcommands.ExitSuccess
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"generate", "Score"})
	for i, generate := range generates {
		if c.maxOutputs > 0 && i == c.maxOutputs {
			break
		}
		table.Append([]string{generate.Text, formatFloat(generate.Score)})
	}
	table.Render()
	return subcommands.ExitSuccess
}

// ScoreCmd is the command for Score.
type ScoreCmd struct {
	extra string
	proxy string
}

// Name returns the name of the ScoreCmd.
func (*ScoreCmd) Name() string { return "lm.score" }

// Synopsis returns the synopsis of ScoreCmd.
func (*ScoreCmd) Synopsis() string { return "Score a prefix and suffix against a given model." }

// Usage returns the full usage of ScoreCmd.
func (*ScoreCmd) Usage() string {
	return `score ModelID prefix and suffix:
	Scores a prefix and suffix using a published language model.
`
}

// SetFlags sets flags for ScoreCmd.
func (c *ScoreCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Score().")
	f.StringVar(&c.proxy, "proxy", "", "SAX Proxy address, e.g., sax.server.lm.lmservice-prod.blade.gslb.googleprod.com")
}

// Execute executes ScoreCmd.
func (c *ScoreCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) < 3 {
		log.Errorf("Provide model and prefix/suffix for score")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0], sax.WithProxy(c.proxy))
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	lm := m.LM()
	if lm == nil {
		log.Errorf("Failed to create language model: %v", err)
		return subcommands.ExitFailure
	}

	ctx, cancel := context.WithTimeout(ctx, *cmdTimeout)
	defer cancel()

	suffixes := f.Args()[2:]
	logPs, err := lm.Score(ctx, f.Args()[1], suffixes, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to score prefix/suffix: %v", err)
		return subcommands.ExitFailure
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Suffix", "Score"})
	logPsPerSuffix := len(logPs) / len(suffixes)
	for index, suffix := range suffixes {
		formattedLogPs := make([]string, len(logPs))
		for i, logP := range logPs[index*logPsPerSuffix : (index+1)*logPsPerSuffix] {
			formattedLogPs[i] = formatFloat(logP)
		}
		table.Append(append([]string{suffix}, formattedLogPs...))
	}
	table.Render()
	return subcommands.ExitSuccess
}

// EmbedTextCmd is the command for embedding text.
type EmbedTextCmd struct {
	extra string
	proxy string
}

// Name returns the name of EmbedTextCmd.
func (*EmbedTextCmd) Name() string { return "lm.embed" }

// Synopsis returns the synopsis of EmbedTextCmd.
func (*EmbedTextCmd) Synopsis() string { return "Embed a text against a given model." }

// Usage returns the full usage of EmbedTextCmd.
func (*EmbedTextCmd) Usage() string {
	return ` Embeds a text input using a published language model.
`
}

// SetFlags sets flags for EmbedTextCmd.
func (c *EmbedTextCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "Extra arguments for Embed().")
	f.StringVar(&c.proxy, "proxy", "", "Sax Proxy address, e.g., sax.server.lm.lmservice-prod.blade.gslb.googleprod.com")
}

// Execute executes EmbedTextCmd.
func (c *EmbedTextCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 {
		log.Errorf("Provide model and text for embed")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0], sax.WithProxy(c.proxy))
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	lm := m.LM()
	ctx, cancel := context.WithTimeout(ctx, *cmdTimeout)
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

// GradientCmd is the command for Gradient.
type GradientCmd struct {
	extra string
	proxy string
}

// Name returns the name of the ScoreCmd.
func (*GradientCmd) Name() string { return "lm.gradient" }

// Synopsis returns the synopsis of ScoreCmd.
func (*GradientCmd) Synopsis() string {
	return "Generate gradient of a pair of prefix and suffix against a given model."
}

// Usage returns the full usage of ScoreCmd.
func (*GradientCmd) Usage() string {
	return `gradient ModelID prefix suffix:
	Gradient of a pair of prefix and suffix using a published language model.
`
}

// SetFlags sets flags for ScoreCmd.
func (c *GradientCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Gradient().")
	f.StringVar(&c.proxy, "proxy", "", "SAX Proxy address, e.g., sax.server.lm.lmservice-prod.blade.gslb.googleprod.com")
}

// Execute executes ScoreCmd.
func (c *GradientCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 3 {
		log.Errorf("Provide model and prefix/suffix for gradient")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0], sax.WithProxy(c.proxy))
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	lm := m.LM()
	if lm == nil {
		log.Errorf("Failed to create language model: %v", err)
		return subcommands.ExitFailure
	}

	ctx, cancel := context.WithTimeout(ctx, *cmdTimeout)
	defer cancel()

	scores, gradients, err := lm.Gradient(ctx, f.Args()[1], f.Args()[2], ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to get the gradient of prefix/suffix: %v", err)
		return subcommands.ExitFailure
	}

	table := tablewriter.NewWriter(os.Stdout)
	header := make([]string, len(gradients)+1)
	row := make([]string, len(gradients)+1)
	header[0] = "Score"
	row[0] = arrayToString(scores)
	i := 1
	for k, v := range gradients {
		header[i] = k
		row[i] = arrayToStringWithLimit(v, 8)
		i++
	}
	table.SetHeader(header)
	table.Append(row)
	table.Render()
	return subcommands.ExitSuccess
}
