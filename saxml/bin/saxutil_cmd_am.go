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
	"os"
	"strconv"

	"flag"
	log "github.com/golang/glog"
	"github.com/google/subcommands"
	"github.com/olekukonko/tablewriter"
	"saxml/client/go/sax"
)

// RecognizeCmd is the command for Asr.
type RecognizeCmd struct{ extra string }

// Name returns the name of SpeechRecognitionCmd.
func (*RecognizeCmd) Name() string { return "am.recognize" }

// Synopsis returns the synopsis of AudioToTextCmd.
func (*RecognizeCmd) Synopsis() string { return "transcribe an audio" }

// Usage returns the full usage of AudioToTextCmd.
func (*RecognizeCmd) Usage() string {
	return `speech_recognition modelID wave_filename

	Transcribe the audio specified by wav_filename.

	For example:
	$ saxutil am.speech_recognition /sax/bar/librispeech_ctc_l googledata/speech/greco3/data/nonlogs/en_us/dictation/55.wav
`
}

// SetFlags sets flags for AudioToTextCmd.
func (c *RecognizeCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Recognize().")
}

// Execute executes AudioToTextCmd.
func (c *RecognizeCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 {
		log.Errorf("Provide model and audio path.")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	am := m.AM()
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	filePath := f.Args()[1]
	var contents []byte
	contents = readFile(filePath)
	results, err := am.Recognize(ctx, contents, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to transrbie audio (%s) due to %v", filePath, err)
		return subcommands.ExitFailure
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Text", "Score"})
	for _, res := range results {
		table.Append([]string{res.Text, strconv.FormatFloat(res.Score, 'G', 8, 64)})
	}
	table.Render()
	return subcommands.ExitSuccess
}
