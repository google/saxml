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
	"bytes"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"flag"
	log "github.com/golang/glog"
	"saxml/client/go/sax"
)

var cmdTimeout = flag.Duration(
	"sax_timeout",
	60*time.Second,
	"Command timeout. Defaults to \"60s\". See https://pkg.go.dev/time#ParseDuration for format.",
)

func writesImagesToDir(images []sax.GeneratedImage, outputDir string, imagePath []string) error {
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		os.MkdirAll(outputDir, 0755)
	}
	for index, one := range images {
		err := os.WriteFile(imagePath[index], one.Image, 0644)
		if err != nil {
			// Won't display image content since images can be big.
			return fmt.Errorf("writesImagesToDir had %v saving %dth image", err, index)
		}
	}
	return nil
}

func readStdin() []byte {
	stdin, err := io.ReadAll(os.Stdin)
	if err != nil {
		log.Errorf("Failed to read stdin %v", err)
		return nil
	}
	return stdin
}

func readFile(path string) []byte {
	// No need to use bufio.
	content, err := os.ReadFile(path)
	if err != nil {
		log.Errorf("Failed to read file (%s) due to %v", path, err)
		return nil
	}
	return content
}

// Turns a array of float64 into a string with good precision.
func arrayToString(arr []float64) string {
	var buffer bytes.Buffer
	for idx, val := range arr {
		buffer.WriteString(formatFloat(val))
		if idx != len(arr)-1 {
			buffer.WriteString(", ")
		}
	}
	return buffer.String()
}

// Turns a array of float64 into a string with good precision, up to a certain number of values.
func arrayToStringWithLimit(arr []float64, limit int) string {
	if len(arr) > limit {
		return arrayToString(arr[:limit]) + ", ..."
	}
	return arrayToString(arr)
}

func formatFloat(val float64) string {
	return strconv.FormatFloat(val, 'G', 8, 64)
}

// ExtraInputs creates a list of options setters from a string in the form of "a:0.5,b:1.2".
func ExtraInputs(extra string) []sax.ModelOptionSetter {
	extraFields := strings.Split(extra, ",")
	options := []sax.ModelOptionSetter{}
	for _, option := range extraFields {
		kv := strings.Split(option, ":")
		if len(kv) != 2 {
			log.V(1).Infof("Cannot get k-v pair by splitting %s with ':'\n", option)
			continue
		}

		value, err := strconv.ParseFloat(kv[1], 32)
		if err != nil {
			log.V(1).Infof("Cannot parse value for %s\n", kv[1])
			continue
		}
		options = append(options, sax.WithExtraInput(kv[0], float32(value)))
	}
	return options
}
