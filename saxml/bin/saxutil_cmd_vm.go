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
	"os"
	"path"

	"flag"
	log "github.com/golang/glog"
	"github.com/google/subcommands"
	"github.com/olekukonko/tablewriter"
	"saxml/client/go/sax"
)

// ClassifyCmd is the command for Classify.
type ClassifyCmd struct{ extra string }

// Name returns the name of ClassifyCmd.
func (*ClassifyCmd) Name() string { return "vm.classify" }

// Synopsis returns the synopsis of ClassifyCmd.
func (*ClassifyCmd) Synopsis() string { return "classify an image against a given model" }

// Usage returns the full usage of ClassifyCmd.
func (*ClassifyCmd) Usage() string {
	return `classify modelID image_filename

	Classifies the image specified by image_filename. image_filename can be a jpg, gif, png file. If image_filename is '-', reads the image content from the stdin.

	For example:
	# Classify an jpeg image.
	$ saxutil vm.classify /sax/bar/resnet50 /tmp/lenna.jpg

	# Flop the image and then ask the model to classify it.
	$ convert -flop /tmp/lenna.jpg | saxutil vm.classify /sax/bar/resnet50 -
`
}

// SetFlags sets flags for ClassifyCmd.
func (c *ClassifyCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Classify().")
}

// Execute executes ClassifyCmd.
func (c *ClassifyCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 {
		log.Errorf("Provide model and image path for classify")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	vm := m.VM()
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	imagePath := f.Args()[1]
	var contents []byte
	if imagePath == "-" {
		contents = readStdin()
	} else {
		contents = readFile(imagePath)
	}

	results, err := vm.Classify(ctx, contents, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to classify image (%s) due to %v", imagePath, err)
		return subcommands.ExitFailure
	}

	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Text", "Score"})
	for _, res := range results {
		table.Append([]string{res.Text, formatFloat(res.Score)})
	}
	table.Render()
	return subcommands.ExitSuccess
}

// TextToImageCmd is the command for TextToImage.
type TextToImageCmd struct{ extra string }

// Name returns the name for TextToImageCmd.
func (*TextToImageCmd) Name() string { return "vm.generate" }

// Synopsis returns the synopsis for TextToImageCmd.
func (*TextToImageCmd) Synopsis() string {
	return "generate list of images and scores for the input text"
}

// Usage returns the full usage of TextToImageCmd.
func (*TextToImageCmd) Usage() string {
	return `vm.generate modelID text output_dir

	Generates a list of images and associated scores for given text.
	[1] if output_dir is set to regular path (e.g. /tmp/xyz)
	    $ saxutil vm.generate /sax/bar/parti dog /tmp
			will show image paths and score:
				+------------+-------+
        |   IMAGE    | SCORE |
        +------------+-------+
        | /tmp/1.PNG |   0.9 |
        | /tmp/2.PNG |   1.2 |
        +------------+-------+
			and images will be saved in /tmp/ as:
				1.PNG
				2.PNG
				.....

  [2] if output_dir is set to "-", vm.generate will pipe images to stdin, and one can directly check images.
		  $ saxutil vm.generate /sax/bar/parti dog - | display
`
}

// SetFlags sets flags for TextToImageCmd.
func (c *TextToImageCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for TextToImage().")
}

// Execute executes TextToImageCmd.
func (c *TextToImageCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 3 {
		log.Errorf("Provide model ID, text and output directory for text_to_image")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	vm := m.VM()
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	text := f.Args()[1]

	results, err := vm.TextToImage(ctx, text, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to generate images for (%s) due to [%v]", text, err)
		return subcommands.ExitFailure
	}

	outputDir := f.Args()[2]
	if outputDir == "-" {
		// Output to stdout.
		for _, one := range results {
			os.Stdout.Write(one.Image)
		}
	} else {
		// Output to disk.
		table := tablewriter.NewWriter(os.Stdout)
		table.SetHeader([]string{"Image", "Score"})
		fileNames := make([]string, 0, len(results))
		for index, one := range results {
			// Assumes TextToImage model produces PNG images.
			// Image name starts with index 1.
			imagePath := path.Join(outputDir, fmt.Sprintf("%04d.PNG", index+1))
			table.Append([]string{imagePath, formatFloat(one.Logp)})
			fileNames = append(fileNames, imagePath)
		}
		table.Render()

		if err := writesImagesToDir(results, outputDir, fileNames); err != nil {
			log.Errorf("Failed to save generated images for (%s) to (%s) due to [%v]", text, outputDir, err)
			return subcommands.ExitFailure
		}
	}

	return subcommands.ExitSuccess
}

// EmbedImageCmd is the command for embedding images.
type EmbedImageCmd struct{ extra string }

// Name returns the name of EmbedImageCmd.
func (*EmbedImageCmd) Name() string { return "vm.embed" }

// Synopsis returns the synopsis of EmbedImageCmd.
func (*EmbedImageCmd) Synopsis() string {
	return "embed an image (as byte array) against a given model"
}

// Usage returns the full usage of EmbedImageCmd.
func (*EmbedImageCmd) Usage() string {
	return `embed modelID image_filename

	Embedding the image (as byte array) specified by image_filename. image_filename can be a jpg, gif, png file. If image_filename is '-', reads the image content from the stdin.

	For example:
	# Embed an jpeg image.
	$ saxutil vm.embed /sax/bar/coca /tmp/lenna.jpg

	# Flop the image and then ask the model to embed it.
	$ convert -flop /tmp/lenna.jpg | saxutil vm.embed /sax/bar/coca -
`
}

// SetFlags sets flags for EmbedImageCmd.
func (c *EmbedImageCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Embed().")
}

// Execute executes EmbedImageCmd.
func (c *EmbedImageCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) != 2 {
		log.Errorf("Provide model and image path for embed")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	vm := m.VM()
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	imagePath := f.Args()[1]
	var contents []byte
	if imagePath == "-" {
		contents = readStdin()
	} else {
		contents = readFile(imagePath)
	}

	results, err := vm.Embed(ctx, contents, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to embed image (%s) due to %v", imagePath, err)
		return subcommands.ExitFailure
	}

	resultStr := arrayToString(results)
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"Image", "Embedding"})
	if imagePath == "-" {
		table.Append([]string{"stdin", resultStr})
	} else {
		table.Append([]string{imagePath, resultStr})
	}
	table.Render()
	return subcommands.ExitSuccess
}

// DetectCmd is the command for detecting objects in images.
type DetectCmd struct{ extra string }

// Name returns the name of DetectCmd.
func (*DetectCmd) Name() string { return "vm.detect" }

// Synopsis returns the synopsis of DetectCmd.
func (*DetectCmd) Synopsis() string {
	return "Detects objects in an image (as byte array) using a given model"
}

// Usage returns the full usage of DetectCmd.
func (*DetectCmd) Usage() string {
	return `detect modelID image_filename [text0 text1 ...]

	Detect objects in the image (as byte array) specified by image_filename. image_filename can be a jpg, gif, png file. If image_filename is '-', reads the image content from the stdin.

	For example:
	# Detect objects in an jpeg image.
	$ saxutil vm.detect /sax/bar/vitmaskrcnn /tmp/lenna.jpg

	# Flop the image and then ask the model to detect objects in it.
	$ convert -flop /tmp/lenna.jpg | saxutil vm.detect /sax/bar/vitmaskrcnn -
`
}

// SetFlags sets flags for DetectCmd.
func (c *DetectCmd) SetFlags(f *flag.FlagSet) {
	f.StringVar(&c.extra, "extra", "", "extra arguments for Detect().")
}

// Execute executes DetectCmd.
func (c *DetectCmd) Execute(ctx context.Context, f *flag.FlagSet, args ...any) subcommands.ExitStatus {
	if len(f.Args()) < 2 {
		log.Errorf("Provide model and image path for detect")
		return subcommands.ExitUsageError
	}

	m, err := sax.Open(f.Args()[0])
	if err != nil {
		log.Errorf("Failed to open model: %v", err)
		return subcommands.ExitFailure
	}
	vm := m.VM()
	ctx, cancel := context.WithTimeout(ctx, cmdTimeout)
	defer cancel()
	imagePath := f.Args()[1]
	var contents []byte
	if imagePath == "-" {
		contents = readStdin()
	} else {
		contents = readFile(imagePath)
	}

	text := f.Args()[2:]
	results, err := vm.Detect(ctx, contents, text, ExtraInputs(c.extra)...)
	if err != nil {
		log.Errorf("Failed to detect objects in image (%s) due to %v", imagePath, err)
		return subcommands.ExitFailure
	}
	table := tablewriter.NewWriter(os.Stdout)
	table.SetHeader([]string{"CenterX", "CenterY", "Width", "Height", "Text", "Score"})
	for _, result := range results {
		table.Append([]string{
			formatFloat(result.CenterX),
			formatFloat(result.CenterY),
			formatFloat(result.Width),
			formatFloat(result.Height),
			result.Text,
			formatFloat(result.Score)})
	}
	table.Render()
	return subcommands.ExitSuccess
}
