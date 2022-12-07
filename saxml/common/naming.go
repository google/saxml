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

// Package naming provides functions to manipulate identifiers.
//
// A SAX model is uniquely identified by its full name, also called a model's ID:
//
//	SAX model full name := /sax/<cell name>/<model name>
//
// The first two components of a model full name form a SAX cell full name, also called a cell's ID:
//
//	SAX cell full name := /sax/<cell name>
//
// By convention, a string variable called `saxCell` usually contains the cell full name while
// `cell` contains just the cell name.
//
// A SAX model key is composed of a SAX model full name and a generation number. It's usually
// not visible to users and is only used between the admin server and the model server to
// distinguish models with the same full name but different metadata (e.g., republish or update
// a model using different checkpoints).
package naming

import (
	"fmt"
	"regexp"
	"strings"

	"saxml/common/errors"
)

// This package is structured in a nested way: ModelFullName contains CellFullName,
// which (conceptually) contains Prefix. This is to avoid duplicating logic and have one place to
// change any specific definition.

const (
	// Prefix is required as the first part of any model and cell full names.
	Prefix = "sax"

	// Valid name pattern.
	pattern = "^[a-z]+[a-z0-9-_]*$"
)

var validName = regexp.MustCompile(pattern)

// T is the minimal viable interface that captures testing.T, defined to avoid depending on testing.
type T interface {
	// Helper corresponds to testing.T.Helper.
	Helper()
	// Helper corresponds to testing.T.Fatalf.
	Fatalf(string, ...any)
}

// Sax returns the SAX namespace, i.e., /sax.
func Sax() string {
	return "/" + Prefix
}

// CellFullName uniquely identifies a SAX cell. It has the form /sax/<cell name>.
//
// Only the cell name is stored. The prefix is implied and defined above as `Prefix`.
type CellFullName string // e.g. test

// CellName returns the cell name in a cell full name.
func (c CellFullName) CellName() string {
	return string(c)
}

// CellFullName returns the fully qualified, canonical name representation of a cell.
func (c CellFullName) CellFullName() string {
	return "/" + Prefix + "/" + string(c)
}

func validateCellName(cell string) error {
	if validName.MatchString(cell) {
		return nil
	}
	return fmt.Errorf("cell name %q must match %q: %w", cell, pattern, errors.ErrInvalidArgument)
}

func splitCellFullName(saxCell string) (prefix, cell string, err error) {
	parts := strings.Split(saxCell, "/")
	if len(parts) != 3 || parts[0] != "" || parts[1] != Prefix {
		return "", "", fmt.Errorf("cell full name %q should have the form /%v/<cell name>: %w",
			saxCell, Prefix, errors.ErrInvalidArgument)
	}

	prefix, cell = parts[1], parts[2]
	if err := validateCellName(cell); err != nil {
		return "", "", err
	}
	return prefix, cell, nil
}

// NewCellFullName creates a CellFullName value from a string representation.
func NewCellFullName(saxCell string) (CellFullName, error) {
	_, cell, err := splitCellFullName(saxCell)
	if err != nil {
		return "", err
	}
	return CellFullName(cell), nil
}

// NewCellFullNameT creates a CellFullName value from a cell name for tests.
func NewCellFullNameT(t T, cell string) CellFullName {
	t.Helper()
	if err := validateCellName(cell); err != nil {
		t.Fatalf("NewCellFullNameT(%v) error: %v", cell, err)
	}
	return CellFullName(cell)
}

// SaxCellToCell returns the cell name in a cell full name, e.g. "foo" from "/sax/foo".
func SaxCellToCell(saxCell string) (string, error) {
	cell, err := NewCellFullName(saxCell)
	if err != nil {
		return "", err
	}
	return cell.CellName(), nil
}

// ModelFullName uniquely identifies a SAX model. It has the form /sax/<cell name>/<model name>.
type ModelFullName struct {
	// Prefix is implied and contained in `cell`.
	cell  CellFullName // e.g. /sax/test
	model string       // e.g. glam_64b64e
}

// ModelName returns the model name in a model full name.
func (m ModelFullName) ModelName() string {
	return m.model
}

// ModelFullName returns the fully qualified, canonical name representation of a model.
func (m ModelFullName) ModelFullName() string {
	return m.cell.CellFullName() + "/" + m.model
}

// CellName returns the cell name in a model full name.
func (m ModelFullName) CellName() string {
	return m.cell.CellName()
}

// CellFullName returns the cell full name in a model full name.
func (m ModelFullName) CellFullName() string {
	return m.cell.CellFullName()
}

func validateModelName(model string) error {
	if validName.MatchString(model) {
		return nil
	}
	return fmt.Errorf("model name %q must match %q: %w", model, pattern, errors.ErrInvalidArgument)
}

func splitModelFullName(fullName string) (cell CellFullName, model string, err error) {
	// Find the last "/".
	lastIdx := strings.LastIndex(fullName, "/")
	if lastIdx < 0 {
		return "", "", fmt.Errorf("model full name %q should have the form /%v/<cell name>/<model name>: %w",
			fullName, Prefix, errors.ErrInvalidArgument)
	}

	saxCell, model := fullName[:lastIdx], fullName[lastIdx+1:]
	// Check what follows the last "/" is a valid model name.
	if err := validateModelName(model); err != nil {
		return "", "", err
	}
	// Let NewCellID check what precedes the last "/".
	cell, err = NewCellFullName(saxCell)
	if err != nil {
		return "", "", err
	}
	return cell, model, nil
}

// NewModelFullName creates a ModelFullName value from a string representation.
func NewModelFullName(fullName string) (ModelFullName, error) {
	cell, model, err := splitModelFullName(fullName)
	if err != nil {
		return ModelFullName{}, err
	}
	return ModelFullName{cell, model}, nil
}

// NewModelFullNameT creates a ModelFullName value from cell and model names for tests.
func NewModelFullNameT(t T, cell, model string) ModelFullName {
	t.Helper()
	cellFullName := NewCellFullNameT(t, cell)
	if err := validateModelName(model); err != nil {
		t.Fatalf("NewModelFullNameT(%v, %v) error: %v", cell, model, err)
	}
	return ModelFullName{cellFullName, model}
}

// ValidateModelFullName returns nil iff the given model full name is valid.
func ValidateModelFullName(fullName string) error {
	_, err := NewModelFullName(fullName)
	return err
}

// ModelFullNameToName returns the model name in a model full name, e.g. "bar" from "/sax/foo/bar".
func ModelFullNameToName(fullName string) (string, error) {
	model, err := NewModelFullName(fullName)
	if err != nil {
		return "", err
	}
	return model.ModelName(), nil
}

// ModelFullNameToNameForDebugging returns the model name in a model full name, ignoring errors.
func ModelFullNameToNameForDebugging(fullName string) string {
	model, err := ModelFullNameToName(fullName)
	if err != nil {
		return "UNKNOWN"
	}
	return model
}
