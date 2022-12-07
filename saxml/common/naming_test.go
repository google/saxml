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

package naming

import (
	"testing"
)

func TestCheckNames(t *testing.T) {
	tests := []struct {
		name    string
		checker func(string) error
		in      string
		valid   bool
	}{
		{"CellNormal", validateCellName, "foo", true},
		{"CellAllowUnderscoreHyphen", validateCellName, "pi_3-14", true},
		{"CellNoBeginWithUnderscore", validateCellName, "_foo", false},
		{"CellNoBeginWithDigit", validateCellName, "42", false},
		{"CellNoSpecialSymbol", validateCellName, "foo$", false},
		{"ModelNormal", validateModelName, "foo", true},
		{"ModelAllowUnderscoreHyphen", validateModelName, "pi_3-14", true},
		{"ModelNoBeginWithUnderscore", validateModelName, "_foo", false},
		{"ModelNoBeginWithDigit", validateModelName, "42", false},
		{"ModelNoSpecialSymbol", validateModelName, "foo$", false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := test.checker(test.in)
			if test.valid {
				if err != nil {
					t.Errorf("%v(%v) error %v, want no error", test.name, test.in, err)
				}
			} else {
				if err == nil {
					t.Errorf("%v(%v) no error, want an error", test.name, test.in)
				}
			}
		})
	}
}

func TestNewIDs(t *testing.T) {
	modelIDToName := func(id string) (string, error) {
		mid, err := NewModelFullName(id)
		if err != nil {
			return "", err
		}
		return mid.ModelName(), nil
	}
	cellIDToName := func(id string) (string, error) {
		cid, err := NewCellFullName(id)
		if err != nil {
			return "", err
		}
		return cid.CellName(), nil
	}
	tests := []struct {
		name  string
		newer func(string) (string, error)
		in    string
		valid bool
		want  string
	}{
		{"CellNormal", cellIDToName, "/sax/foo", true, "foo"},
		{"CellWrongPrefix", cellIDToName, "/Sax/foo", false, ""},
		{"CellExtraTrailingSlash", cellIDToName, "/sax/foo/", false, ""},
		{"CellMissingLeadingSlash", cellIDToName, "sax/foo", false, ""},
		{"CellTooManyComponents", cellIDToName, "/sax/foo/bar", false, ""},
		{"ModelNormal", modelIDToName, "/sax/foo/bar", true, "bar"},
		{"ModelWrongPrefix", modelIDToName, "/Sax/foo/bar", false, ""},
		{"ModelExtraTrailingSlash", modelIDToName, "/sax/foo/bar/", false, ""},
		{"ModelMissingLeadingSlash", modelIDToName, "sax/foo/bar", false, ""},
		{"ModelTooManyComponents", modelIDToName, "/sax/foo/bar/foobar", false, ""},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.newer(test.in)
			if test.valid {
				if err != nil {
					t.Errorf("%v(%v) error %v, want no error", test.name, test.in, err)
				} else if got != test.want {
					t.Errorf("%v(%v) got %v, want %v", test.name, test.in, got, test.want)
				}
			} else {
				if err == nil {
					t.Errorf("%v(%v) no error, want an error", test.name, test.in)
				}
			}
		})
	}
}

func TestSaxCellToCell(t *testing.T) {
	saxCell := "/sax/foo"
	want := "foo"
	if got, err := SaxCellToCell(saxCell); err != nil {
		t.Errorf("SaxCellToCell(%v) error %v, want no error", saxCell, err)
	} else if got != want {
		t.Errorf("SaxCellToCell(%v) = %v, want %v", saxCell, got, want)
	}

	saxCell = "/sax/foo/"
	if _, err := SaxCellToCell(saxCell); err == nil {
		t.Errorf("SaxCellToCell(%v) no error, want an error", saxCell)
	}
}
