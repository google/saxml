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

package errors_test

import (
	"fmt"
	"testing"

	"saxml/common/errors"
)

var tests = []struct {
	err                error
	adminRetry         bool
	adminPoison        bool
	serverRetry        bool
	serverPoison       bool
	joinRetry          bool
	isDeadlineExceeded bool
	isNotFound         bool
}{
	{
		err:                fmt.Errorf("%w", errors.ErrUnavailable),
		adminRetry:         true,
		adminPoison:        true,
		serverRetry:        true,
		serverPoison:       true,
		joinRetry:          false,
		isDeadlineExceeded: false,
		isNotFound:         false,
	},
	{
		err:                fmt.Errorf("%w", errors.ErrResourceExhausted),
		adminRetry:         true,
		adminPoison:        false,
		serverRetry:        true,
		serverPoison:       true,
		joinRetry:          false,
		isDeadlineExceeded: false,
		isNotFound:         false,
	},
	{
		err:                fmt.Errorf("%w", errors.ErrNotFound),
		adminRetry:         false,
		adminPoison:        false,
		serverRetry:        true,
		serverPoison:       true,
		joinRetry:          false,
		isDeadlineExceeded: false,
		isNotFound:         true,
	},
	{
		err:                fmt.Errorf("%w", errors.ErrDeadlineExceeded),
		adminRetry:         false,
		adminPoison:        false,
		serverRetry:        false,
		serverPoison:       false,
		joinRetry:          true,
		isDeadlineExceeded: true,
		isNotFound:         false,
	},
	{
		err:                fmt.Errorf("%w", errors.ErrCanceled),
		adminRetry:         false,
		adminPoison:        false,
		serverRetry:        false,
		serverPoison:       false,
		joinRetry:          true,
		isDeadlineExceeded: false,
		isNotFound:         false,
	},
	{
		err:                fmt.Errorf("%w", errors.ErrUnknown),
		adminRetry:         false,
		adminPoison:        false,
		serverRetry:        false,
		serverPoison:       false,
		joinRetry:          false,
		isDeadlineExceeded: false,
		isNotFound:         false,
	},
	{
		err:                nil,
		adminRetry:         false,
		adminPoison:        false,
		serverRetry:        false,
		serverPoison:       false,
		joinRetry:          false,
		isDeadlineExceeded: false,
		isNotFound:         false,
	},
}

func TestError(t *testing.T) {
	for _, test := range tests {
		if got := errors.AdminShouldRetry(test.err); got != test.adminRetry {
			t.Errorf("err.AdminShouldRetry(%v) = %v, want %v", test.err, got, test.adminRetry)
		}
		if got := errors.AdminShouldPoison(test.err); got != test.adminPoison {
			t.Errorf("err.AdminShouldPoison(%v) = %v, want %v", test.err, got, test.adminPoison)
		}
		if got := errors.ServerShouldRetry(test.err); got != test.serverRetry {
			t.Errorf("err.ServerShouldRetry(%v) = %v, want %v", test.err, got, test.serverRetry)
		}
		if got := errors.ServerShouldPoison(test.err); got != test.serverPoison {
			t.Errorf("err.ServerShouldPoison(%v) = %v, want %v", test.err, got, test.serverPoison)
		}
		if got := errors.JoinShouldRetry(test.err); got != test.joinRetry {
			t.Errorf("err.JoinShouldRetry(%v) = %v, want %v", test.err, got, test.joinRetry)
		}
		if got := errors.IsDeadlineExceeded(test.err); got != test.isDeadlineExceeded {
			t.Errorf("err.IsDeadlineExceeded(%v) = %v, want %v", test.err, got, test.isDeadlineExceeded)
		}
		if got := errors.IsNotFound(test.err); got != test.isNotFound {
			t.Errorf("err.IsNotFound(%v) = %v, want %v", test.err, got, test.isNotFound)
		}
	}
}
