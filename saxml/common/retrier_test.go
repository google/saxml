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

package retrier_test

import (
	"context"
	"fmt"
	"testing"
	"time"

	"saxml/common/errors"
	"saxml/common/retrier"
)

type multipleReturn struct {
	errors []error // A list of predefined errors.
	index  int     // The index of the error to return.
}

// query returns a list of predefined errors (including nil) one by one.
func (m *multipleReturn) query() error {
	if m.index < len(m.errors) {
		curr := m.errors[m.index]
		m.index++
		return curr
	}
	return nil
}

// Retry is successful when query returns no error at first try.
func TestDirectSuccess(t *testing.T) {
	nonretriableError := fmt.Errorf("%w", errors.ErrInternal)
	errs := []error{nil, nonretriableError}
	m := multipleReturn{errors: errs, index: 0}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	err := retrier.Do(ctx, m.query, errors.AdminShouldRetry)
	if err != nil {
		t.Fatalf("TestDirectSuccess should return nil error but got %v\n", err)
	}
}

// Retry is successful when query returns no error after 2 retriable errors.
func TestRetrySuccess(t *testing.T) {
	retriableError := fmt.Errorf("%w", errors.ErrResourceExhausted)
	errs := []error{retriableError, retriableError, nil}
	m := multipleReturn{errors: errs, index: 0}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	err := retrier.Do(ctx, m.query, errors.AdminShouldRetry)
	if err != nil {
		t.Fatalf("TestRetrySuccess should return nil error but got %v\n", err)
	}
}

// Retry fails when query times outs after a few retriable errors.
func TestRetryTimeout(t *testing.T) {
	retriableError := fmt.Errorf("%w", errors.ErrResourceExhausted)
	// Initial interval is 0.1s, 1.5 is the multiplier, hence 12
	// retries take 0.1s * (1 - 1.5^12)(/1-1.5) ~= 2.57s.  Considering
	// randomness in the interval time, 15 retries are sufficient.
	errs := []error{}
	for i := 0; i < 15; i++ {
		errs = append(errs, retriableError)
	}
	errs = append(errs, nil)
	m := multipleReturn{errors: errs, index: 0}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	err := retrier.Do(ctx, m.query, errors.AdminShouldRetry)
	if err == nil {
		t.Fatalf("TestRetryTimeout should fail\n")
	}
}

// Retry fails when query returns an error that is not retriable.
func TestDirectFail(t *testing.T) {
	nonretriableError := fmt.Errorf("%w", errors.ErrInternal)
	errs := []error{nonretriableError, nil}
	m := multipleReturn{errors: errs, index: 0}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	err := retrier.Do(ctx, m.query, errors.AdminShouldRetry)
	if err == nil {
		t.Fatalf("TestDirectFail should fail\n")
	}
}
