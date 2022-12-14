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

// Package retrier provides a retry function.
package retrier

import (
	"context"
	"fmt"
	"time"

	log "github.com/golang/glog"
	"github.com/cenkalti/backoff"
)

// Closure returns an error.
type Closure func() error

// IsRetriable returns true iff the error is retriable.
type IsRetriable func(error) bool

// queryRetrier manages retry logic.
type queryRetrier struct {
	retryCount int // For logging purpose.
}

func (q queryRetrier) Do(ctx context.Context, query Closure, retriable IsRetriable) error {
	withRetryCheck := func() error {
		err := query()

		// No error; don't retry.
		if err == nil {
			return nil
		}

		// Retry on recoverable errors.
		if retriable(err) {
			log.V(1).Infof("Retriable Error on attempt %d: %s", q.retryCount, err)
			q.retryCount++
			return fmt.Errorf("retries(%d): %w", q.retryCount, err)
		}

		// Don't retry on any other type of error, by marking it as PermanentError.
		log.V(1).Infof("Non-Retriable Error on attempt %d: %s", q.retryCount, err)
		return backoff.Permanent(err)
	}
	// Retries with exponential backoff.
	opts := backoff.NewExponentialBackOff()
	// Fast initial retries.
	opts.InitialInterval = 10 * time.Millisecond
	// Because the retry loop still respects ctx's deadline, we want
	// the loop retries as many times as necessary.
	opts.MaxElapsedTime = 0
	err := backoff.Retry(withRetryCheck, backoff.WithContext(opts, ctx))

	// Check if canceled or deadline exceeded.
	//
	// backoff.Retry has a "bug" that it returns the last retry error
	// before ctx deadline/cancellation. To make the error clearer, we
	// explicitly check ctx.Err() and prefer returnning ctx.Err()
	// instead.
	if ctx.Err() != nil {
		return ctx.Err()
	}

	return err
}

// Do executes query with retries until either success or a permanent
// error.  It retries the query using exponential backoffs until
// reaching deadline on the context. If an error is not retrieable,
// the error is considered a permanent error.
func Do(ctx context.Context, query Closure, retriable IsRetriable) error {
	return queryRetrier{retryCount: 0}.Do(ctx, query, retriable)
}

// CreatePermanentError creates permanent error so client code can inform retrier explicitly.
func CreatePermanentError(err error) error {
	return backoff.Permanent(err)
}
