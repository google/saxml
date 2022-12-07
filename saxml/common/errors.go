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

// Package errors provides canonical errors and helper functions.
package errors

import (
	"context"
	"errors"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Canonical error values.
//
// Sax errors should start with one of these values, borrowed from gRPC.
// Add additional context by using the %w formatting verb to wrap errors.
var (
	ErrCanceled           error = status.Error(codes.Canceled, codes.Canceled.String())
	ErrUnknown            error = status.Error(codes.Unknown, codes.Unknown.String())
	ErrInvalidArgument    error = status.Error(codes.InvalidArgument, codes.InvalidArgument.String())
	ErrDeadlineExceeded   error = status.Error(codes.DeadlineExceeded, codes.DeadlineExceeded.String())
	ErrNotFound           error = status.Error(codes.NotFound, codes.NotFound.String())
	ErrAlreadyExists      error = status.Error(codes.AlreadyExists, codes.AlreadyExists.String())
	ErrPermissionDenied   error = status.Error(codes.PermissionDenied, codes.PermissionDenied.String())
	ErrResourceExhausted  error = status.Error(codes.ResourceExhausted, codes.ResourceExhausted.String())
	ErrFailedPrecondition error = status.Error(codes.FailedPrecondition, codes.FailedPrecondition.String())
	ErrAborted            error = status.Error(codes.Aborted, codes.Aborted.String())
	ErrOutOfRange         error = status.Error(codes.OutOfRange, codes.OutOfRange.String())
	ErrUnimplemented      error = status.Error(codes.Unimplemented, codes.Unimplemented.String())
	ErrInternal           error = status.Error(codes.Internal, codes.Internal.String())
	ErrUnavailable        error = status.Error(codes.Unavailable, codes.Unavailable.String())
	ErrDataLoss           error = status.Error(codes.DataLoss, codes.DataLoss.String())
	ErrUnauthenticated    error = status.Error(codes.Unauthenticated, codes.Unauthenticated.String())
)

// Code returns the code of a saxError, OK for a nil error, or best effort for others.
func Code(err error) codes.Code {
	if err == nil {
		return codes.OK
	}
	for {
		// Check if the error is a gRPC error.
		if s, ok := status.FromError(err); ok {
			return s.Code()
		}
		// Unwrap the error until the innermost level, while retrying the checks above.
		u := errors.Unwrap(err)
		if u != nil {
			err = u
			continue
		}
		// Try converting two context-related standard errors into a gRPC error code.
		switch err {
		case context.Canceled:
			return codes.Canceled
		case context.DeadlineExceeded:
			return codes.DeadlineExceeded
		}
		return codes.Unknown
	}
}

func isOneOf(err error, codes []codes.Code) bool {
	c := Code(err)
	for _, target := range codes {
		if c == target {
			return true
		}
	}
	return false
}

var (
	adminRetryCodes = []codes.Code{
		codes.Unavailable,       // admin unreachable
		codes.ResourceExhausted, // admin busy
	}
	adminPoisonCodes = []codes.Code{
		codes.Unavailable, // another admin instance may take over
	}
	serverRetryCodes = []codes.Code{
		codes.Unavailable,       // server unreachable
		codes.ResourceExhausted, // server busy
		codes.NotFound,          // server does not have the desired model, another server may
	}
	serverPoisonCodes = []codes.Code{
		codes.Unavailable,       // another server may be needed
		codes.ResourceExhausted, // another server may not be overloaded
		codes.NotFound,          // another server may load the model
	}
	joinRetryCodes = []codes.Code{
		codes.DeadlineExceeded, // server not ready to respond to GetStatus yet
		codes.Canceled,         // admin canceled a timed-out GetStatus request
	}
)

// AdminShouldRetry returns true iff the error indicates the operation
// sent to the admin should be retried.
func AdminShouldRetry(err error) bool {
	return isOneOf(err, adminRetryCodes)
}

// AdminShouldPoison returns true iff the error indicates the connection
// to the admin server should be reestablished.
func AdminShouldPoison(err error) bool {
	return isOneOf(err, adminPoisonCodes)
}

// ServerShouldRetry returns true iff the error indicates the operation
// sent to the server should be retried.
func ServerShouldRetry(err error) bool {
	return isOneOf(err, serverRetryCodes)
}

// ServerShouldPoison returns true iff the error indicates the connection
// to the server should be reestablished.
func ServerShouldPoison(err error) bool {
	return isOneOf(err, serverPoisonCodes)
}

// JoinShouldRetry returns true iff the error indicates the Join RPC
// should be retried.
func JoinShouldRetry(err error) bool {
	return isOneOf(err, joinRetryCodes)
}

// IsDeadlineExceeded checks if an error is deadline exceeded.
func IsDeadlineExceeded(err error) bool {
	return Code(err) == codes.DeadlineExceeded
}

// IsNotFound checks if an error is not found.
func IsNotFound(err error) bool {
	return Code(err) == codes.NotFound
}
