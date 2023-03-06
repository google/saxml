#!/bin/bash
# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Builds Saxml and runs unit tests.

RETURN_CODE=0

print_result () {
	if [[ $1 -eq 0 ]] ; then
	  echo "$2 Passed"
	else
	  echo "$2 Failed"
	fi
}

# Builds the targets.
bazel build -c opt saxml/server:all
SERVER_BUILD_RESULT=$?
if [[ $SERVER_BUILD_RESULT -ne 0 ]]; then RETURN_CODE=$SERVER_BUILD_RESULT; fi

bazel build -c opt saxml/bin:all
ADMIN_BUILD_RESULT=$?
if [[ $ADMIN_BUILD_RESULT -ne 0 ]]; then RETURN_CODE=$ADMIN_BUILD_RESULT; fi

# Runs tests that exist.
bazel test -c opt saxml/server:all
SERVER_TEST_RESULT=$?
if [[ $SERVER_TEST_RESULT -ne 0 ]]; then RETURN_CODE=$SERVER_TEST_RESULT; fi

# Prints summary results.
print_result $SERVER_BUILD_RESULT "bazel build saxml/server:all"
print_result $ADMIN_BUILD_RESULT "bazel build saxml/bin:all"
print_result $SERVER_TEST_RESULT "bazel test saxml/server:all"

# Return code is non-zero if any of the build or test steps failed.
exit $RETURN_CODE
