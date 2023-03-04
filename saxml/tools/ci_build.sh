#!/bin/bash
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
