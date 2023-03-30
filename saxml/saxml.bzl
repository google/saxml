# Copyright 2022 Google LLC
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

"""Build macros for Saxml."""

load("@rules_proto//proto:defs.bzl", _proto_library = "proto_library")
load("@rules_cc//cc:defs.bzl", _cc_proto_library = "cc_proto_library")
load("@com_github_grpc_grpc//bazel:cc_grpc_library.bzl", _cc_grpc_library = "cc_grpc_library")
load("@com_google_protobuf//:protobuf.bzl", _py_proto_library = "py_proto_library")
load("@com_github_grpc_grpc//bazel:python_rules.bzl", _py_grpc_library = "py_grpc_library")
load("@io_bazel_rules_go//proto:def.bzl", _go_grpc_library = "go_grpc_library", _go_proto_library = "go_proto_library")
load("@pybind11_bazel//:build_defs.bzl", _pybind_extension = "pybind_extension", _pybind_library = "pybind_library")
load("@io_bazel_rules_go//go:def.bzl", _go_binary = "go_binary", _go_library = "go_library", _go_test = "go_test")
load("@third_party//:requirements.bzl", "requirement")

def proto_library(
        name,
        srcs = [],
        has_services = None,  # @unused
        deps = [],
        **kwargs):
    _proto_library(name = name, srcs = srcs, deps = deps, **kwargs)

def cc_proto_library(name, deps = [], **kwargs):
    _cc_proto_library(name = name, deps = deps, **kwargs)

def cc_stubby_library(
        name,  # @unused
        srcs = [],  # @unused
        deps = [],  # @unused
        **kwargs):  # @unused
    pass

def cc_grpc_library(name, srcs = [], deps = [], **kwargs):
    _cc_grpc_library(name = name, srcs = srcs, grpc_only = True, deps = deps, **kwargs)

def py_proto_library(
        name,
        deps = [],
        extra_deps = [],
        api_version = None,  # @unused
        **kwargs):
    """Generates Python code from proto files.

    Args:
        name: A unique name for this target.
        deps: The list of proto_library rules to generate Python code for.
        extra_deps: The list of py_proto_library rules that correspond to the proto_library rules
            referenced by deps.
        api_version: The Python API version to generate.
        **kwargs: Args passed through to py_proto_library rules.
    """

    srcs = []
    for dep in deps:
        if not dep.startswith(":") or not dep.endswith("_proto"):
            fail("py_proto_library %s's dep %s has an invalid name")
        src = dep[1:-6] + ".proto"
        srcs.append(src)
    _py_proto_library(name = name, srcs = srcs, deps = extra_deps, **kwargs)

def py_grpc_library(name, srcs = [], deps = [], **kwargs):
    _py_grpc_library(name = name, srcs = srcs, deps = deps, **kwargs)

def go_proto_library(name, deps = [], extra_deps = [], **kwargs):
    """Generates Go code from proto files.

    Args:
        name: A unique name for this target.
        deps: The list of proto_library rules to generate Go code for.
        extra_deps: The list of go_proto_library rules that correspond to the proto_library rules
            referenced by deps.
        **kwargs: Args passed through to go_proto_library rules.
    """

    _go_proto_library(
        name = name,
        protos = deps,
        deps = extra_deps,
        importpath = native.package_name() + "/" + name,
        **kwargs
    )

def go_grpc_library(name, srcs, deps, extra_deps = [], **kwargs):
    """Generates Go gRPC code from proto files.

    Args:
        name: A unique name for this target.
        srcs: The list of proto_library rules to generate Go code for.
        deps: The list of go_proto_library rules that correspond to srcs.
        extra_deps: The list of go_proto_library rules that correspond to the proto_library rules
            referenced by deps.
        **kwargs: Args passed through to go_grpc_library rules.
    """

    _go_grpc_library(
        name = name,
        protos = srcs,
        importpath = native.package_name() + "/" + name,
        deps = deps + extra_deps,
        **kwargs
    )

def pybind_extension(name, srcs = [], deps = [], **kwargs):
    _pybind_extension(name = name, srcs = srcs, deps = deps, **kwargs)

def pybind_library(name, srcs = [], deps = [], **kwargs):
    _pybind_library(name = name, srcs = srcs, deps = deps, **kwargs)

def clean_py_deps(deps):
    """Merges Python subpackage targets from the same third_party package into a single requirement.

    Args:
        deps: A list of Python targets to clean up.

    Returns:
        A list of cleaned up targets.
"""

    cleaned = []
    seen = {}
    for dep in deps:
        if not dep.startswith("//third_party/py/"):
            cleaned.append(dep)
            continue

        # Extract the package name, e.g. "foo" from "//third_party/py/foo/bar:baz".
        pkg = dep[len("//third_party/py/"):]
        pkg = pkg[:pkg.find("/")] if "/" in pkg else pkg
        pkg = pkg[:pkg.find(":")] if ":" in pkg else pkg
        if not pkg in seen:
            seen[pkg] = True
            cleaned.append(requirement(pkg))
    return cleaned

def pytype_strict_library(name, srcs = [], deps = [], pybind_deps = [], **kwargs):
    data = []
    for pybind_dep in pybind_deps:
        data.append(pybind_dep + ".so")
    native.py_library(name = name, srcs = srcs, deps = clean_py_deps(deps), data = data, **kwargs)

def pytype_strict_binary(name, srcs = [], deps = [], **kwargs):
    native.py_binary(name = name, srcs = srcs, deps = clean_py_deps(deps), **kwargs)

def py_strict_test(name, srcs = [], deps = [], pybind_deps = [], data = None, **kwargs):
    data = [] if data == None else data
    for pybind_dep in pybind_deps:
        data.append(pybind_dep + ".so")
    native.py_test(name = name, srcs = srcs, deps = clean_py_deps(deps), data = data, **kwargs)

def go_library(name, srcs = [], deps = [], **kwargs):
    _go_library(
        name = name,
        srcs = srcs,
        importpath = native.package_name() + "/" + name,
        deps = deps,
        **kwargs
    )

def go_binary(name, srcs = [], deps = [], cgo = None, **kwargs):
    """Generates a cgo or non-cgo go_binary.

    Args:
        name: A unique name for this target.
        srcs: The list of source files.
        deps: The list of dependencies.
        cgo: If True, enable cgo support.
        **kwargs: Args passed through to go_binary or go_library rules.
    """
    if cgo:
        _go_binary(
            name = name,
            srcs = srcs,
            importpath = native.package_name() + "/" + name,
            cgo = True,
            linkmode = "c-shared",
            deps = deps,
            **kwargs
        )
    else:
        _go_binary(name = name, srcs = srcs, deps = deps, **kwargs)

def go_test(name, srcs = [], library = None, deps = [], **kwargs):
    _go_test(
        name,
        srcs = srcs,
        embed = [] if (library == None) else [library],
        deps = deps,
        **kwargs
    )

def rpc_endpoint_interface(
        name,
        rpc_service_name,  # @unused
        proto_library,  # @unused
        **kwargs):  # @unused
    pass
