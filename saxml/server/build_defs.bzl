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

"""Build macros for SAX."""

load("@com_google_paxml//paxml:build_defs.bzl", "export_binary", "export_sources")
load("//saxml:saxml.bzl", "pytype_strict_binary")

def create_binary(
        imports_targets = None,
        extra_deps = None,
        name = "",
        main = "",
        tags = [],
        **kwargs):
    """Macro to define a binary with selected imports.

    Args:
      imports_targets: List of targets that contains all necessary imports.
      extra_deps: Extra deps not included in imports_targets
      name: binary name.
      main: Main source file.
      tags: Tags when creating binary.
    """
    if not name:
        fail("name is empty")
    if not main:
        fail("main is empty")
    if not imports_targets:
        fail("imports_srcs is not provided.")

    extra_deps = extra_deps or []

    exp_sources = "_all_internal_sources_" + name
    export_sources(
        name = exp_sources,
        deps = imports_targets,
    )

    export_binary(
        name = name,
        main = main,
        py_binary_rule = pytype_strict_binary,
        deps = imports_targets + extra_deps,
        exp_sources = exp_sources,
        # Unused internal paropts
        # Unused internal exec_properties
        tags = tags,
        **kwargs
    )

def create_server_binary(
        imports_targets = None,
        extra_deps = [],
        name = "server",
        main = "//saxml/server:model_service_main.py",
        default_deps = ["//saxml/server:server_deps"],
        use_tpu = False):
    """Macro to define a server binary with selected imports.

    Args:
      imports_targets: List of targets that contains all necessary imports.
      extra_deps: Extra deps not included in the target
        //saxml/server:server_deps or imports_targets.
      name: binary name.
    """

    if use_tpu:
        pass  # Unused internal TPU extra deps
    create_binary(
        imports_targets = imports_targets,
        extra_deps = extra_deps + default_deps,
        name = name,
        main = main,
    )
