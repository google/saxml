# Saxml: Serving for Paxml.

load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(
    default_visibility = ["//saxml:internal"],
)

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//saxml/..."],
)

bzl_library(
    name = "saxml_bzl",
    srcs = ["saxml.bzl"],
    deps = [
        "//devtools/build_cleaner/skylark:build_defs_lib",
        "//frameworks/platform/endpoints/builddefs:rpc_endpoint_interface",
    ],
)
