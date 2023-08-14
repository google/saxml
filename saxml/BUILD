# Saxml: Serving for Paxml.

# unused internal load
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//saxml:internal"])

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//saxml/..."],
)

bzl_library(
    name = "saxml_bzl",
    srcs = ["saxml.bzl"],
    deps = ["//third_party/bazel_rules/rules_python/python:proto_bzl"],
)
