# Saxml: Serving for Paxml.

# unused internal load
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

package(default_visibility = ["//saxml:internal"])

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//gdm/sys/gammaray/...",
        "//gdm/sys/sax/...",
        "//saxml/...",
    ],
)

bzl_library(
    name = "saxml_bzl",
    srcs = ["saxml.bzl"],
    deps = [
        "//third_party/protobuf/bazel:proto_library_bzl",
        "//tools/build_defs/go:go_binary_bzl",
        "//tools/build_defs/go:go_library_bzl",
        "//tools/build_defs/go:go_proto_library_bzl",
        "//tools/build_defs/go:go_test_bzl",
    ],
)
