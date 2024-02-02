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
        "//saxml/...",
    ],
)

bzl_library(
    name = "saxml_bzl",
    srcs = ["saxml.bzl"],
)
