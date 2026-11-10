# Public interface of LM inference service.

load("//third_party/bazel_rules/rules_python/python:py_library.bzl", "py_library")

package(
    # unused internal package attribute
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "common",
    srcs = ["common.py"],
    type_checking = True,
    visibility = ["//visibility:public"],
    deps = [
        "//saxml/server/pax/lm:lm_tokenizer",
        "//saxml/server/pax/lm:servable_lm_common",
        "//saxml/server/pax/lm/params:template",
    ],
)
