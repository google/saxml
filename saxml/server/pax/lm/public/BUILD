# Public interface of LM inference service.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(
    # unused internal package attribute
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pytype_strict_library(
    name = "common",
    srcs = ["common.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//saxml/server/pax/lm:lm_tokenizer",
        "//saxml/server/pax/lm:servable_lm_common",
        "//saxml/server/pax/lm/params:template",
    ],
)
