# Python API wrappers.

load("//saxml:saxml.bzl", "pybind_extension")

pybind_extension(
    name = "testutil",
    testonly = True,
    srcs = ["testutil.cc"],
    visibility = ["//visibility:public"],
    deps = ["//saxml/common:testutil_cc"],
)
