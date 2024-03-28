# Python API wrappers.

load("//saxml:saxml.bzl", "pybind_extension")

package(
    default_visibility = ["//saxml:internal"],
)

pybind_extension(
    name = "testutil",
    testonly = True,
    srcs = ["testutil.cc"],
    visibility = ["//visibility:public"],
    deps = ["//saxml/common:testutil_cc"],
)
