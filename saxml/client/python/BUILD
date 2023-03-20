load("//saxml:saxml.bzl", "pybind_extension", "pybind_library")

# This is the public python SAX client library to link / use.
pybind_extension(
    name = "sax",
    srcs = ["sax.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":wrapper",
        "//saxml/protobuf:common_cc_proto",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
    ],
)

# API wrapper on user facing C++ API for python API (pybind11 with sax.cc).
pybind_library(
    name = "wrapper",
    srcs = ["wrapper.cc"],
    hdrs = ["wrapper.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//saxml/client/cc:sax",
    ],
)
