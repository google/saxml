load("//saxml:saxml.bzl", "pybind_extension", "pybind_library")

package(
    # unused internal package attribute
    default_visibility = ["//visibility:private"],
)

# This is the public Python Sax client library to link/use.
pybind_extension(
    name = "sax",
    srcs = ["sax.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":wrapper",
        "//saxml/client/cc:sax",
        "//saxml/protobuf:audio_cc_proto",
        "//saxml/protobuf:common_cc_proto",
        "//saxml/protobuf:custom_cc_proto",
        "//saxml/protobuf:lm_cc_proto",
        "//saxml/protobuf:multimodal_cc_proto",
        "//saxml/protobuf:vision_cc_proto",
        "@pybind11_abseil//pybind11_abseil:absl_casters",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

# API wrapper on user facing C++ API for python API (pybind11 with sax.cc).
pybind_library(
    name = "wrapper",
    srcs = ["wrapper.cc"],
    hdrs = ["wrapper.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "//saxml/client/cc:sax",
        "//saxml/protobuf:multimodal_cc_proto",
    ],
)
