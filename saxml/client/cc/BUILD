# C++ client for SAX.

load("//saxml:saxml.bzl", "go_binary")

package(
    default_visibility = ["//saxml:internal"],
)

cc_library(
    name = "sax",
    srcs = ["sax.cc"],
    hdrs = ["sax.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":saxwrapper",
        "//saxml/protobuf:admin_cc_proto",
        "//saxml/protobuf:audio_cc_proto",
        "//saxml/protobuf:common_cc_proto",
        "//saxml/protobuf:custom_cc_proto",
        "//saxml/protobuf:lm_cc_proto",
        "//saxml/protobuf:vision_cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

cc_test(
    name = "sax_test",
    srcs = ["sax_test.cc"],
    flaky = True,
    deps = [
        ":sax",
        "//saxml/common/platform:env_cc",
        "@com_google_googletest//:gtest_main",
    ],
)

go_binary(
    name = "saxwrapper",
    srcs = ["saxwrapper.go"],
    cgo = True,
    visibility = ["//visibility:private"],
    deps = [
        "//saxml/client/go:sax",
        "//saxml/client/go:saxadmin",
        "//saxml/common:errors",
        "//saxml/common:naming",
        "//saxml/common/platform:register",
        "//saxml/protobuf:admin_go_proto_grpc",
        "//saxml/protobuf:audio_go_proto_grpc",
        "//saxml/protobuf:common_go_proto",
        "//saxml/protobuf:custom_go_proto_grpc",
        "//saxml/protobuf:lm_go_proto_grpc",
        "//saxml/protobuf:vision_go_proto_grpc",
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_protobuf//proto",
    ],
)
