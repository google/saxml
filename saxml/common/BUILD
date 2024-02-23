# Sax components used by the client, admin server, and model server.

load("//saxml:saxml.bzl", "go_binary", "go_library", "go_test")

package(
    default_visibility = ["//saxml:internal"],
)

go_library(
    name = "errors",
    srcs = ["errors.go"],
    deps = [
        "@org_golang_google_grpc//codes:go_default_library",
        "@org_golang_google_grpc//status:go_default_library",
    ],
)

go_test(
    name = "errors_test",
    size = "small",
    srcs = ["errors_test.go"],
    deps = [":errors"],
)

go_library(
    name = "naming",
    srcs = ["naming.go"],
    deps = [":errors"],
)

go_test(
    name = "naming_test",
    size = "small",
    srcs = ["naming_test.go"],
    library = ":naming",
)

go_library(
    name = "ipaddr",
    srcs = ["ipaddr.go"],
    deps = ["@com_github_golang_glog//:go_default_library"],
)

go_library(
    name = "cell",
    srcs = ["cell.go"],
    deps = [
        ":errors",
        ":naming",
        "//saxml/common/platform:env",
        "@com_github_golang_glog//:go_default_library",
    ],
)

go_library(
    name = "addr",
    srcs = ["addr.go"],
    deps = [
        ":cell",
        ":errors",
        ":ipaddr",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_protobuf//proto",
    ],
)

go_library(
    name = "location",
    srcs = ["location.go"],
    deps = [
        ":addr",
        ":cell",
        ":errors",
        ":retrier",
        "//saxml/admin",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        # unused internal admin gRPC dependency,
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_protobuf//proto",
    ],
)

go_test(
    name = "location_test",
    srcs = ["location_test.go"],
    deps = [
        ":addr",
        ":location",
        ":testutil",
        ":watchable",
        "//saxml/common/platform:env",
        "//saxml/common/platform:register",
        "//saxml/protobuf:admin_go_proto_grpc",
    ],
)

go_binary(
    name = "locationwrapper",
    srcs = ["locationwrapper.go"],
    cgo = True,
    deps = [
        ":location",
        "//saxml/common/platform:register",
        "//saxml/protobuf:admin_go_proto_grpc",
        "@org_golang_google_protobuf//proto",
    ],
)

cc_library(
    name = "location_cc",
    srcs = ["location.cc"],
    hdrs = ["location.h"],
    deps = [":locationwrapper"],
)

cc_test(
    name = "location_cc_test",
    srcs = ["location_test.cc"],
    deps = [
        ":location_cc",
        ":testutil_cc",
        # unused internal gRPC dependency,
        # unused internal gRPC resolver dependency,
        "//saxml/common/platform:env_cc",
        "//saxml/protobuf:admin_cc_grpc",
        "//saxml/protobuf:admin_cc_proto",
        "@com_google_googletest//:gtest_main",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

go_library(
    name = "config",
    srcs = ["config.go"],
    deps = [
        ":cell",
        ":errors",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_protobuf//encoding/prototext",
        "@org_golang_google_protobuf//proto",
    ],
)

go_library(
    name = "state",
    srcs = ["state.go"],
    deps = [
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        "@org_golang_google_protobuf//proto",
    ],
)

go_library(
    name = "retrier",
    srcs = ["retrier.go"],
    deps = [
        "@com_github_cenkalti_backoff//:go_default_library",
        "@com_github_golang_glog//:go_default_library",
    ],
)

go_test(
    name = "retrier_test",
    size = "small",
    srcs = ["retrier_test.go"],
    deps = [
        ":errors",
        ":retrier",
    ],
)

go_library(
    name = "watchable",
    srcs = ["watchable.go"],
    deps = [
        "//saxml/protobuf:admin_go_proto_grpc",
        "@com_github_golang_glog//:go_default_library",
    ],
)

go_test(
    name = "watchable_test",
    srcs = ["watchable_test.go"],
    deps = [
        ":watchable",
        "@com_github_google_go_cmp//cmp:go_default_library",
    ],
)

go_library(
    name = "testutil",
    testonly = True,
    srcs = ["testutil.go"],
    visibility = ["//visibility:public"],
    deps = [
        ":addr",
        ":cell",
        ":config",
        ":errors",
        ":naming",
        ":watchable",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        # unused internal admin gRPC dependency,
        "//saxml/protobuf:audio_go_proto_grpc",
        # unused internal audio gRPC dependency,
        "//saxml/protobuf:common_go_proto",
        "//saxml/protobuf:custom_go_proto_grpc",
        # unused internal custom gRPC dependency,
        "//saxml/protobuf:lm_go_proto_grpc",
        # unused internal lm gRPC dependency,
        "//saxml/protobuf:modelet_go_proto_grpc",
        # unused internal modelet gRPC dependency,
        "//saxml/protobuf:multimodal_go_proto_grpc",
        # unused internal multimodal gRPC dependency,
        "//saxml/protobuf:vision_go_proto_grpc",
        # unused internal vision gRPC dependency,
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_protobuf//proto",
    ],
)

go_binary(
    name = "testutilwrapper",
    testonly = True,
    srcs = ["testutilwrapper.go"],
    cgo = True,
    deps = [
        ":testutil",
        "//saxml/common/platform:register",
        "@com_github_golang_glog//:go_default_library",
    ],
)

cc_library(
    name = "testutil_cc",
    testonly = True,
    srcs = ["testutil.cc"],
    hdrs = ["testutil.h"],
    visibility = ["//visibility:public"],
    deps = [":testutilwrapper"],
)

go_library(
    name = "waitable",
    srcs = ["waitable.go"],
    deps = [":errors"],
)

go_test(
    name = "waitable_test",
    srcs = ["waitable_test.go"],
    deps = [
        ":errors",
        ":waitable",
    ],
)

go_library(
    name = "basiceventlogger",
    srcs = ["basiceventlogger.go"],
    deps = [
        ":eventlog",
        "@com_github_golang_glog//:go_default_library",
    ],
)

go_library(
    name = "eventlog",
    srcs = ["eventlog.go"],
)

go_library(
    name = "skiplist",
    srcs = ["skiplist.go"],
)

go_test(
    name = "skiplist_test",
    srcs = ["skiplist_test.go"],
    library = ":skiplist",
)
