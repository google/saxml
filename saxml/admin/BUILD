# Admin service for Sax.

load("//saxml:saxml.bzl", "go_library", "go_test")

package(
    default_visibility = ["//saxml:internal"],
)

go_library(
    name = "protobuf",
    srcs = ["protobuf.go"],
    deps = [
        "//saxml/common:errors",
        "//saxml/protobuf:admin_go_proto_grpc",
        "//saxml/protobuf:common_go_proto",
    ],
)

go_library(
    name = "validator",
    srcs = ["validator.go"],
    deps = [
        "//saxml/common:errors",
        "//saxml/common:naming",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
    ],
)

go_test(
    name = "validator_test",
    size = "small",
    srcs = ["validator_test.go"],
    deps = [
        ":validator",
        "//saxml/common/platform:env",
        "//saxml/common/platform:register",
        "//saxml/protobuf:admin_go_proto_grpc",
        "//saxml/protobuf:common_go_proto",
        "@com_github_google_go_cmp//cmp:go_default_library",
        "@com_github_google_go_cmp//cmp/cmpopts:go_default_library",
    ],
)

go_library(
    name = "state",
    srcs = ["state.go"],
    deps = [
        ":protobuf",
        "//saxml/common:errors",
        "//saxml/common:naming",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        "//saxml/protobuf:common_go_proto",
        "//saxml/protobuf:modelet_go_proto_grpc",
        # unused internal modelet gRPC dependency,
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_grpc//:go_default_library",
    ],
)

go_library(
    name = "mgr",
    srcs = ["mgr.go"],
    deps = [
        ":protobuf",
        ":state",
        ":validator",
        "//saxml/common:errors",
        "//saxml/common:naming",
        "//saxml/common:watchable",
        "//saxml/protobuf:admin_go_proto_grpc",
        "//saxml/protobuf:common_go_proto",
        "//saxml/protobuf:modelet_go_proto_grpc",
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_protobuf//proto",
    ],
)

go_library(
    name = "admin",
    srcs = [
        "admin.go",
        "admin_status.go",
    ],
    deps = [
        ":mgr",
        ":validator",
        "//saxml/common:addr",
        "//saxml/common:config",
        "//saxml/common:ipaddr",
        "//saxml/common:naming",
        "//saxml/common:state",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        # unused internal admin gRPC dependency,
        "//saxml/protobuf:common_go_proto",
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_protobuf//encoding/prototext",
    ],
)
