# Go client for Sax.

load("//saxml:saxml.bzl", "go_library", "go_test")

package(
    default_visibility = ["//saxml:internal"],
)

go_library(
    name = "sax",
    srcs = [
        "sax.go",
        "sax_am.go",
        "sax_custom.go",
        "sax_export.go",
        "sax_lm.go",
        "sax_save.go",
        "sax_vm.go",
    ],
    visibility = ["//saxml:internal"],
    deps = [
        ":connection",
        ":location",
        ":saxadmin",
        "//saxml/common:errors",
        "//saxml/common:naming",
        "//saxml/common:retrier",
        "//saxml/common/platform:env",
        "//saxml/protobuf:audio_go_proto_grpc",
        # unused internal audio gRPC dependency,
        "//saxml/protobuf:common_go_proto",
        "//saxml/protobuf:custom_go_proto_grpc",
        # unused internal custom gRPC dependency,
        "//saxml/protobuf:lm_go_proto_grpc",
        # unused internal lm gRPC dependency,
        "//saxml/protobuf:modelet_go_proto_grpc",
        # unused internal modelet gRPC dependency,
        "//saxml/protobuf:vision_go_proto_grpc",
        # unused internal vision gRPC dependency,
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_grpc//:go_default_library",
    ],
)

go_library(
    name = "connection",
    srcs = ["connection.go"],
    deps = [
        "//saxml/common:errors",
        "//saxml/common/platform:env",
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_grpc//:go_default_library",
    ],
)

go_test(
    name = "connection_test",
    size = "small",
    srcs = ["connection_test.go"],
    library = ":connection",
    deps = [
        "//saxml/common:errors",
        "//saxml/common:testutil",
        "//saxml/common/platform:env",
        "//saxml/common/platform:register",
        "//saxml/protobuf:lm_go_proto_grpc",
        # unused internal lm gRPC dependency,
        "@com_github_google_go_cmp//cmp:go_default_library",
        "@org_golang_google_grpc//codes:go_default_library",
    ],
)

go_library(
    name = "location",
    srcs = ["location.go"],
    deps = [
        ":saxadmin",
        "//saxml/common:errors",
        "@com_github_golang_glog//:go_default_library",
    ],
)

go_library(
    name = "saxadmin",
    srcs = ["admin.go"],
    deps = [
        "//saxml/common:addr",
        "//saxml/common:errors",
        "//saxml/common:retrier",
        "//saxml/common:watchable",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        # unused internal admin gRPC dependency,
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_grpc//:go_default_library",
    ],
)
