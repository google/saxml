# Platform-specific packages.
#
# Any Go rule can import and use the env package, but to prevent conflicting
# platform targets from getting included at the same time, observe the
# following rules:
#
# 1.  go_binary and go_test targets that use the env package (including
#     transitively) should import the register target to register a platform.
# 2.  go_library targets, including testutil, should not import the register
#     target.
# 3.  Cgo libraries are a special case. Because they can get linked into
#     any non-Go targets that can't import Go platform targets, they should
#     import the register target.
#
# In addition, go_binary targets should call env.Get().Init() near the
# beginning of their main() functions. Tests shouldn't call it.

load("//saxml:saxml.bzl", "go_library", "go_test", "pytype_strict_library")

package(
    default_visibility = ["//saxml:internal"],
)

go_library(
    name = "env",
    srcs = ["env.go"],
    deps = [
        "//saxml/protobuf:admin_go_proto_grpc",
        "@com_github_golang_glog//:go_default_library",
        "@org_golang_google_grpc//:go_default_library",
    ],
)

config_setting(
    name = "vertex",
    values = {"define": "sax_env_vertex=1"},
)

go_library(
    name = "register",
    srcs = ["register_cloud.go"],
    visibility = ["//visibility:public"],
    deps = select({
        ":vertex": [":cloud"],
        "//conditions:default": [":cloud"],
    }),
)

go_library(
    name = "cloud",
    srcs = ["cloud.go"],
    visibility = ["//visibility:private"],
    deps = [
        ":env",
        # unused internal flag dependency,
        "//saxml/common:errors",
        "//saxml/protobuf:admin_go_proto_grpc",
        "@com_github_golang_glog//:go_default_library",
        "@com_github_google_safehtml//template:go_default_library",
        "@com_google_cloud_go_storage//:go_default_library",
        "@org_golang_google_api//iterator:go_default_library",
        "@org_golang_google_grpc//:go_default_library",
        "@org_golang_google_grpc//reflection:go_default_library",
        "@org_golang_x_oauth2//google:go_default_library",
    ],
)

go_test(
    name = "cloud_test",
    srcs = ["cloud_test.go"],
    deps = [
        ":cloud",
        ":env",
    ],
)

pytype_strict_library(
    name = "env_py",
    srcs = ["env.py"],
    deps = [":cloud_py"],
)

pytype_strict_library(
    name = "cloud_py",
    srcs = ["cloud.py"],
    deps = ["//third_party/py/grpcio"],
)

cc_library(
    name = "env_cc",
    srcs = ["env_cloud.cc"],
    hdrs = ["env.h"],
    deps = [
        # unused internal gRPC dependency,
        "@com_github_grpc_grpc//:grpc++",
    ],
)
