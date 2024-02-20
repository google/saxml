# Binaries for Sax.

load("//saxml:saxml.bzl", "go_binary", "go_library")

package(
    # unused internal package attribute
    default_visibility = [
        "//saxml:internal",
        "//saxml/server:friends",
    ],
)

go_binary(
    name = "admin_server",
    srcs = ["admin_server.go"],
    deps = [
        # unused internal flag dependency,
        # internal dependencies,
        "//saxml/admin",
        "//saxml/common/platform:env",
        "//saxml/common/platform:register",
        "@com_github_golang_glog//:go_default_library",
    ],
)

go_binary(
    name = "admin_config",
    srcs = ["admin_config.go"],
    visibility = ["//visibility:public"],
    deps = [
        # unused internal flag dependency,
        "//saxml/common:cell",
        "//saxml/common:config",
        "//saxml/common:naming",
        "//saxml/common/platform:env",
        "//saxml/common/platform:register",
        "@com_github_golang_glog//:go_default_library",
    ],
)

go_library(
    name = "saxcommand",
    srcs = [
        "saxutil_cmd_admin.go",
        "saxutil_cmd_am.go",
        "saxutil_cmd_common.go",
        "saxutil_cmd_export.go",
        "saxutil_cmd_lm.go",
        "saxutil_cmd_save.go",
        "saxutil_cmd_vm.go",
    ],
    deps = [
        # unused internal flag dependency,
        # Internal storage dependencies,
        "//saxml/admin:validator",
        "//saxml/client/go:sax",
        "//saxml/client/go:saxadmin",
        "//saxml/common:addr",
        "//saxml/common:cell",
        "//saxml/common:config",
        "//saxml/common:naming",
        "//saxml/common:watchable",
        "//saxml/common/platform:env",
        "//saxml/protobuf:admin_go_proto_grpc",
        "//saxml/protobuf:common_go_proto",
        "@com_github_golang_glog//:go_default_library",
        "@com_github_google_subcommands//:go_default_library",
        "@com_github_olekukonko_tablewriter//:go_default_library",
        "@org_golang_google_protobuf//proto",
    ],
)

go_binary(
    name = "saxutil",
    srcs = ["saxutil.go"],
    visibility = ["//visibility:public"],
    deps = [
        ":saxcommand",
        "//saxml/common/platform:env",
        "//saxml/common/platform:register",
        "@com_github_golang_glog//:go_default_library",
        "@com_github_google_subcommands//:go_default_library",
    ],
)
