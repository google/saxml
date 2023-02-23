# Protobuf definitions for Sax.

load(
    "//saxml:saxml.bzl",
    "cc_grpc_library",
    "cc_proto_library",
    "cc_stubby_library",
    "go_grpc_library",
    "go_proto_library",
    "proto_library",
    "py_grpc_library",
    "py_proto_library",
    "rpc_endpoint_interface",
)

package(
    default_visibility = ["//saxml:internal"],
)

proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
)

go_proto_library(
    name = "common_go_proto",
    deps = [":common_proto"],
)

cc_proto_library(
    name = "common_cc_proto",
    deps = [":common_proto"],
)

py_proto_library(
    name = "common_py_pb2",
    api_version = 2,
    deps = [":common_proto"],
)

proto_library(
    name = "test_proto",
    srcs = ["test.proto"],
    deps = [":common_proto"],
)

py_proto_library(
    name = "test_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    deps = [":test_proto"],
)

proto_library(
    name = "admin_proto",
    srcs = ["admin.proto"],
    has_services = True,
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

rpc_endpoint_interface(
    name = "admin_endpoint",
    proto_library = ":admin_proto",
    rpc_service_name = "sax.Admin",
    visibility = ["//visibility:public"],
)

go_proto_library(
    name = "admin_go_proto",
    extra_deps = [":common_go_proto"],
    deps = [":admin_proto"],
)

go_grpc_library(
    name = "admin_go_proto_grpc",
    srcs = [":admin_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":admin_go_proto"],
)

cc_proto_library(
    name = "admin_cc_proto",
    deps = [":admin_proto"],
)

cc_grpc_library(
    name = "admin_cc_grpc",
    srcs = [":admin_proto"],
    deps = [":admin_cc_proto"],
)

py_proto_library(
    name = "admin_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    deps = [":admin_proto"],
)

py_grpc_library(
    name = "admin_py_pb2_grpc",
    srcs = [":admin_proto"],
    deps = [":admin_py_pb2"],
)

proto_library(
    name = "modelet_proto",
    srcs = ["modelet.proto"],
    has_services = True,
    deps = [":common_proto"],
)

go_proto_library(
    name = "modelet_go_proto",
    extra_deps = [":common_go_proto"],
    deps = [":modelet_proto"],
)

go_grpc_library(
    name = "modelet_go_proto_grpc",
    srcs = [":modelet_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":modelet_go_proto"],
)

py_proto_library(
    name = "modelet_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    deps = [":modelet_proto"],
)

py_grpc_library(
    name = "modelet_py_pb2_grpc",
    srcs = [":modelet_proto"],
    deps = [":modelet_py_pb2"],
)

proto_library(
    name = "custom_proto",
    srcs = ["custom.proto"],
    has_services = True,
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

cc_proto_library(
    name = "custom_cc_proto",
    deps = [":custom_proto"],
)

py_proto_library(
    name = "custom_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    visibility = ["//visibility:public"],
    deps = [":custom_proto"],
)

py_grpc_library(
    name = "custom_py_pb2_grpc",
    srcs = [":custom_proto"],
    deps = [":custom_py_pb2"],
)

go_proto_library(
    name = "custom_go_proto",
    extra_deps = [":common_go_proto"],
    deps = [":custom_proto"],
)

go_grpc_library(
    name = "custom_go_proto_grpc",
    srcs = [":custom_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":custom_go_proto"],
)

proto_library(
    name = "lm_proto",
    srcs = ["lm.proto"],
    has_services = True,
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

rpc_endpoint_interface(
    name = "lm_endpoint",
    proto_library = ":lm_proto",
    rpc_service_name = "sax.server.lm.LMService",
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "lm_cc_proto",
    deps = [":lm_proto"],
)

cc_stubby_library(
    name = "lm_cc_stubby",
    srcs = [":lm_proto"],
    deps = [":lm_cc_proto"],
)

py_proto_library(
    name = "lm_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    visibility = ["//visibility:public"],
    deps = [":lm_proto"],
)

py_grpc_library(
    name = "lm_py_pb2_grpc",
    srcs = [":lm_proto"],
    deps = [":lm_py_pb2"],
)

go_proto_library(
    name = "lm_go_proto",
    extra_deps = [":common_go_proto"],
    deps = [":lm_proto"],
)

go_grpc_library(
    name = "lm_go_proto_grpc",
    srcs = [":lm_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":lm_go_proto"],
)

proto_library(
    name = "audio_proto",
    srcs = ["audio.proto"],
    has_services = True,
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

rpc_endpoint_interface(
    name = "audio_endpoint",
    proto_library = ":audio_proto",
    rpc_service_name = "sax.server.audio.AudioService",
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "audio_cc_proto",
    deps = [":audio_proto"],
)

cc_stubby_library(
    name = "audio_cc_stubby",
    srcs = [":audio_proto"],
    deps = [":audio_cc_proto"],
)

py_proto_library(
    name = "audio_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    visibility = ["//visibility:public"],
    deps = [":audio_proto"],
)

py_grpc_library(
    name = "audio_py_pb2_grpc",
    srcs = [":audio_proto"],
    deps = [":audio_py_pb2"],
)

go_proto_library(
    name = "audio_go_proto",
    extra_deps = [":common_go_proto"],
    deps = [":audio_proto"],
)

go_grpc_library(
    name = "audio_go_proto_grpc",
    srcs = [":audio_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":audio_go_proto"],
)

proto_library(
    name = "vision_proto",
    srcs = ["vision.proto"],
    has_services = True,
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

rpc_endpoint_interface(
    name = "vision_endpoint",
    proto_library = ":vision_proto",
    rpc_service_name = "sax.server.vision.VisionService",
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "vision_cc_proto",
    deps = [":vision_proto"],
)

cc_stubby_library(
    name = "vision_cc_stubby",
    srcs = [":vision_proto"],
    deps = [":vision_cc_proto"],
)

py_proto_library(
    name = "vision_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    visibility = ["//visibility:public"],
    deps = [":vision_proto"],
)

py_grpc_library(
    name = "vision_py_pb2_grpc",
    srcs = [":vision_proto"],
    deps = [":vision_py_pb2"],
)

go_proto_library(
    name = "vision_go_proto",
    extra_deps = [":common_go_proto"],
    deps = [":vision_proto"],
)

go_grpc_library(
    name = "vision_go_proto_grpc",
    srcs = [":vision_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":vision_go_proto"],
)

proto_library(
    name = "internal_proto",
    srcs = ["internal.proto"],
    has_services = True,
)

py_proto_library(
    name = "internal_py_pb2",
    api_version = 2,
    deps = [":internal_proto"],
)

py_grpc_library(
    name = "internal_py_pb2_grpc",
    srcs = [":internal_proto"],
    deps = [":internal_py_pb2"],
)
