# Protobuf definitions for Sax.

load(
    "//saxml:saxml.bzl",
    "cc_grpc_library",
    "cc_proto_library",
    "cc_stubby_library",
    "go_grpc_library",
    "go_proto_library",
    "jspb_proto_library",
    "kt_jvm_grpc_library",
    "kt_jvm_proto_library",
    "kt_jvm_stubby_library",
    "proto_library",
    "py_grpc_library",
    "py_proto_library",
    "rpc_endpoint_interface",
)

package(
    default_visibility = ["//saxml:internal"],
)

package_group(
    name = "sax_rpc_visibility",
    includes = [
        "//saxml:internal",
    ],
    packages = [
        "//java/com/google/vr/multimodal/...",
        "//javatests/com/google/vr/multimodal/...",
        "//research/bigpicture/llm_api_use/sax/...",
        "//vr/multimodal/...",
    ],
)

proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
    visibility = ["//visibility:public"],
)

go_proto_library(
    name = "common_go_proto",
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

cc_proto_library(
    name = "common_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

py_proto_library(
    name = "common_py_pb2",
    api_version = 2,
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

java_proto_library(
    name = "common_java_proto",
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

kt_jvm_proto_library(
    name = "common_kt_proto",
    visibility = ["//visibility:public"],
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

cc_stubby_library(
    name = "admin_cc_stubby",
    srcs = [":admin_proto"],
    visibility = [
        "//visibility:private",  # Only private by automation, not intent. Owner may accept CLs adding visibility. See go/scheuklappen#explicit-private.
    ],
    deps = [":admin_cc_proto"],
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
    visibility = [":sax_rpc_visibility"],
    deps = [":admin_proto"],
)

py_grpc_library(
    name = "admin_py_pb2_grpc",
    srcs = [":admin_proto"],
    visibility = [":sax_rpc_visibility"],
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

cc_proto_library(
    name = "modelet_cc_proto",
    deps = [":modelet_proto"],
)

cc_grpc_library(
    name = "modelet_cc_proto_grpc",
    srcs = [":modelet_proto"],
    deps = [":modelet_cc_proto"],
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
    use_java_stubby_library = True,
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

rpc_endpoint_interface(
    name = "custom_endpoint",
    proto_library = ":custom_proto",
    rpc_service_name = "sax.server.custom.CustomService",
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "custom_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":custom_proto"],
)

cc_stubby_library(
    name = "custom_cc_stubby",
    srcs = [":custom_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":custom_cc_proto"],
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
    visibility = [":sax_rpc_visibility"],
    deps = [":custom_proto"],
)

go_grpc_library(
    name = "custom_go_proto_grpc",
    srcs = [":custom_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":custom_go_proto"],
)

java_proto_library(
    name = "custom_java_proto",
    visibility = ["//visibility:public"],
    deps = [":custom_proto"],
)

kt_jvm_proto_library(
    name = "custom_kt_proto",
    visibility = ["//visibility:public"],
    deps = [":custom_proto"],
)

kt_jvm_stubby_library(
    name = "custom_kt_stubby",
    srcs = [":custom_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":custom_kt_proto"],
)

proto_library(
    name = "lm_proto",
    srcs = ["lm.proto"],
    has_services = True,
    use_java_stubby_library = True,
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
    visibility = [
        "//visibility:public",
    ],
    deps = [":lm_proto"],
)

cc_stubby_library(
    name = "lm_cc_stubby",
    srcs = [":lm_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":lm_cc_proto"],
)

py_proto_library(
    name = "lm_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    visibility = ["//visibility:public"],
    deps = [":lm_proto"],
)

cc_grpc_library(
    name = "lm_cc_grpc",
    srcs = [":lm_proto"],
    deps = [":lm_cc_proto"],
)

py_grpc_library(
    name = "lm_py_pb2_grpc",
    srcs = [":lm_proto"],
    visibility = ["//visibility:public"],
    deps = [":lm_py_pb2"],
)

go_proto_library(
    name = "lm_go_proto",
    extra_deps = [":common_go_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":lm_proto"],
)

go_grpc_library(
    name = "lm_go_proto_grpc",
    srcs = [":lm_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":lm_go_proto"],
)

java_proto_library(
    name = "lm_java_proto",
    visibility = ["//visibility:public"],
    deps = [":lm_proto"],
)

kt_jvm_proto_library(
    name = "lm_kt_proto",
    visibility = ["//visibility:public"],
    deps = [":lm_proto"],
)

kt_jvm_stubby_library(
    name = "lm_kt_stubby",
    srcs = [":lm_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":lm_kt_proto"],
)

kt_jvm_grpc_library(
    name = "lm_kt_proto_grpc",
    srcs = [":lm_proto"],
    visibility = ["//java/com/google/wireless/android/devtools/studio/ml/eval/auto:__subpackages__"],
    deps = [":lm_kt_proto"],
)

proto_library(
    name = "audio_proto",
    srcs = ["audio.proto"],
    has_services = True,
    use_java_stubby_library = True,
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
    visibility = [":sax_rpc_visibility"],
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
    visibility = [":sax_rpc_visibility"],
    deps = [":audio_proto"],
)

go_grpc_library(
    name = "audio_go_proto_grpc",
    srcs = [":audio_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":audio_go_proto"],
)

java_proto_library(
    name = "audio_java_proto",
    visibility = ["//visibility:public"],
    deps = [":audio_proto"],
)

kt_jvm_proto_library(
    name = "audio_kt_proto",
    visibility = ["//visibility:public"],
    deps = [":audio_proto"],
)

kt_jvm_stubby_library(
    name = "audio_kt_stubby",
    srcs = [":audio_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":audio_kt_proto"],
)

proto_library(
    name = "vision_proto",
    srcs = ["vision.proto"],
    has_services = True,
    use_java_stubby_library = True,
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
    visibility = ["//visibility:public"],
    deps = [":vision_proto"],
)

cc_stubby_library(
    name = "vision_cc_stubby",
    srcs = [":vision_proto"],
    visibility = [":sax_rpc_visibility"],
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
    visibility = [":sax_rpc_visibility"],
    deps = [":vision_proto"],
)

go_grpc_library(
    name = "vision_go_proto_grpc",
    srcs = [":vision_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":vision_go_proto"],
)

java_proto_library(
    name = "vision_java_proto",
    visibility = ["//visibility:public"],
    deps = [":vision_proto"],
)

kt_jvm_proto_library(
    name = "vision_kt_proto",
    visibility = ["//visibility:public"],
    deps = [":vision_proto"],
)

kt_jvm_stubby_library(
    name = "vision_kt_stubby",
    srcs = [":vision_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":vision_kt_proto"],
)

proto_library(
    name = "multimodal_proto",
    srcs = ["multimodal.proto"],
    has_services = True,
    use_java_stubby_library = True,
    visibility = ["//visibility:public"],
    deps = [":common_proto"],
)

rpc_endpoint_interface(
    name = "multimodal_endpoint",
    proto_library = ":multimodal_proto",
    rpc_service_name = "sax.server.multimodal.MultimodalService",
    visibility = ["//visibility:public"],
)

cc_proto_library(
    name = "multimodal_cc_proto",
    visibility = ["//visibility:public"],
    deps = [":multimodal_proto"],
)

cc_stubby_library(
    name = "multimodal_cc_stubby",
    srcs = [":multimodal_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":multimodal_cc_proto"],
)

py_proto_library(
    name = "multimodal_py_pb2",
    api_version = 2,
    extra_deps = [":common_py_pb2"],
    visibility = ["//visibility:public"],
    deps = [":multimodal_proto"],
)

py_grpc_library(
    name = "multimodal_py_pb2_grpc",
    srcs = [":multimodal_proto"],
    deps = [":multimodal_py_pb2"],
)

go_proto_library(
    name = "multimodal_go_proto",
    extra_deps = [":common_go_proto"],
    visibility = ["//visibility:public"],
    deps = [":multimodal_proto"],
)

go_grpc_library(
    name = "multimodal_go_proto_grpc",
    srcs = [":multimodal_proto"],
    extra_deps = [":common_go_proto"],
    deps = [":multimodal_go_proto"],
)

kt_jvm_proto_library(
    name = "multimodal_kt_proto",
    visibility = ["//visibility:public"],
    deps = [":multimodal_proto"],
)

kt_jvm_stubby_library(
    name = "multimodal_kt_stubby",
    srcs = [":multimodal_proto"],
    visibility = [":sax_rpc_visibility"],
    deps = [":multimodal_kt_proto"],
)

jspb_proto_library(
    name = "multimodal_jspb_proto",
    visibility = ["//visibility:public"],
    deps = [":multimodal_proto"],
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
