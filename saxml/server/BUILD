# Sax model server.

load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_binary", "pytype_strict_library")
load("//saxml/server:build_defs.bzl", "create_server_binary")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])

package_group(name = "users")

package(
    default_visibility = [
        "//saxml:internal",
    ],
)

bzl_library(
    name = "build_defs_bzl",
    srcs = ["build_defs.bzl"],
    visibility = ["//visibility:public"],
    deps = ["//saxml:saxml_bzl"],
)

pytype_strict_library(
    name = "all_imports",
    srcs = ["all_imports.py"],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax:all_imports",
    ],
)

create_server_binary(
    imports_targets = [
        ":all_imports",
    ],
)

pytype_strict_library(
    name = "acl",
    srcs = ["acl.py"],
    srcs_version = "PY3",
)

pytype_strict_library(
    name = "multi_host_sync",
    srcs = ["multi_host_sync.py"],
    srcs_version = "PY3",
    deps = [
        ":spmd_backend",
        ":utils",
        "//saxml/protobuf:internal_py_pb2",
        "//saxml/protobuf:internal_py_pb2_grpc",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
    ],
)

py_strict_test(
    name = "multi_host_sync_test",
    srcs = ["multi_host_sync_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":multi_host_sync",
        "//third_party/py/absl-py/testing:absltest",
    ],
)

pytype_strict_library(
    name = "servable_model_params",
    srcs = ["servable_model_params.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "servable_model_registry",
    srcs = ["servable_model_registry.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":servable_model_params",
    ],
)

pytype_strict_library(
    name = "servable_model",
    srcs = ["servable_model.py"],
    srcs_version = "PY3",
    deps = [
        ":servable_model_params",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "spmd_backend",
    srcs = ["spmd_backend.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/grpcio",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "validate",
    srcs = ["validate.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_strict_test(
    name = "validate_test",
    srcs = ["validate_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":validate",
        "//saxml/protobuf:test_py_pb2",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/grpcio",
    ],
)

pytype_strict_library(
    name = "proto_util",
    srcs = ["proto_util.py"],
    srcs_version = "PY3",
    deps = ["//saxml/protobuf:admin_py_pb2"],
)

pytype_strict_library(
    name = "ipaddr",
    srcs = ["ipaddr.py"],
    srcs_version = "PY3",
)

pytype_strict_library(
    name = "model_service_base",
    srcs = ["model_service_base.py"],
    srcs_version = "PY3",
    deps = [
        ":acl",
        ":ipaddr",
        ":multi_host_sync",
        ":proto_util",
        ":servable_model",
        ":servable_model_params",
        ":servable_model_registry",
        ":spmd_backend",
        ":utils",
        ":validate",
        "//saxml/common/python:location",
        "//saxml/protobuf:admin_py_pb2",
        "//saxml/protobuf:common_py_pb2",
        "//saxml/protobuf:modelet_py_pb2",
        "//saxml/protobuf:modelet_py_pb2_grpc",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
        "//third_party/py/grpcio-reflection",
        "@com_google_protobuf//:protobuf_python",
    ],
)

pytype_strict_library(
    name = "server_deps",
    deps = [
        ":model_service_base",
        ":model_service_main_lib",
        ":servable_model_registry",
        "//saxml/protobuf:modelet_py_pb2",
        "//saxml/protobuf:modelet_py_pb2_grpc",
        "//saxml/server:spmd_backend",
        "//saxml/server/jax:jax_spmd_backend",
        "//third_party/py/absl-py:app",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
        "//third_party/py/jax",
        "//third_party/py/libtpu-nightly",
        "//third_party/py/paxml:setup_jax",
    ],
)

pytype_strict_library(
    name = "model_service_main_lib",
    srcs = ["model_service_main.py"],
    srcs_version = "PY3",
    deps = [
        ":model_service_base",
        ":servable_model_registry",
        "//saxml/protobuf:modelet_py_pb2",
        "//saxml/protobuf:modelet_py_pb2_grpc",
        "//saxml/server:spmd_backend",
        "//saxml/server/jax:jax_spmd_backend",
        "//third_party/py/absl-py:app",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
        "//third_party/py/jax",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_binary(
    name = "model_service_main",
    srcs = ["model_service_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_service_main_lib",
    ],
)
