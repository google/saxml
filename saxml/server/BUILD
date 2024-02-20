# Sax model server.

load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_binary", "pytype_strict_library")
load("//saxml/server:build_defs.bzl", "create_server_binary")
load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

default_visibility = ["//saxml:internal"]

package(
    # unused internal package attribute
    default_visibility = default_visibility,
)

licenses(["notice"])

# NOTE In most cases, if you find that you need to add a package to
# :friends here, you should first consider creating a sub-directory in
# learning/multipod/sax/server/{lm,audio,...}, adds an OWNERS file
# listing your team. Your model params and implementation should be in
# that directory. If the above suggestion is not actionable for you,
# please schedule a meeting talking to zhifengc@.
package_group(
    name = "friends",
    packages = [
        "//waymo/planner/...",  # //waymo/... is siloed.
        "//waymo/research/...",  # //waymo/... is siloed.
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
    visibility = default_visibility + ["//saxml/server:friends"],
    deps = [
        ":servable_model_registry",
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
    ],
)

pytype_strict_library(
    name = "servable_model_params",
    srcs = ["servable_model_params.py"],
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//third_party/py/absl-py/logging",
        "//third_party/py/numpy",
    ],
)

py_strict_test(
    name = "servable_model_params_test",
    srcs = ["servable_model_params_test.py"],
    deps = [
        ":servable_model_params",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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

py_strict_test(
    name = "servable_model_registry_test",
    srcs = ["servable_model_registry_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":all_imports",
        ":servable_model_registry",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
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
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/grpcio",
    ],
)

pytype_strict_library(
    name = "proto_util",
    srcs = ["proto_util.py"],
    srcs_version = "PY3",
    deps = ["//saxml/protobuf:admin_py_pb2"],
)

py_strict_test(
    name = "proto_util_test",
    srcs = ["proto_util_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":proto_util",
        "//saxml/protobuf:admin_py_pb2",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/absl-py/testing:parameterized",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
    ],
)

pytype_strict_library(
    name = "ipaddr",
    srcs = ["ipaddr.py"],
    pybind_deps = ["//saxml/common/python:pybind_ipaddr"],
    srcs_version = "PY3",
)

py_strict_test(
    name = "ipaddr_test",
    srcs = ["ipaddr_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":ipaddr",
        "//third_party/py/absl-py/testing:absltest",
    ],
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
        "//third_party/py/jaxtyping",
        "//third_party/py/numpy",
        "@com_google_protobuf//:protobuf_python",
    ],
)

py_strict_test(
    name = "model_service_base_test",
    srcs = ["model_service_base_test.py"],
    deps = [
        ":model_service_base",
        ":utils",
        "//saxml/protobuf:common_py_pb2",
        "//saxml/protobuf:modelet_py_pb2",
        "//testing/pybase",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/portpicker",
    ],
)

pytype_strict_library(
    name = "server_deps",
    visibility = default_visibility + ["//saxml/server:friends"],
    deps = [
        ":model_service_base",
        ":model_service_main_lib",
        ":servable_model_registry",
        ":spmd_backend",
        "//saxml/protobuf:modelet_py_pb2",
        "//saxml/protobuf:modelet_py_pb2_grpc",
        "//saxml/server/jax:jax_spmd_backend",
        "//third_party/py/absl-py:app",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
        "//third_party/py/jax",
        "//third_party/py/libtpu-nightly",
        "//third_party/py/paxml:setup_jax",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "model_service_main_lib",
    srcs = ["model_service_main.py"],
    srcs_version = "PY3",
    deps = [
        ":model_service_base",
        ":servable_model_registry",
        ":spmd_backend",
        "//saxml/protobuf:modelet_py_pb2",
        "//saxml/protobuf:modelet_py_pb2_grpc",
        "//saxml/server/jax:jax_spmd_backend",
        "//third_party/py/absl-py:app",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
        "//third_party/py/jax",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

exports_files(
    srcs = [
        "model_service_main.py",
    ],
    visibility = default_visibility + ["//saxml/server:friends"],
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
