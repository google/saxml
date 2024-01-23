load(
    "//saxml:saxml.bzl",
    "proto_library",
    "py_grpc_library",
    "py_proto_library",
    "pytype_strict_binary",
    "pytype_strict_library",
)

pytype_strict_binary(
    name = "launcher",
    srcs = ["launcher.py"],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":grpc_prediction_server",
        ":http_prediction_server",
        ":launcher_lib",
        ":sax_model_server",
        ":translate",
        "//third_party/py/absl-py:app",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
        "//third_party/py/tornado",
    ],
)

# Workaround since pytype_strict_binary doesn't support pybind_deps
pytype_strict_library(
    name = "launcher_lib",
    srcs = ["launcher.py"],
    pybind_deps = [
        "//saxml/client/python:sax",
    ],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":grpc_prediction_server",
        ":http_prediction_server",
        ":sax_model_server",
        ":translate",
        "//third_party/py/absl-py:app",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
        "//third_party/py/tornado",
    ],
)

filegroup(
    name = "launcher_zip",
    srcs = [":launcher"],
    output_group = "python_zip_file",
)

pytype_strict_library(
    name = "http_prediction_server",
    srcs = ["http_prediction_server.py"],
    pybind_deps = [
        "//saxml/client/python:sax",
    ],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":sax_model_server",
        ":translate",
        "//saxml/protobuf:admin_py_pb2",
        "//saxml/protobuf:admin_py_pb2_grpc",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//third_party/py/grpcio",
        "//third_party/py/tornado",
    ],
)

proto_library(
    name = "prediction_service_proto",
    srcs = ["prediction_service.proto"],
    has_services = True,
    deps = [
        "//saxml/protobuf:common_proto",
        "//saxml/protobuf:lm_proto",
    ],
)

py_proto_library(
    name = "prediction_service_py_pb2",
    api_version = 2,
    extra_deps = [
        "//saxml/protobuf:common_py_pb2",
        "//saxml/protobuf:lm_py_pb2",
    ],
    deps = [":prediction_service_proto"],
)

py_grpc_library(
    name = "prediction_service_py_pb2_grpc",
    srcs = [":prediction_service_proto"],
    deps = [":prediction_service_py_pb2"],
)

pytype_strict_library(
    name = "grpc_prediction_server",
    srcs = ["grpc_prediction_server.py"],
    pybind_deps = [
        "//saxml/client/python:sax",
    ],
    srcs_version = "PY3",
    deps = [
        ":constants",
        ":prediction_service_py_pb2",
        ":prediction_service_py_pb2_grpc",
        "//saxml/protobuf:lm_py_pb2",
        "//third_party/py/grpcio",
    ],
)

pytype_strict_library(
    name = "sax_model_server",
    srcs = ["sax_model_server.py"],
    deps = [
        ":constants",
        "//third_party/py/absl-py/logging",
        "//third_party/py/tornado",
    ],
)

pytype_strict_library(
    name = "translate",
    srcs = ["translate.py"],
    pybind_deps = [
        "//saxml/client/python:sax",
    ],
    deps = [
        ":constants",
        #"//saxml/client/python:sax",
    ],
)

pytype_strict_library(
    name = "constants",
    srcs = ["constants.py"],
)
