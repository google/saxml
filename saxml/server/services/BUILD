load("//saxml:saxml.bzl", "pytype_strict_library")

package(
    default_visibility = [
        "//saxml:internal",
        "//saxml/server:users",
    ],
)

pytype_strict_library(
    name = "custom_service",
    srcs = ["custom_service.py"],
    srcs_version = "PY3",
    visibility = [
        "//saxml:internal",
        "//saxml/server/pax/custom:__pkg__",
    ],
    deps = [
        "//saxml/protobuf:custom_py_pb2",
        "//saxml/protobuf:custom_py_pb2_grpc",
        "//saxml/server:model_service_base",
    ],
)

pytype_strict_library(
    name = "lm_service",
    srcs = ["lm_service.py"],
    srcs_version = "PY3",
    deps = [
        "//saxml/protobuf:lm_py_pb2",
        "//saxml/protobuf:lm_py_pb2_grpc",
        "//saxml/server:model_service_base",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "vision_service",
    srcs = ["vision_service.py"],
    srcs_version = "PY3",
    visibility = [
        "//learning/multipod/sax/server/vision:__pkg__",
        "//saxml:internal",
        "//saxml/server/pax/vision:__pkg__",
    ],
    deps = [
        "//saxml/protobuf:vision_py_pb2",
        "//saxml/protobuf:vision_py_pb2_grpc",
        "//saxml/server:model_service_base",
        "//third_party/py/numpy",
    ],
)
