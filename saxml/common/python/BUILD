# Python API wrappers.

load("//saxml:saxml.bzl", "py_strict_test", "pybind_extension", "pytype_strict_library")

package(
    default_visibility = ["//saxml:internal"],
)

pybind_extension(
    name = "pybind_location",
    srcs = ["pybind_location.cc"],
    deps = ["//saxml/common:location_cc"],
)

pytype_strict_library(
    name = "location",
    srcs = ["location.py"],
    pybind_deps = [":pybind_location"],
    srcs_version = "PY3",
)

py_strict_test(
    name = "location_test",
    srcs = ["location_test.py"],
    pybind_deps = [":testutil"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":location",
        "//third_party/py/grpcio",
        "//third_party/py/absl-py/testing:absltest",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
        "//third_party/py/portpicker",
        "//saxml/common/platform:env_py",
        "//saxml/protobuf:admin_py_pb2",
        "//saxml/protobuf:admin_py_pb2_grpc",
    ],
)

pybind_extension(
    name = "testutil",
    testonly = True,
    srcs = ["testutil.cc"],
    visibility = ["//visibility:public"],
    deps = ["//saxml/common:testutil_cc"],
)
