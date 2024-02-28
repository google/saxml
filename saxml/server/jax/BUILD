# JAX-specific code.

load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = ["//saxml:internal"],
)

pytype_strict_library(
    name = "jax_spmd_backend",
    srcs = ["jax_spmd_backend.py"],
    srcs_version = "PY3",
    deps = [
        "//saxml/server:spmd_backend",
        "//saxml/server:utils",
        "//third_party/py/absl-py/logging",
        "//third_party/py/jax",
        "//third_party/py/jax:mesh_utils",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "serialize",
    srcs = ["serialize.py"],
    srcs_version = "PY3",
    visibility = [
        "//saxml:internal",
    ],
    deps = [
        "//third_party/py/jax",
    ],
)

py_strict_test(
    name = "serialize_test",
    srcs = ["serialize_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    deps = [
        ":serialize",
        "//third_party/py/absl-py/testing:absltest",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
        "//third_party/py/jax",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "servable_model",
    srcs = ["servable_model.py"],
    srcs_version = "PY3",
    deps = [
        "//saxml/server:servable_model",
        "//saxml/server:servable_model_params",
        "//third_party/py/absl-py/logging",
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/jax:experimental",
        "//third_party/py/numpy",
        "//third_party/py/paxml:host_callback",
        "//third_party/py/praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "np_tf_sess_wrapper",
    srcs = ["np_tf_sess_wrapper.py"],
    srcs_version = "PY3",
    deps = [
        "//saxml/server/tf:np_tf_sess_wrapper",
        "//third_party/py/jax",
    ],
)
