# JAX-specific code.

load("//saxml:saxml.bzl", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = ["//saxml:internal"],
)

pytype_strict_library(
    name = "jax_spmd_backend",
    srcs = ["jax_spmd_backend.py"],
    deps = [
        "//saxml/server:spmd_backend",
        "//saxml/server:utils",
        "//third_party/py/absl-py/logging",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental:mesh_utils",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "servable_model",
    srcs = ["servable_model.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//saxml/server:servable_model",
        "//saxml/server:servable_model_params",
        "//third_party/py/absl-py/logging",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental",
        "//third_party/py/jaxtyping",
        "//third_party/py/numpy",
        "//third_party/py/paxml:host_callback",
        "//third_party/py/praxis:pytypes",
    ],
)

pytype_strict_library(
    name = "np_tf_sess_wrapper",
    srcs = ["np_tf_sess_wrapper.py"],
    deps = [
        "//saxml/server/tf:np_tf_sess_wrapper",
        "//third_party/py/jax",
    ],
)
