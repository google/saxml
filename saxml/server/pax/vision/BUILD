# Vision inference service.

load("//saxml:saxml.bzl", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = [
        "//saxml:internal",
    ],
)

pytype_strict_library(
    name = "imagenet_metadata",
    srcs = ["imagenet_metadata.py"],
)

pytype_strict_library(
    name = "servable_vision_model",
    srcs = ["servable_vision_model.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//saxml/server/pax:servable_model",
        "//saxml/server/pax:servable_model_params",
        "//saxml/server/services:vision_service",
        "//third_party/py/jax",
        "//third_party/py/lingvo/core:cluster_factory",
        "//third_party/py/numpy",
        "//third_party/py/paxml:base_task",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:base_model",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "all_imports",
    srcs = ["all_imports.py"],
    deps = [
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax/vision/params:resnet",
    ],
)
