# LM inference service.

load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = [
        "//saxml:internal",
    ],
)

pytype_strict_library(
    name = "all_imports",
    srcs = ["all_imports.py"],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        ":lm_service",
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax/lm/params:lm_cloud",
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
        "//saxml/server/pax:servable_model_params",
        "//third_party/py/numpy",
        "//third_party/py/paxml:checkpoint_py_pb2",
    ],
)

pytype_strict_library(
    name = "servable_lm_model",
    srcs = ["servable_lm_model.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":lm_service",
        "//saxml/server/jax:np_tf_sess_wrapper",
        "//saxml/server/pax:servable_model",
        "//saxml/server/pax:servable_model_params",
        "//third_party/py/absl-py/logging",
        "//third_party/py/jax",
        "//third_party/py/jax:experimental",
        "//third_party/py/lingvo/core:tokenizers",  # buildcleaner: keep
        "//third_party/py/numpy",
        "//third_party/py/paxml:checkpoint_py_pb2",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:base_model",
        "//third_party/py/praxis:decoder_hparams",
        "//third_party/py/praxis:decoder_utils",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "lm_tokenizer",
    srcs = ["lm_tokenizer.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/praxis:base_hyperparams",
        "//third_party/py/seqio",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
