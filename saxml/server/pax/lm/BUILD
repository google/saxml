# LM inference service.

load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = [
        "//saxml:internal",
    ],
)

filegroup(
    name = "test_data",
    srcs = glob(["test_data/*"]),
)

pytype_strict_library(
    name = "all_imports",
    srcs = ["all_imports.py"],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax/lm/params:lm_cloud",
    ],
)

pytype_strict_library(
    name = "lm_tokenizer",
    srcs = ["lm_tokenizer.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/praxis:base_hyperparams",
        "//third_party/py/seqio",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "lm_tokenizer_test",
    srcs = ["lm_tokenizer_test.py"],
    data = [":test_data"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lm_tokenizer",
        "//third_party/py/absl-py/flags",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "servable_lm_common",
    srcs = ["servable_lm_common.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//saxml/server/pax:branch_selection",
        "//saxml/server/pax:servable_model",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "servable_lm_common_test",
    srcs = ["servable_lm_common_test.py"],
    data = [":test_data"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lm_tokenizer",
        ":servable_lm_common",
        "//saxml/server/jax:np_tf_sess_wrapper",
        "//third_party/py/absl-py/flags",
        "//third_party/py/numpy",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:test_utils",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "servable_lm_model",
    srcs = ["servable_lm_model.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":lm_tokenizer",
        ":servable_lm_common",
        "//saxml/server/jax:np_tf_sess_wrapper",
        "//saxml/server/pax:servable_model",
        "//saxml/server/pax:servable_model_params",
        "//saxml/server/services:lm_service",
        "//third_party/py/absl-py/logging",
        "//third_party/py/jax",
        "//third_party/py/jax:experimental",
        "//third_party/py/numpy",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:base_model",
        "//third_party/py/praxis:decoder_hparams",
        "//third_party/py/praxis:decoder_utils",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
