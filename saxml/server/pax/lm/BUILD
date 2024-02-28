# LM inference service.

load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_library")

package(
    # unused internal package attribute
    default_visibility = ["//saxml:internal"],
)

licenses(["notice"])

pytype_strict_library(
    name = "all_imports",
    srcs = ["all_imports.py"],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax/lm/params:c4",
        "//saxml/server/pax/lm/params:gemma",
        "//saxml/server/pax/lm/params:gptj",
        "//saxml/server/pax/lm/params:lm_cloud",
    ],
)

pytype_strict_library(
    name = "lm_tokenizer",
    srcs = ["lm_tokenizer.py"],
    srcs_version = "PY3",
    visibility = [
        "//saxml:internal",
        "//saxml/server:friends",
    ],
    deps = [
        ":vocabularies",
        "//third_party/py/praxis:base_hyperparams",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "lm_tokenizer_test",
    srcs = ["lm_tokenizer_test.py"],
    data = [
        "//saxml/server/pax/lm/test_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lm_tokenizer",
        "//third_party/py/absl-py/flags",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "servable_lm_common",
    srcs = ["servable_lm_common.py"],
    srcs_version = "PY3",
    visibility = [
        "//saxml:internal",
        "//saxml/server:friends",
    ],
    deps = [
        "//saxml/server/pax:branch_selection",
        "//saxml/server/pax:servable_model",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/orbax-export",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "servable_lm_common_test",
    srcs = ["servable_lm_common_test.py"],
    data = [
        "//saxml/server/pax/lm/test_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lm_tokenizer",
        ":servable_lm_common",
        "//third_party/py/absl-py/flags",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
        "//third_party/py/numpy",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:sample_decode",
        "//third_party/py/praxis:test_utils",
        "//saxml/server/jax:np_tf_sess_wrapper",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "servable_lm_model",
    srcs = ["servable_lm_model.py"],
    srcs_version = "PY3",
    visibility = [
        "//saxml:internal",
        "//saxml/server:friends",
    ],
    deps = [
        ":lm_tokenizer",
        ":servable_lm_common",
        "//saxml/server/jax:np_tf_sess_wrapper",
        "//saxml/server/jax:servable_model",
        "//saxml/server/pax:servable_model",
        "//saxml/server/pax:servable_model_params",
        "//saxml/server/services:lm_service",
        "//third_party/py/absl-py/logging",
        "//third_party/py/jax",
        "//third_party/py/jax:experimental",
        "//third_party/py/numpy",
        "//third_party/py/orbax-export",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:base_model",
        "//third_party/py/praxis:decoder_hparams",
        "//third_party/py/praxis:decoder_utils",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/praxis/layers:multi_query_attention",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "servable_lm_model_test",
    srcs = ["servable_lm_model_test.py"],
    data = [
        "//saxml/server/pax/lm/test_data",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "requires-net:external",
    ],
    deps = [
        ":lm_tokenizer",
        ":servable_lm_common",
        ":servable_lm_model",
        "//saxml/server/pax/lm/params:lm_cloud",
        "//third_party/py/absl-py/flags",
        "//third_party/py/absl-py/logging",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/paxml:checkpoints",
        "//third_party/py/paxml/tasks/lm/params",
        "//third_party/py/praxis:decoder_hparams",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:test_utils",
    ],
)

pytype_strict_library(
    name = "layers",
    srcs = ["layers.py"],
    deps = [
        "//third_party/py/flax:core",
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/praxis/layers",
        "//third_party/py/praxis/layers:embedding_softmax",
    ],
)

py_strict_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    shard_count = 1,
    deps = [
        ":layers",
        "//third_party/py/absl-py/logging",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/absl-py/testing:parameterized",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:test_utils",
        "//third_party/py/praxis/layers",
        "//third_party/py/praxis/layers:attentions",
    ],
)

pytype_strict_library(
    name = "vocabularies",
    srcs = ["vocabularies.py"],
    deps = [
        "//third_party/py/seqio",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
        "//third_party/py/transformers",
    ],
)

py_strict_test(
    name = "vocabularies_test",
    srcs = ["vocabularies_test.py"],
    data = [
        "//saxml/server/pax/lm/test_data",
    ],
    deps = [
        ":vocabularies",
        "//third_party/py/absl-py/flags",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "transformer_models",
    srcs = ["transformer_models.py"],
    deps = [
        ":layers",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis/layers",
        "//third_party/py/praxis/layers:multi_query_attention",
    ],
)
