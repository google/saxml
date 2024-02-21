# Description:
#   Language model servable parameters definition.

load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_library")

package(
    # unused internal package attribute
    default_visibility = [
        "//saxml:internal",
        "//saxml/server:friends",
    ],
)

licenses(["notice"])

filegroup(
    name = "test_model",
    srcs = ["test_model.model"],
)

pytype_strict_library(
    name = "template",
    srcs = ["template.py"],
    data = [":test_model"],
    srcs_version = "PY3",
    deps = [
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax/lm:lm_tokenizer",
        "//saxml/server/pax/lm:servable_lm_model",
        "//third_party/py/numpy",
        "//third_party/py/paxml:base_task",
        "//third_party/py/paxml:tasks_lib",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:decoder_hparams",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:token_samplers",
        "//third_party/py/praxis/layers:attentions",
        "//third_party/py/praxis/layers:multi_query_attention",
        "//third_party/py/praxis/layers:transformer_models",
        "//third_party/py/praxis/layers:transformers",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "template_test",
    srcs = ["template_test.py"],
    deps = [
        ":lm_cloud",
        ":template",
        "//third_party/py/google/protobuf:use_fast_cpp_protos",  # Automatically added go/proto_python_upb_flip
        "//third_party/py/praxis:test_utils",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pytype_strict_library(
    name = "lm_cloud",
    srcs = ["lm_cloud.py"],
    srcs_version = "PY3",
    deps = [
        ":template",
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax:quantization",
        "//saxml/server/pax/lm:layers",
        "//third_party/py/jax",
        "//third_party/py/paxml:base_experiment",
        "//third_party/py/paxml:tasks_lib",
        "//third_party/py/paxml/tasks/lm/params:lm_cloud",
        "//third_party/py/praxis:base_input",
        "//third_party/py/praxis:optimizers",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:schedules",
        "//third_party/py/praxis/layers",
        "//third_party/py/praxis/layers:activations",
        "//third_party/py/praxis/layers:multi_query_attention",
    ],
)

pytype_strict_library(
    name = "c4",
    srcs = ["c4.py"],
    deps = [
        ":template",
        "//saxml/server:servable_model_registry",
        "//third_party/py/paxml/tasks/lm/params",
    ],
)

pytype_strict_library(
    name = "gptj",
    srcs = ["gptj.py"],
    deps = [
        ":template",
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax/lm:layers",
        "//third_party/py/jax",
        "//third_party/py/paxml:base_experiment",
        "//third_party/py/paxml:tasks_lib",
        "//third_party/py/praxis:base_input",
        "//third_party/py/praxis:optimizers",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:schedules",
        "//third_party/py/praxis/layers",
        "//third_party/py/praxis/layers:activations",
    ],
)

pytype_strict_library(
    name = "gemma",
    srcs = ["gemma.py"],
    srcs_version = "PY3",
    deps = [
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax:quantization",
        "//saxml/server/pax/lm:transformer_models",
        "//saxml/server/pax/lm/params:template",
        "//third_party/py/jax",
        "//third_party/py/paxml:base_experiment",
        "//third_party/py/paxml:tasks_lib",
        "//third_party/py/praxis:base_input",
        "//third_party/py/praxis:optimizers",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:schedules",
        "//third_party/py/praxis/layers",
    ],
)
