# PAX implementation of servable model.

load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_library")

package(
    # unused internal package attribute
    default_visibility = [
        "//saxml:internal",
        "//saxml/server:friends",
    ],
)

licenses(["notice"])

pytype_strict_library(
    name = "all_imports",
    srcs = ["all_imports.py"],
    srcs_version = "PY3",
    tags = ["keep_dep"],
    deps = [
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax/lm:all_imports",
        "//saxml/server/pax/vision:all_imports",
    ],
)

pytype_strict_library(
    name = "branch_selection",
    srcs = ["branch_selection.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "branch_selection_test",
    srcs = ["branch_selection_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//saxml:internal"],
    deps = [
        ":branch_selection",
        "//third_party/py/absl-py/testing:absltest",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
    ],
)

pytype_strict_library(
    name = "servable_model_params",
    srcs = ["servable_model_params.py"],
    srcs_version = "PY3",
    deps = [
        "//saxml/server:servable_model_params",
        "//saxml/server:utils",
        "//third_party/py/absl-py/logging",
        "//third_party/py/jax",
        "//third_party/py/jax:mesh_utils",
        "//third_party/py/numpy",
        "//third_party/py/paxml:base_experiment",
        "//third_party/py/paxml:checkpoints",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis/layers/quantization:quantization_hparams",
    ],
)

pytype_strict_library(
    name = "servable_model",
    srcs = ["servable_model.py"],
    srcs_version = "PY3",
    deps = [
        ":branch_selection",
        ":servable_model_params",
        "//saxml/server/jax:servable_model",
        "//third_party/py/absl-py/logging",
        "//third_party/py/etils/epath",
        "//third_party/py/flax/training:checkpoints",
        "//third_party/py/jax",
        "//third_party/py/jax/experimental/array_serialization:serialization",
        "//third_party/py/numpy",
        "//third_party/py/paxml:checkpoints",
        "//third_party/py/paxml:tasks_lib",
        "//third_party/py/paxml:train_states",
        "//third_party/py/paxml:trainer_lib",
        "//third_party/py/praxis:base_hyperparams",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:base_model",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/praxis/layers/quantization:quantization_hparams",
        "//third_party/py/praxis/layers/quantization:quantize",
    ],
)

pytype_strict_library(
    name = "auto_sharding",
    srcs = ["auto_sharding.py"],
    deps = [
        "//third_party/py/jax",
    ],
)

pytype_strict_library(
    name = "quantization",
    srcs = ["quantization.py"],
    deps = [
        "//third_party/py/jax",
        "//third_party/py/praxis/layers/quantization:quantization_hparams",
        "//third_party/py/praxis/layers/quantization:quantize",
    ],
)

py_strict_test(
    name = "quantization_test",
    srcs = ["quantization_test.py"],
    deps = [
        ":quantization",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/absl-py/testing:parameterized",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
        "//third_party/py/praxis:test_utils",
        "//third_party/py/praxis/layers",
        "//third_party/py/praxis/layers/quantization:layers",
        "//third_party/py/praxis/layers/quantization:quantization_hparams",
        "//saxml/server/pax/lm/params:lm_cloud",
    ],
)

pytype_strict_library(
    name = "union_model",
    srcs = ["union_model.py"],
    srcs_version = "PY3",
    deps = [
        ":servable_model",
        ":servable_model_params",
        "//third_party/py/absl-py/logging",
        "//third_party/py/jax",
        "//third_party/py/paxml:base_task",
        "//third_party/py/paxml:checkpoints",
        "//third_party/py/praxis:base_input",
        "//third_party/py/praxis:base_model",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
    ],
)
