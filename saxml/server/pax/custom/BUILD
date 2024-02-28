load("//saxml:saxml.bzl", "py_strict_test", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = [
        "//saxml:internal",
    ],
)

pytype_strict_library(
    name = "servable_custom_model",
    srcs = ["servable_custom_model.py"],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        "//saxml/server/pax:servable_model",
        "//saxml/server/pax:servable_model_params",
        "//saxml/server/services:custom_service",
        "//third_party/py/numpy",
        "//third_party/py/praxis:base_model",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "servable_custom_model_test",
    srcs = ["servable_custom_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":servable_custom_model",
        "//third_party/py/absl-py/logging",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/flax:core",
        # Unused internal protobuf deps,  # Automatically added go/proto_python_upb_flip
        "//third_party/py/jax",
        "//third_party/py/numpy",
        "//third_party/py/paxml:base_experiment",
        "//third_party/py/paxml:checkpoints",
        "//third_party/py/paxml:learners",
        "//third_party/py/paxml:tasks_lib",
        "//third_party/py/praxis:base_input",
        "//third_party/py/praxis:base_layer",
        "//third_party/py/praxis:base_model",
        "//third_party/py/praxis:optimizers",
        "//third_party/py/praxis:pax_fiddle",
        "//third_party/py/praxis:py_utils",
        "//third_party/py/praxis:pytypes",
        "//third_party/py/praxis:schedules",
        "//third_party/py/praxis:test_utils",
    ],
)
