# Description:
#   Vision model servable parameters definition.
load("//saxml:saxml.bzl", "pytype_strict_library")

package(
    default_visibility = [
        "//saxml:internal",
    ],
)

pytype_strict_library(
    name = "resnet",
    srcs = ["resnet.py"],
    srcs_version = "PY3",
    deps = [
        "//saxml/server:servable_model_registry",
        "//saxml/server/pax/vision:imagenet_metadata",
        "//saxml/server/pax/vision:servable_vision_model",
        "//third_party/py/numpy",
        "//third_party/py/paxml/tasks/vision/params",
        "//third_party/py/praxis:base_input",
        "//third_party/py/praxis:py_utils",
    ],
)
