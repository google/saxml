# TensorFlow-related code.

load("//saxml:saxml.bzl", "py_strict_test", "pybind_extension", "pytype_strict_library")

licenses(["notice"])

package(
    default_visibility = [
        "//saxml:internal",
    ],
)

pytype_strict_library(
    name = "np_tf_sess_wrapper",
    srcs = ["np_tf_sess_wrapper.py"],
    pybind_deps = [":tf_session_runner"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/absl-py/logging",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

py_strict_test(
    name = "np_tf_sess_wrapper_test",
    srcs = ["np_tf_sess_wrapper_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["manual"],
    deps = [
        ":np_tf_sess_wrapper",
        "//third_party/py/absl-py/testing:absltest",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

pybind_extension(
    name = "tf_session_runner",
    srcs = ["tf_session_runner.cc"],
    deps = [
        ":np_conversions",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core/platform:status",
        "@pybind11_abseil//pybind11_abseil:import_status_module",
        "@pybind11_abseil//pybind11_abseil:status_casters",
        "@pybind11_protobuf//pybind11_protobuf:native_proto_caster",
    ],
)

cc_library(
    name = "np_conversions",
    srcs = ["np_conversions.cc"],
    hdrs = ["np_conversions.h"],
    deps = [
        ":numpy_headers",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@pybind11",  # buildcleaner: keep
    ],
)

cc_library(
    name = "numpy_headers",
    hdrs = [
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/__multiarray_api.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/_neighborhood_iterator_imp.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/_numpyconfig.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/arrayobject.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/ndarrayobject.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/ndarraytypes.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/npy_1_7_deprecated_api.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/npy_common.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/npy_cpu.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/npy_endian.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/npy_interrupt.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/numpyconfig.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/old_defines.h",
        "@third_party_numpy//:site-packages/numpy/core/include/numpy/utils.h",
    ],
    strip_include_prefix = "/site-packages/numpy/core/include",
)
