package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sample_generation",
    srcs = glob(["*.cc"]),
    hdrs = glob(["*.h"]),
)

cc_library(
    name = "algorithm",
    hdrs = ["algorithm.h"],
    deps = [
        ":pj",
        ":pmj",
        ":pmj02",
    ]
)

cc_library(
    name = "select_subquad",
    srcs = ["select_subquad.cc"],
    hdrs = ["select_subquad.h"],
    deps = [":util"]
)

cc_library(
    name = "pj",
    srcs = ["pj.cc"],
    hdrs = ["pj.h"],
    deps = [":util"]
)

cc_library(
    name = "pmj",
    srcs = ["pmj.cc"],
    hdrs = ["pmj.h"],
    deps = [
        ":select_subquad",
        ":util",
    ]
)

cc_library(
    name = "pmj02",
    srcs = ["pmj02.cc"],
    hdrs = ["pmj02.h"],
    deps = [
        ":pmj02_util",
        ":select_subquad",
        ":util",
    ]
)

cc_library(
    name = "pmj02_util",
    srcs = ["pmj02_util.cc"],
    hdrs = ["pmj02_util.h"],
    deps = []
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = []
)