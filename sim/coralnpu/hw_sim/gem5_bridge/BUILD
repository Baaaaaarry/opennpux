cc_binary(
    name = "libcoralnpu_gem5_bridge.so",
    srcs = [
        "coralnpu_gem5_abi.cc",
        "coralnpu_gem5_abi.h",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "//hw_sim:core_mini_axi_wrapper",
    ],
)
