load("@rules_hdl//verilator:defs.bzl", "verilator_cc_library")
load("@rules_hdl//verilog:providers.bzl", "verilog_library")

verilog_library(
    name = "custom_mac_verilog",
    srcs = ["OpenNpuXCustomMac.sv"],
    visibility = ["//visibility:public"],
)

verilator_cc_library(
    name = "custom_mac_cc",
    module = ":custom_mac_verilog",
    module_top = "OpenNpuXCustomMac",
    systemc = False,
    visibility = ["//visibility:public"],
)
