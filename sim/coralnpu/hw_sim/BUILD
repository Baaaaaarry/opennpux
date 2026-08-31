# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//rules:coralnpu_v2.bzl", "coralnpu_v2_binary")

cc_library(
    name = "hw_primitives",
    srcs = [
        "hw_primitives.cc",
    ],
    hdrs = [
        "hw_primitives.h",
    ],
    deps = [
        "@com_google_absl//absl/types:span",
        "@verilator//:verilator_cpp",
    ],
)

cc_library(
    name = "core_mini_axi_wrapper",
    hdrs = [
        "core_mini_axi_wrapper.h",
        "mailbox.h",
    ],
    deps = [
        ":hw_primitives",
        "//hdl/chisel/src/coralnpu:core_mini_axi_cc_library_cc",
        "//hdl/chisel/src/coralnpu:rvv_core_mini_axi_cc_library_cc",
        "//hdl/verilog:sram_backdoor",
    ],
)

cc_binary(
    name = "core_mini_axi_wrapper_example",
    srcs = [
        "core_mini_axi_wrapper_example.cc",
    ],
    data = [
        "//tests/cocotb:wfi_slot_0.elf",
    ],
    deps = [
        ":core_mini_axi_wrapper",
        "//tests/verilator_sim:elf",
    ],
)

cc_library(
    name = "coralnpu_simulator_headers",
    hdrs = [
        "coralnpu_simulator.h",
        "mailbox.h",
    ],
)

cc_library(
    name = "core_mini_axi_simulator",
    srcs = ["core_mini_axi_simulator.cc"],
    linkstatic = True,
    deps = [
        ":core_mini_axi_wrapper",
        ":coralnpu_simulator_headers",
    ],
    alwayslink = True,
)

cc_library(
    name = "core_mini_axi_simulator_rvv",
    srcs = ["core_mini_axi_simulator.cc"],
    copts = ["-DENABLE_RVV"],
    linkstatic = True,
    deps = [
        ":core_mini_axi_wrapper",
        ":coralnpu_simulator_headers",
    ],
    alwayslink = True,
)

coralnpu_v2_binary(
    name = "mailbox_example",
    srcs = [
        "mailbox_example.cc",
    ],
)

cc_binary(
    name = "core_mini_axi_simulator_example",
    srcs = [
        "core_mini_axi_simulator_example.cc",
    ],
    data = [
        ":mailbox_example.elf",
    ],
    deps = [
        ":core_mini_axi_simulator",
        "//tests/verilator_sim:elf",
    ],
)

# Generate libraries for external projects.
cc_binary(
    name = "libcoralnpu_simulator.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [":core_mini_axi_simulator"],
)

cc_binary(
    name = "libcoralnpu_simulator_rvv.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [":core_mini_axi_simulator_rvv"],
)

cc_library(
    name = "gem5_npu_submission_headers",
    hdrs = [
        "gem5_bridge/npu_functional_request.h",
        "gem5_bridge/npu_inference_io.h",
        "gem5_bridge/npu_submission.h",
        "gem5_bridge/npu_numerical_result.h",
        "gem5_bridge/npu_page_bundle.h",
        "gem5_bridge/npu_weight_queue.h",
        "gem5_bridge/npu_weight_residency.h",
    ],
)

cc_library(
    name = "gem5_tensor_plan_runtime",
    srcs = ["gem5_bridge/opennpux/npu_tensor_plan.c"],
    hdrs = ["gem5_bridge/opennpux/npu_tensor_plan.h"],
    includes = ["gem5_bridge"],
)

cc_library(
    name = "gem5_host_tensor_arena",
    srcs = ["gem5_bridge/gem5_host_tensor_arena.cc"],
    hdrs = ["gem5_bridge/gem5_host_tensor_arena.h"],
    deps = [":gem5_tensor_plan_runtime"],
)

cc_library(
    name = "gem5_functional_materializer_runtime",
    srcs = [
        "gem5_bridge/opennpux/npu_functional_materializer.c",
        "gem5_bridge/opennpux/npu_submission.c",
    ],
    hdrs = [
        "gem5_bridge/opennpux/npu_functional_materializer.h",
        "gem5_bridge/opennpux/npu_functional_request.h",
        "gem5_bridge/opennpux/npu_submission.h",
        "gem5_bridge/opennpux/npu_weight_ranges.h",
    ],
    includes = ["gem5_bridge"],
    deps = [":gem5_tensor_plan_runtime"],
)

cc_library(
    name = "gem5_xgraph_lowering_runtime",
    srcs = [
        "gem5_bridge/opennpux/npu_gptq_tile_plan.c",
        "gem5_bridge/opennpux/npu_xgraph_lowering.c",
    ],
    hdrs = [
        "gem5_bridge/opennpux/npu_functional_request.h",
        "gem5_bridge/opennpux/npu_gptq_tile_plan.h",
        "gem5_bridge/opennpux/npu_submission.h",
        "gem5_bridge/opennpux/npu_xgraph_lowering.h",
        "gem5_bridge/opennpux/xopennpux_graph.h",
    ],
    includes = ["gem5_bridge"],
    deps = [":gem5_npu_submission_headers"],
)

cc_library(
    name = "gem5_generic_command_dispatch_headers",
    hdrs = [
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/gem5_generic_command_dispatch.h",
    ],
    deps = [":gem5_npu_submission_headers"],
)

cc_library(
    name = "gem5_transformer_kernel_headers",
    hdrs = ["gem5_bridge/gem5_transformer_kernels.h"],
)

cc_library(
    name = "gem5_host_functional_graph",
    srcs = ["gem5_bridge/gem5_host_functional_graph.cc"],
    hdrs = ["gem5_bridge/gem5_host_functional_graph.h"],
    deps = [
        ":gem5_functional_materializer_runtime",
        ":gem5_generic_command_dispatch_headers",
        ":gem5_host_routed_expert",
        ":gem5_host_tensor_arena",
        ":gem5_transformer_kernel_headers",
        ":gem5_xgraph_lowering_runtime",
    ],
)

cc_library(
    name = "gem5_xgraph_lowering_audit",
    srcs = ["gem5_bridge/gem5_xgraph_lowering_audit.cc"],
    hdrs = ["gem5_bridge/gem5_xgraph_lowering_audit.h"],
    deps = [
        ":gem5_host_functional_graph",
        ":gem5_xgraph_lowering_runtime",
    ],
)

cc_test(
    name = "gem5_xgraph_lowering_audit_test",
    srcs = ["gem5_bridge/gem5_xgraph_lowering_audit_test.cc"],
    deps = [
        ":gem5_npu_submission_headers",
        ":gem5_xgraph_lowering_audit",
    ],
)

cc_library(
    name = "gem5_model_weight_runtime",
    srcs = [
        "gem5_bridge/opennpux/model_package.c",
        "gem5_bridge/opennpux/npu_gptq_weights.c",
        "gem5_bridge/opennpux/npu_weight_ranges.c",
    ],
    hdrs = [
        "gem5_bridge/opennpux/model_package.h",
        "gem5_bridge/opennpux/npu_gptq_weights.h",
        "gem5_bridge/opennpux/npu_submission.h",
        "gem5_bridge/opennpux/npu_weight_ranges.h",
    ],
    includes = ["gem5_bridge"],
)

cc_library(
    name = "gem5_host_weight_provider",
    srcs = ["gem5_bridge/gem5_host_weight_provider.cc"],
    hdrs = ["gem5_bridge/gem5_host_weight_provider.h"],
    deps = [
        ":gem5_generic_gptq_executor_headers",
        ":gem5_model_weight_runtime",
        ":gem5_npu_submission_headers",
    ],
)

cc_library(
    name = "gem5_generic_gptq_executor_headers",
    hdrs = [
        "gem5_bridge/gem5_generic_gptq_executor.h",
        "gem5_bridge/gem5_gptq_kernels.h",
    ],
    deps = [":gem5_npu_submission_headers"],
)

cc_library(
    name = "gem5_host_routed_expert",
    srcs = ["gem5_bridge/gem5_host_routed_expert.cc"],
    hdrs = ["gem5_bridge/gem5_host_routed_expert.h"],
    deps = [
        ":gem5_generic_gptq_executor_headers",
        ":gem5_host_weight_provider",
        ":gem5_xgraph_lowering_runtime",
    ],
)

cc_binary(
    name = "libcoralnpu_gem5_bridge.so",
    srcs = [
        "gem5_bridge/coralnpu_gem5_abi.cc",
        "gem5_bridge/coralnpu_gem5_abi.h",
        "gem5_bridge/coral_mobilenet.h",
        "gem5_bridge/coral_gptq_matmul.h",
        "gem5_bridge/coral_gptq_expert.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_gptq_paged.h",
        "gem5_bridge/gem5_axi_master_drivers.h",
        "gem5_bridge/gem5_core_mini_axi_wrapper.h",
        "gem5_bridge/gem5_coprocessor_command.cc",
        "gem5_bridge/gem5_coprocessor_command.h",
        "gem5_bridge/gem5_custom_mac.h",
        "gem5_bridge/gem5_dma_request_builder.h",
        "gem5_bridge/gem5_generic_gptq_executor.cc",
        "gem5_bridge/gem5_generic_gptq_executor.h",
        "gem5_bridge/gem5_generic_command_dispatch.cc",
        "gem5_bridge/gem5_generic_command_dispatch.h",
        "gem5_bridge/gem5_gptq_kernels.cc",
        "gem5_bridge/gem5_gptq_kernels.h",
        "gem5_bridge/gem5_host_functional_backend.cc",
        "gem5_bridge/gem5_host_functional_backend.h",
        "gem5_bridge/gem5_hybrid_kernels.cc",
        "gem5_bridge/gem5_hybrid_kernels.h",
        "gem5_bridge/gem5_hybrid_operator.cc",
        "gem5_bridge/gem5_hybrid_operator.h",
        "gem5_bridge/gem5_sim_host_pager.cc",
        "gem5_bridge/gem5_sim_host_pager.h",
        "gem5_bridge/gem5_sim_host_functional.cc",
        "gem5_bridge/gem5_sim_host_functional.h",
        "gem5_bridge/gem5_sim_host_numerical.cc",
        "gem5_bridge/gem5_sim_host_numerical.h",
        "gem5_bridge/gem5_tmma_coprocessor.cc",
        "gem5_bridge/gem5_tmma_coprocessor.h",
        "gem5_bridge/xopennpux_isa.h",
        "gem5_bridge/gem5_transformer_kernels.cc",
        "gem5_bridge/gem5_transformer_kernels.h",
        "gem5_bridge/qwen_device_inference.h",
        "gem5_bridge/npu_functional_request.h",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":gem5_host_functional_graph",
        ":gem5_host_tensor_arena",
        ":gem5_host_weight_provider",
        ":gem5_npu_submission_headers",
        ":hw_primitives",
        "//hdl/chisel/src/coralnpu:core_mini_axi_cc_library_cc",
        "//hdl/verilog/opennpux:custom_mac_cc",
        "//hdl/verilog:sram_backdoor",
    ],
)

cc_binary(
    name = "libcoralnpu_gem5_rvv_highmem_bridge.so",
    srcs = [
        "gem5_bridge/coralnpu_gem5_abi.cc",
        "gem5_bridge/coralnpu_gem5_abi.h",
        "gem5_bridge/coral_mobilenet.h",
        "gem5_bridge/coral_gptq_matmul.h",
        "gem5_bridge/coral_gptq_expert.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_gptq_paged.h",
        "gem5_bridge/gem5_axi_master_drivers.h",
        "gem5_bridge/gem5_core_mini_axi_wrapper.h",
        "gem5_bridge/gem5_coprocessor_command.cc",
        "gem5_bridge/gem5_coprocessor_command.h",
        "gem5_bridge/gem5_custom_mac.h",
        "gem5_bridge/gem5_dma_request_builder.h",
        "gem5_bridge/gem5_generic_gptq_executor.cc",
        "gem5_bridge/gem5_generic_gptq_executor.h",
        "gem5_bridge/gem5_generic_command_dispatch.cc",
        "gem5_bridge/gem5_generic_command_dispatch.h",
        "gem5_bridge/gem5_gptq_kernels.cc",
        "gem5_bridge/gem5_gptq_kernels.h",
        "gem5_bridge/gem5_host_functional_backend.cc",
        "gem5_bridge/gem5_host_functional_backend.h",
        "gem5_bridge/gem5_hybrid_kernels.cc",
        "gem5_bridge/gem5_hybrid_kernels.h",
        "gem5_bridge/gem5_hybrid_operator.cc",
        "gem5_bridge/gem5_hybrid_operator.h",
        "gem5_bridge/gem5_sim_host_pager.cc",
        "gem5_bridge/gem5_sim_host_pager.h",
        "gem5_bridge/gem5_sim_host_functional.cc",
        "gem5_bridge/gem5_sim_host_functional.h",
        "gem5_bridge/gem5_sim_host_numerical.cc",
        "gem5_bridge/gem5_sim_host_numerical.h",
        "gem5_bridge/gem5_tmma_coprocessor.cc",
        "gem5_bridge/gem5_tmma_coprocessor.h",
        "gem5_bridge/xopennpux_isa.h",
        "gem5_bridge/gem5_transformer_kernels.cc",
        "gem5_bridge/gem5_transformer_kernels.h",
        "gem5_bridge/qwen_device_inference.h",
        "gem5_bridge/npu_functional_request.h",
    ],
    copts = ["-DCORAL_GEM5_RVV_HIGHMEM"],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        ":gem5_host_functional_graph",
        ":gem5_host_tensor_arena",
        ":gem5_host_weight_provider",
        ":gem5_npu_submission_headers",
        ":hw_primitives",
        "//hdl/chisel/src/coralnpu:rvv_core_mini_highmem_axi_cc_library_cc",
        "//hdl/verilog/opennpux:custom_mac_cc",
        "//hdl/verilog:sram_backdoor",
    ],
)

cc_test(
    name = "gem5_coprocessor_command_test",
    srcs = [
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/gem5_coprocessor_command.cc",
        "gem5_bridge/gem5_coprocessor_command.h",
        "gem5_bridge/gem5_coprocessor_command_test.cc",
    ],
)

cc_test(
    name = "gem5_tmma_coprocessor_test",
    srcs = [
        "gem5_bridge/gem5_tmma_coprocessor.cc",
        "gem5_bridge/gem5_tmma_coprocessor.h",
        "gem5_bridge/gem5_tmma_coprocessor_test.cc",
        "gem5_bridge/xopennpux_isa.h",
    ],
)

cc_test(
    name = "gem5_sim_host_pager_test",
    srcs = [
        "gem5_bridge/gem5_sim_host_pager.cc",
        "gem5_bridge/gem5_sim_host_pager.h",
        "gem5_bridge/gem5_sim_host_pager_test.cc",
    ],
    deps = [":gem5_npu_submission_headers"],
)

cc_test(
    name = "gem5_sim_host_numerical_test",
    srcs = [
        "gem5_bridge/gem5_sim_host_numerical.cc",
        "gem5_bridge/gem5_sim_host_numerical.h",
        "gem5_bridge/gem5_sim_host_numerical_test.cc",
    ],
    deps = [":gem5_npu_submission_headers"],
)

cc_test(
    name = "gem5_transformer_kernels_test",
    srcs = [
        "gem5_bridge/gem5_transformer_kernels.cc",
        "gem5_bridge/gem5_transformer_kernels.h",
        "gem5_bridge/gem5_transformer_kernels_test.cc",
    ],
)

cc_test(
    name = "gem5_host_functional_backend_test",
    srcs = [
        "gem5_bridge/gem5_generic_gptq_executor.cc",
        "gem5_bridge/gem5_generic_gptq_executor.h",
        "gem5_bridge/gem5_gptq_kernels.cc",
        "gem5_bridge/gem5_gptq_kernels.h",
        "gem5_bridge/gem5_host_functional_backend.cc",
        "gem5_bridge/gem5_host_functional_backend.h",
        "gem5_bridge/gem5_host_functional_backend_test.cc",
        "gem5_bridge/gem5_transformer_kernels.cc",
        "gem5_bridge/gem5_transformer_kernels.h",
    ],
    deps = [":gem5_npu_submission_headers"],
)

cc_test(
    name = "gem5_generic_command_dispatch_test",
    srcs = [
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/gem5_generic_command_dispatch.cc",
        "gem5_bridge/gem5_generic_command_dispatch.h",
        "gem5_bridge/gem5_generic_command_dispatch_test.cc",
        "gem5_bridge/gem5_generic_gptq_executor.cc",
        "gem5_bridge/gem5_generic_gptq_executor.h",
        "gem5_bridge/gem5_gptq_kernels.cc",
        "gem5_bridge/gem5_gptq_kernels.h",
        "gem5_bridge/gem5_host_functional_backend.cc",
        "gem5_bridge/gem5_host_functional_backend.h",
        "gem5_bridge/gem5_transformer_kernels.cc",
        "gem5_bridge/gem5_transformer_kernels.h",
        "gem5_bridge/npu_functional_request.h",
    ],
    deps = [":gem5_npu_submission_headers"],
)

cc_test(
    name = "gem5_npu_launch_descriptor_test",
    srcs = [
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/gem5_npu_launch_descriptor.h",
        "gem5_bridge/gem5_npu_launch_descriptor_test.cc",
    ],
)

cc_test(
    name = "gem5_custom_mac_test",
    srcs = [
        "gem5_bridge/gem5_custom_mac.h",
        "gem5_bridge/gem5_custom_mac_test.cc",
    ],
    deps = ["//hdl/verilog/opennpux:custom_mac_cc"],
)

cc_test(
    name = "gem5_gptq_kernels_test",
    srcs = [
        "gem5_bridge/gem5_gptq_kernels.cc",
        "gem5_bridge/gem5_gptq_kernels.h",
        "gem5_bridge/gem5_gptq_kernels_test.cc",
    ],
)

cc_test(
    name = "gem5_generic_gptq_executor_test",
    srcs = [
        "gem5_bridge/gem5_generic_gptq_executor.cc",
        "gem5_bridge/gem5_generic_gptq_executor.h",
        "gem5_bridge/gem5_generic_gptq_executor_test.cc",
        "gem5_bridge/gem5_gptq_kernels.cc",
        "gem5_bridge/gem5_gptq_kernels.h",
    ],
    deps = [":gem5_npu_submission_headers"],
)

cc_test(
    name = "gem5_hybrid_kernels_test",
    srcs = [
        "gem5_bridge/coral_gptq_expert.h",
        "gem5_bridge/coral_gptq_matmul.h",
        "gem5_bridge/coral_gptq_paged.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/gem5_generic_gptq_executor.cc",
        "gem5_bridge/gem5_generic_gptq_executor.h",
        "gem5_bridge/gem5_generic_command_dispatch.cc",
        "gem5_bridge/gem5_generic_command_dispatch.h",
        "gem5_bridge/gem5_gptq_kernels.cc",
        "gem5_bridge/gem5_gptq_kernels.h",
        "gem5_bridge/gem5_host_functional_backend.cc",
        "gem5_bridge/gem5_host_functional_backend.h",
        "gem5_bridge/gem5_hybrid_kernels.cc",
        "gem5_bridge/gem5_hybrid_kernels.h",
        "gem5_bridge/gem5_hybrid_kernels_test.cc",
        "gem5_bridge/gem5_hybrid_operator.cc",
        "gem5_bridge/gem5_hybrid_operator.h",
        "gem5_bridge/gem5_transformer_kernels.cc",
        "gem5_bridge/gem5_transformer_kernels.h",
        "gem5_bridge/npu_functional_request.h",
        "gem5_bridge/qwen_device_inference.h",
    ],
    deps = [":gem5_npu_submission_headers"],
)

cc_test(
    name = "gem5_axi_master_drivers_test",
    srcs = [
        "gem5_bridge/coralnpu_gem5_abi.h",
        "gem5_bridge/gem5_axi_master_drivers.h",
        "gem5_bridge/gem5_axi_master_drivers_test.cc",
        "gem5_bridge/gem5_dma_request_builder.h",
    ],
    deps = [
        ":hw_primitives",
    ],
)

coralnpu_v2_binary(
    name = "gem5_smoke_halt",
    srcs = [
        "gem5_bridge/gem5_smoke_halt.c",
    ],
)

coralnpu_v2_binary(
    name = "gem5_tmma_smoke",
    srcs = [
        "gem5_bridge/gem5_tmma_smoke.c",
        "gem5_bridge/xopennpux_intrinsics.h",
    ],
)

coralnpu_v2_binary(
    name = "gem5_tmma_operator_e2e",
    srcs = [
        "gem5_bridge/gem5_tmma_operator_e2e.c",
        "gem5_bridge/xopennpux_intrinsics.h",
        "gem5_bridge/xopennpux_ops.h",
    ],
)

coralnpu_v2_binary(
    name = "gem5_dma_smoke",
    srcs = [
        "gem5_bridge/gem5_dma_smoke.c",
    ],
)

coralnpu_v2_binary(
    name = "gem5_command_smoke",
    srcs = [
        "gem5_bridge/coral_command.h",
        "gem5_bridge/gem5_command_smoke.c",
    ],
)

coralnpu_v2_binary(
    name = "gem5_qwen_tcb_smoke",
    srcs = [
        "gem5_bridge/gem5_qwen_tcb_smoke.c",
        "gem5_bridge/qwen_tcb.h",
    ],
)

coralnpu_v2_binary(
    name = "gem5_qwen_device_infer",
    srcs = [
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_operator_client.h",
        "gem5_bridge/gem5_qwen_device_infer.cc",
        "gem5_bridge/qwen_device_inference.h",
    ],
    dtcm_size_kbytes = 64,
    itcm_size_kbytes = 64,
    stack_size_bytes = 4096,
    visibility = ["//visibility:public"],
)

coralnpu_v2_binary(
    name = "gem5_gptq_matmul_smoke",
    srcs = [
        "gem5_bridge/coral_gptq_matmul.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_operator_client.h",
        "gem5_bridge/gem5_gptq_matmul_smoke.cc",
    ],
    dtcm_size_kbytes = 64,
    itcm_size_kbytes = 64,
    stack_size_bytes = 4096,
    visibility = ["//visibility:public"],
)

coralnpu_v2_binary(
    name = "gem5_gptq_projection",
    srcs = [
        "gem5_bridge/coral_gptq_matmul.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_operator_client.h",
        "gem5_bridge/gem5_gptq_projection.cc",
    ],
    dtcm_size_kbytes = 64,
    itcm_size_kbytes = 64,
    stack_size_bytes = 4096,
    visibility = ["//visibility:public"],
)

coralnpu_v2_binary(
    name = "gem5_gptq_expert",
    srcs = [
        "gem5_bridge/coral_gptq_expert.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_operator_client.h",
        "gem5_bridge/gem5_gptq_expert.cc",
    ],
    dtcm_size_kbytes = 64,
    itcm_size_kbytes = 64,
    stack_size_bytes = 4096,
    visibility = ["//visibility:public"],
)

coralnpu_v2_binary(
    name = "gem5_npu_command_processor_smoke",
    srcs = [
        "gem5_bridge/coral_gptq_paged.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/gem5_npu_command_processor_smoke.c",
        "gem5_bridge/npu_inference_io.h",
        "gem5_bridge/npu_route_table.h",
        "gem5_bridge/npu_submission.h",
        "gem5_bridge/npu_weight_queue.h",
        "gem5_bridge/npu_weight_residency.h",
    ],
    # The paged command path suspends inside page_weight() while the host
    # services a fault.  Its saved registers, 64-bit temporaries, and stacked
    # call arguments do not fit in coralnpu_v2_binary's 128-byte default.
    stack_size_bytes = 4096,
)

coralnpu_v2_binary(
    name = "gem5_mobilenet",
    srcs = [
        "gem5_bridge/coral_mobilenet.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_operator_client.h",
        "gem5_bridge/gem5_mobilenet.cc",
    ],
    dtcm_size_kbytes = 1024,
    itcm_size_kbytes = 1024,
    stack_size_bytes = 65536,
    visibility = ["//visibility:public"],
    deps = [
        "//sw/opt:rvv_opt",
        "//sw/opt/litert-micro:conv",
        "//sw/opt/litert-micro:depthwise_conv",
        "//sw/utils:utils",
        "//tests/cocotb/tutorial/tfmicro:mobilenet_v1_025_224_int8_dummy_lib",
        "@tflite_micro//tensorflow/lite/micro:micro_framework",
        "@tflite_micro//tensorflow/lite/micro:micro_log",
        "@tflite_micro//tensorflow/lite/micro:op_resolvers",
        "@tflite_micro//tensorflow/lite/micro:system_setup",
    ],
)

coralnpu_v2_binary(
    name = "gem5_mobilenet_partial",
    srcs = [
        "gem5_bridge/coral_mobilenet.h",
        "gem5_bridge/gem5_mobilenet_partial.c",
    ],
    dtcm_size_kbytes = 64,
    itcm_size_kbytes = 64,
    stack_size_bytes = 4096,
    visibility = ["//visibility:public"],
)

coralnpu_v2_binary(
    name = "gem5_npu_launch_smoke",
    srcs = [
        "gem5_bridge/coral_mobilenet.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_operator_client.h",
        "gem5_bridge/gem5_npu_launch_descriptor.h",
        "gem5_bridge/gem5_npu_launch_smoke.cc",
    ],
    dtcm_size_kbytes = 64,
    itcm_size_kbytes = 64,
    stack_size_bytes = 4096,
    visibility = ["//visibility:public"],
)

coralnpu_v2_binary(
    name = "gem5_generic_launch_smoke",
    srcs = [
        "gem5_bridge/coral_generic_test.h",
        "gem5_bridge/coral_operator.h",
        "gem5_bridge/coral_operator_client.h",
        "gem5_bridge/gem5_generic_launch_smoke.cc",
        "gem5_bridge/npu_functional_request.h",
        "gem5_bridge/npu_submission.h",
    ],
    dtcm_size_kbytes = 64,
    itcm_size_kbytes = 64,
    stack_size_bytes = 4096,
    visibility = ["//visibility:public"],
)

coralnpu_v2_binary(
    name = "gem5_qwen_command_flow_smoke",
    srcs = [
        "gem5_bridge/coral_generic_test.h",
        "gem5_bridge/gem5_qwen_command_flow_smoke.cc",
        "gem5_bridge/opennpux/xopennpux_graph.h",
        "gem5_bridge/xopennpux_intrinsics.h",
        "gem5_bridge/xopennpux_ops.h",
    ],
    dtcm_size_kbytes = 64,
    itcm_size_kbytes = 64,
    stack_size_bytes = 4096,
    visibility = ["//visibility:public"],
)
