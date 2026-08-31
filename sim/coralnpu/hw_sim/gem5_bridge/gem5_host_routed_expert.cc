#include "hw_sim/gem5_bridge/gem5_host_routed_expert.h"

#include <vector>

#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"
#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"

bool RunGem5HostRoutedExpert(
    const void* operator_parameters, uint32_t rows, const float* input,
    size_t input_bytes, const uint32_t* expert_ids,
    const float* route_weights, uint32_t active_experts, float* output,
    size_t output_bytes, Gem5HostWeightProvider* provider,
    Gem5HostRoutedExpertStats* stats) {
  if (operator_parameters == nullptr || rows == 0 || input == nullptr ||
      expert_ids == nullptr || route_weights == nullptr ||
      active_experts == 0 || output == nullptr || provider == nullptr ||
      stats == nullptr) {
    return false;
  }
  const uint64_t route_count =
      static_cast<uint64_t>(rows) * active_experts;
  if (route_count > SIZE_MAX / sizeof(uint64_t)) {
    return false;
  }
  std::vector<uint64_t> expanded_ids;
  try {
    expanded_ids.assign(expert_ids, expert_ids + route_count);
  } catch (...) {
    return false;
  }
  Gem5GptqKernelStats kernel_stats = {};
  const auto& parameters = *static_cast<
      const opennpux_npu_operator_parameters*>(operator_parameters);
  if (!RunGem5RoutedGptqExperts(
          parameters, rows, {input, input_bytes}, expanded_ids.data(),
          route_weights, active_experts,
          Gem5HostWeightProvider::ProvideRoutedExpert, provider,
          {output, output_bytes}, &kernel_stats)) {
    return false;
  }
  *stats = {kernel_stats.operations, kernel_stats.bytes_read,
            kernel_stats.bytes_written, kernel_stats.modeled_cycles,
            route_count, route_count};
  return true;
}

bool RunGem5XGraphRoutedExpert(
    const opennpux_xgraph_command& command, const void* operator_parameters,
    uint32_t extmem_base, uint8_t* extmem, size_t extmem_size,
    Gem5HostWeightProvider* provider, Gem5HostRoutedExpertStats* stats) {
  (void)extmem_base;
  if (command.opcode != OPENNPUX_XGRAPH_OP_TROUTED_EXPERT ||
      command.flags != OPENNPUX_XGRAPH_TROUTED_EXPERT_WEIGHT_PLAN ||
      command.data_type != OPENNPUX_XGRAPH_DTYPE_FP32 || command.dim0 == 0 ||
      command.dim1 == 0 || command.dim2 == 0 || command.scalar0 == 0 ||
      command.reserved[2] == 0 || operator_parameters == nullptr ||
      extmem == nullptr || provider == nullptr || stats == nullptr) {
    return false;
  }
  const auto& parameters =
      *static_cast<const opennpux_npu_operator_parameters*>(
          operator_parameters);
  const uint32_t quantization_bits = command.reserved[3] & 0xffu;
  const uint32_t quantization_group_size = command.reserved[3] >> 8;
  if (parameters.input_features != command.dim1 ||
      parameters.intermediate_features != command.dim2 ||
      parameters.output_features != command.reserved[2] ||
      parameters.quantization_bits != quantization_bits ||
      parameters.quantization_group_size != quantization_group_size ||
      parameters.scale_data_type != command.reserved[4]) {
    return false;
  }

  const uint64_t input_bytes = static_cast<uint64_t>(command.dim0) *
                               command.dim1 * sizeof(float);
  const uint64_t route_count =
      static_cast<uint64_t>(command.dim0) * command.scalar0;
  const uint64_t route_bytes = route_count * sizeof(float);
  const uint64_t output_bytes = static_cast<uint64_t>(command.dim0) *
                                command.reserved[2] * sizeof(float);
  const auto translate = [&](uint32_t offset, uint64_t bytes) -> uint8_t* {
    if (offset > extmem_size || bytes > extmem_size - offset) {
      return nullptr;
    }
    return extmem + offset;
  };
  auto* input = reinterpret_cast<const float*>(
      translate(command.source0_offset, input_bytes));
  auto* expert_ids = reinterpret_cast<const uint32_t*>(
      translate(command.source1_offset, route_bytes));
  auto* route_weights = reinterpret_cast<const float*>(
      translate(command.reserved[0], route_bytes));
  auto* output = reinterpret_cast<float*>(
      translate(command.destination_offset, output_bytes));
  if (input == nullptr || expert_ids == nullptr || route_weights == nullptr ||
      output == nullptr) {
    return false;
  }
  return RunGem5HostRoutedExpert(
      operator_parameters, command.dim0, input, input_bytes, expert_ids,
      route_weights, command.scalar0, output, output_bytes, provider, stats);
}
