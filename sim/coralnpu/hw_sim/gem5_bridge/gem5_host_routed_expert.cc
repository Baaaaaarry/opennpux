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
            kernel_stats.bytes_written, kernel_stats.modeled_cycles};
  return true;
}
