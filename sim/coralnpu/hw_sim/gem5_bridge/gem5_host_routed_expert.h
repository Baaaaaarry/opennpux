#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HOST_ROUTED_EXPERT_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HOST_ROUTED_EXPERT_H_

#include <cstddef>
#include <cstdint>

class Gem5HostWeightProvider;

struct Gem5HostRoutedExpertStats {
  uint64_t operations;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t modeled_cycles;
};

bool RunGem5HostRoutedExpert(
    const void* operator_parameters, uint32_t rows, const float* input,
    size_t input_bytes, const uint32_t* expert_ids,
    const float* route_weights, uint32_t active_experts, float* output,
    size_t output_bytes, Gem5HostWeightProvider* provider,
    Gem5HostRoutedExpertStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HOST_ROUTED_EXPERT_H_
