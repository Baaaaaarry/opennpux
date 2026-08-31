#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HOST_ROUTED_EXPERT_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HOST_ROUTED_EXPERT_H_

#include <cstddef>
#include <cstdint>

#include "opennpux/xopennpux_graph.h"

class Gem5HostWeightProvider;

struct Gem5HostRoutedExpertStats {
  uint64_t operations;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t modeled_cycles;
  uint64_t routes_issued;
  uint64_t routes_completed;
};

bool RunGem5HostRoutedExpert(
    const void* operator_parameters, uint32_t rows, const float* input,
    size_t input_bytes, const uint32_t* expert_ids,
    const float* route_weights, uint32_t active_experts, float* output,
    size_t output_bytes, Gem5HostWeightProvider* provider,
    Gem5HostRoutedExpertStats* stats);

// Functional NPU-controller entry point for TROUTED_EXPERT. The command uses
// EXTMEM-relative offsets and a logical weight-plan command ID; no host pointer
// is encoded in the architectural record.
bool RunGem5XGraphRoutedExpert(
    const opennpux_xgraph_command& command, const void* operator_parameters,
    uint32_t extmem_base, uint8_t* extmem, size_t extmem_size,
    Gem5HostWeightProvider* provider, Gem5HostRoutedExpertStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HOST_ROUTED_EXPERT_H_
