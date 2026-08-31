#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HOST_XGRAPH_EXECUTOR_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HOST_XGRAPH_EXECUTOR_H_

#include <cstdint>

#include "opennpux/npu_xgraph_lowering.h"
#include "hw_sim/gem5_bridge/gem5_generic_command_dispatch.h"
#include "hw_sim/gem5_bridge/gem5_host_tensor_arena.h"

enum class Gem5HostXGraphExecutionOutcome {
  kNotEligible,
  kExecuted,
  kError,
};

struct Gem5HostXGraphExecutionStats {
  uint32_t commands = 0;
  uint64_t operations = 0;
  uint64_t modeled_cycles = 0;
  uint64_t bytes_read = 0;
  uint64_t bytes_written = 0;
};

// Lowers one generic request and executes its XOpenNPUX instruction sequence
// through the same L2 decoder and functional coprocessor used by Coral RTL.
Gem5HostXGraphExecutionOutcome ExecuteGem5HostXGraphRequest(
    const opennpux_npu_functional_request& request,
    const opennpux_npu_operator_parameters& parameters,
    const Gem5FunctionalMemoryRegion* regions, size_t region_count,
    Gem5HostTensorArena* arena, Gem5HostXGraphExecutionStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HOST_XGRAPH_EXECUTOR_H_
