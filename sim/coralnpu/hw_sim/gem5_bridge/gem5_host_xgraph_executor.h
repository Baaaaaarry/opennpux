#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HOST_XGRAPH_EXECUTOR_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HOST_XGRAPH_EXECUTOR_H_

#include <cstdint>

#include "hw_sim/gem5_bridge/gem5_host_tensor_arena.h"
#include "opennpux/npu_functional_request.h"
#include "opennpux/npu_submission.h"

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

// Lowers one generic request and executes its primitive XOpenNPUX instruction
// through the same L2 decoder and functional coprocessor used by Coral RTL.
Gem5HostXGraphExecutionOutcome ExecuteGem5HostXGraphPrimitive(
    const opennpux_npu_functional_request& request,
    const opennpux_npu_operator_parameters& parameters,
    Gem5HostTensorArena* arena, Gem5HostXGraphExecutionStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HOST_XGRAPH_EXECUTOR_H_
