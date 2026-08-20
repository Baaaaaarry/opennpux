#ifndef HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_COMMAND_DISPATCH_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_COMMAND_DISPATCH_H_

#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/npu_functional_request.h"

struct Gem5FunctionalMemoryRegion {
  uint32_t base;
  uint8_t* data;
  size_t size;
};

// Executes an already-materialized command without requiring a Coral custom
// instruction envelope. Graph-level schedulers use this entry point.
bool ExecuteGem5FunctionalRequest(
    opennpux_npu_functional_request* request, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size);
bool ExecuteGem5FunctionalRequestInRegions(
    opennpux_npu_functional_request* request,
    const Gem5FunctionalMemoryRegion* regions, size_t region_count);

bool DispatchGem5GenericCommand(coral_operator_descriptor* descriptor,
                                uint8_t* extmem, uint32_t extmem_base,
                                size_t extmem_size);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_COMMAND_DISPATCH_H_
