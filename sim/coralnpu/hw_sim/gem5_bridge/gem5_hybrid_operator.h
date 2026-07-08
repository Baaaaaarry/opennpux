#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HYBRID_OPERATOR_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HYBRID_OPERATOR_H_

#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_operator.h"

struct Gem5HybridOperatorResult {
  bool has_mobilenet_output;
  int32_t mobilenet_output[5];
};

bool ValidateGem5HybridDescriptor(
    const coral_operator_descriptor& descriptor, uint32_t extmem_base,
    size_t extmem_size, uint32_t* error);

bool DispatchGem5HybridOperator(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size,
    Gem5HybridOperatorResult* result);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HYBRID_OPERATOR_H_
