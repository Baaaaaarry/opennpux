#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HYBRID_KERNELS_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HYBRID_KERNELS_H_

#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_operator.h"

bool RunGem5HybridConv2D(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size);

bool RunGem5HybridDepthwiseConv2D(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HYBRID_KERNELS_H_
