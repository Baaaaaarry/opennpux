#ifndef HW_SIM_GEM5_BRIDGE_XOPENNPUX_INTRINSICS_H_
#define HW_SIM_GEM5_BRIDGE_XOPENNPUX_INTRINSICS_H_

#include <stdint.h>

static inline void xopennpux_write_mma_shape(uint32_t value) {
  __asm__ volatile("csrw 0x800, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_mma_data_type(uint32_t value) {
  __asm__ volatile("csrw 0x801, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_tmma_fp32(void* destination, const void* lhs,
                                       const void* rhs) {
  // rd is architecturally a source address even though GNU .insn describes
  // the R-type field as a destination. +r prevents the compiler from dropping
  // the incoming destination pointer.
  register uintptr_t destination_register = (uintptr_t)destination;
  __asm__ volatile(".insn r 0x7b, 0, 0, %0, %1, %2"
                   : "+r"(destination_register)
                   : "r"((uintptr_t)lhs), "r"((uintptr_t)rhs)
                   : "memory");
}

static inline void xopennpux_tfence(void) {
  __asm__ volatile(".insn r 0x7b, 6, 0, x0, x0, x0" : : : "memory");
}

#endif  // HW_SIM_GEM5_BRIDGE_XOPENNPUX_INTRINSICS_H_
