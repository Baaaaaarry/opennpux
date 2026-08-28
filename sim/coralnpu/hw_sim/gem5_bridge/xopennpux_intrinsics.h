#ifndef HW_SIM_GEM5_BRIDGE_XOPENNPUX_INTRINSICS_H_
#define HW_SIM_GEM5_BRIDGE_XOPENNPUX_INTRINSICS_H_

#include <stdint.h>

static inline void xopennpux_write_mma_shape(uint32_t value) {
  __asm__ volatile("csrw 0x800, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_mma_data_type(uint32_t value) {
  __asm__ volatile("csrw 0x801, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_tensor_shape(uint32_t value) {
  __asm__ volatile("csrw 0x802, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_tensor_data_type(uint32_t value) {
  __asm__ volatile("csrw 0x806, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_scalar_param0(uint32_t value) {
  __asm__ volatile("csrw 0x80b, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_tmma_fp32(void* destination, const void* lhs,
                                       const void* rhs) {
  // Publish scalar/AXI stores before the coprocessor reads its operands.
  __asm__ volatile("fence rw, rw" : : : "memory");
  // The encoded rd field is an address source for TMMA. Declare all three
  // operands as inputs because the Coral scalar register file is not written.
  __asm__ volatile(".insn r 0x7b, 0, 0, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)lhs),
                     "r"((uintptr_t)rhs)
                   : "memory");
}

static inline void xopennpux_tadd_fp32(void* destination, const void* lhs,
                                       const void* rhs) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 1, 1, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)lhs),
                     "r"((uintptr_t)rhs)
                   : "memory");
}

static inline void xopennpux_tmul_fp32(void* destination, const void* lhs,
                                       const void* rhs) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 1, 2, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)lhs),
                     "r"((uintptr_t)rhs)
                   : "memory");
}

static inline void xopennpux_trmsnorm_fp32(void* destination,
                                           const void* input,
                                           const void* weight) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 2, 0x31, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)input),
                     "r"((uintptr_t)weight)
                   : "memory");
}

static inline void xopennpux_tfence(void) {
  __asm__ volatile(".insn r 0x7b, 6, 0, x0, x0, x0" : : : "memory");
  // Prevent scalar loads from observing memory before NPU writeback.
  __asm__ volatile("fence rw, rw" : : : "memory");
}

#endif  // HW_SIM_GEM5_BRIDGE_XOPENNPUX_INTRINSICS_H_
