#ifndef HW_SIM_GEM5_BRIDGE_XOPENNPUX_OPS_H_
#define HW_SIM_GEM5_BRIDGE_XOPENNPUX_OPS_H_

#include <stdint.h>

#include "hw_sim/gem5_bridge/xopennpux_intrinsics.h"

static inline uint32_t xopennpux_fp32_bits(float value) {
  union {
    float fp32;
    uint32_t bits;
  } conversion = {value};
  return conversion.bits;
}

static inline void xopennpux_configure_tensor_fp32(uint32_t rows,
                                                   uint32_t features) {
  xopennpux_write_tensor_shape((features << 16) | rows);
  xopennpux_write_tensor_data_type((2u << 8) | (2u << 4) | 2u);
}

// Firmware/compiler-facing functional operator library. These helpers own CSR
// materialization and synchronization; callers provide only tensor metadata
// and addresses. The instruction encoding remains independent of model names.
static inline void xopennpux_matmul_fp32(void* destination, const void* lhs,
                                         const void* rhs, uint32_t m,
                                         uint32_t n, uint32_t k) {
  xopennpux_write_mma_shape((k << 20) | (n << 10) | m);
  xopennpux_write_mma_data_type((2u << 8) | (2u << 4) | 2u);
  xopennpux_tmma_fp32(destination, lhs, rhs);
  xopennpux_tfence();
}

static inline void xopennpux_add_fp32(void* destination, const void* lhs,
                                      const void* rhs, uint32_t dim0,
                                      uint32_t dim1, uint32_t dim2) {
  xopennpux_configure_tensor_fp32(dim0, dim1 * dim2);
  xopennpux_tadd_fp32(destination, lhs, rhs);
  xopennpux_tfence();
}

static inline void xopennpux_mul_fp32(void* destination, const void* lhs,
                                      const void* rhs, uint32_t dim0,
                                      uint32_t dim1, uint32_t dim2) {
  xopennpux_configure_tensor_fp32(dim0, dim1 * dim2);
  xopennpux_tmul_fp32(destination, lhs, rhs);
  xopennpux_tfence();
}


static inline void xopennpux_rmsnorm_fp32(void* destination,
                                          const void* input,
                                          const void* weight, uint32_t rows,
                                          uint32_t features, float epsilon) {
  xopennpux_configure_tensor_fp32(rows, features);
  xopennpux_write_scalar_param0(xopennpux_fp32_bits(epsilon));
  xopennpux_trmsnorm_fp32(destination, input, weight);
  xopennpux_tfence();
}

#endif  // HW_SIM_GEM5_BRIDGE_XOPENNPUX_OPS_H_
