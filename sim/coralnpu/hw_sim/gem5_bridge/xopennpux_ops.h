#ifndef HW_SIM_GEM5_BRIDGE_XOPENNPUX_OPS_H_
#define HW_SIM_GEM5_BRIDGE_XOPENNPUX_OPS_H_

#include <stdint.h>

#include "hw_sim/gem5_bridge/xopennpux_intrinsics.h"

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
  // Experimental v0.1 maps the tensor element count to M*N*K until the
  // tensor_dim CSR ports are implemented. This is an ABI compatibility layer,
  // not an architectural dependence of TADD on MMA.
  xopennpux_write_mma_shape((dim2 << 20) | (dim1 << 10) | dim0);
  xopennpux_write_mma_data_type((2u << 8) | (2u << 4) | 2u);
  xopennpux_tadd_fp32(destination, lhs, rhs);
  xopennpux_tfence();
}

static inline void xopennpux_mul_fp32(void* destination, const void* lhs,
                                      const void* rhs, uint32_t dim0,
                                      uint32_t dim1, uint32_t dim2) {
  xopennpux_write_mma_shape((dim2 << 20) | (dim1 << 10) | dim0);
  xopennpux_write_mma_data_type((2u << 8) | (2u << 4) | 2u);
  xopennpux_tmul_fp32(destination, lhs, rhs);
  xopennpux_tfence();
}

#endif  // HW_SIM_GEM5_BRIDGE_XOPENNPUX_OPS_H_
