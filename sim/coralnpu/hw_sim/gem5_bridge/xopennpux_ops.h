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

static inline void xopennpux_softmax_fp32(void* destination,
                                          const void* input, uint32_t rows,
                                          uint32_t features) {
  xopennpux_configure_tensor_fp32(rows, features);
  xopennpux_tsoftmax_fp32(destination, input);
  xopennpux_tfence();
}

enum xopennpux_rope_layout {
  XOPENNPUX_ROPE_ADJACENT = 0,
  XOPENNPUX_ROPE_HALF_SPLIT = 1,
};

static inline void xopennpux_rope_fp32(void* destination, const void* input,
                                       const void* cos_sin_table,
                                       uint32_t rows, uint32_t features,
                                       enum xopennpux_rope_layout layout) {
  xopennpux_configure_tensor_fp32(rows, features);
  xopennpux_write_scalar_param0((uint32_t)layout);
  xopennpux_trope_fp32(destination, input, cos_sin_table);
  xopennpux_tfence();
}

static inline void xopennpux_silu_fp32(void* destination, const void* input,
                                       uint32_t rows, uint32_t features) {
  xopennpux_configure_tensor_fp32(rows, features);
  xopennpux_tsilu_fp32(destination, input);
  xopennpux_tfence();
}

static inline void xopennpux_gather_fp32(void* destination,
                                         const void* table,
                                         const uint32_t* indices,
                                         uint32_t index_count,
                                         uint32_t features,
                                         uint32_t source_rows) {
  xopennpux_configure_tensor_fp32(index_count, features);
  xopennpux_write_scalar_param0(source_rows);
  xopennpux_tgather_fp32(destination, table, indices);
  xopennpux_tfence();
}

static inline void xopennpux_dma_fp32(void* destination, const void* source,
                                      uint32_t rows, uint32_t features) {
  xopennpux_configure_tensor_fp32(rows, features);
  xopennpux_tdma_fp32(destination, source);
  xopennpux_tfence();
}

// Dequantizes one AutoGPTQ output-channel tile to contiguous FP32 [K, N].
// qweight/qzeros/scales may remain strided views into the full matrix.
static inline void xopennpux_dequant_int4_fp32(
    void* destination, const void* qweight, const void* qzeros,
    const void* scales, const uint32_t* g_idx, uint32_t n, uint32_t k,
    uint32_t group_size, uint32_t zero_bias, uint32_t scale_data_type,
    uint32_t qweight_stride, uint32_t qzeros_stride,
    uint32_t scales_stride, uint32_t group_base, uint32_t group_count) {
  xopennpux_write_mma_shape((k << 20) | (n << 10) | 1u);
  xopennpux_write_mma_data_type((2u << 8) | (2u << 4) | 7u);
  xopennpux_write_quant_qzeros_address((uint32_t)(uintptr_t)qzeros);
  xopennpux_write_quant_scales_address((uint32_t)(uintptr_t)scales);
  xopennpux_write_quant_g_idx_address((uint32_t)(uintptr_t)g_idx);
  xopennpux_write_quant_config(
      (group_size & 0xffffu) | ((zero_bias & 0xfu) << 16) |
      ((scale_data_type & 0xfu) << 20) |
      (g_idx != 0 ? (1u << 24) : 0u));
  xopennpux_write_quant_qweight_stride(qweight_stride);
  xopennpux_write_quant_qzeros_stride(qzeros_stride);
  xopennpux_write_quant_scales_stride(scales_stride);
  xopennpux_write_quant_group_range((group_count << 16) | group_base);
  xopennpux_tdequant_int4_fp32(destination, qweight);
  xopennpux_tfence();
}

// The packed result contains rows*k FP32 values followed by rows*k uint32
// indices. Equal values are ordered by ascending source index.
static inline void xopennpux_topk_fp32(void* packed_destination,
                                       const void* input, uint32_t rows,
                                       uint32_t features, uint32_t k) {
  xopennpux_configure_tensor_fp32(rows, features);
  xopennpux_write_scalar_param0(k);
  xopennpux_ttopk_fp32(packed_destination, input);
  xopennpux_tfence();
}

#endif  // HW_SIM_GEM5_BRIDGE_XOPENNPUX_OPS_H_
