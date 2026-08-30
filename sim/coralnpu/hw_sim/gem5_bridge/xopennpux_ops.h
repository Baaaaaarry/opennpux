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

static inline void xopennpux_causal_depthwise_conv_fp32(
    void* destination, const void* input, const void* weight, uint32_t rows,
    uint32_t features, uint32_t kernel_width, const void* previous_state,
    void* next_state, uint32_t flags) {
  xopennpux_configure_tensor_fp32(rows, features);
  xopennpux_write_scalar_param0((kernel_width & 0xffffu) |
                                ((flags & 0xffffu) << 16));
  xopennpux_write_tensor_aux_source_address(
      (uint32_t)(uintptr_t)previous_state);
  xopennpux_write_tensor_aux_destination_address(
      (uint32_t)(uintptr_t)next_state);
  xopennpux_tcausalconv_fp32(destination, input, weight);
  xopennpux_tfence();
}

static inline void xopennpux_attention_fp32(
    void* destination, const void* query, const void* kv_state,
    uint32_t query_rows, uint32_t heads, uint32_t kv_heads,
    uint32_t head_dim, uint32_t kv_length, const void* gate,
    uint32_t flags) {
  xopennpux_configure_tensor_fp32(query_rows, heads * head_dim);
  xopennpux_write_attention_heads((heads & 0xffffu) |
                                  ((kv_heads & 0xffffu) << 16));
  xopennpux_write_attention_head_dim_flags(
      (head_dim & 0xffffu) | ((flags & 0xffffu) << 16));
  xopennpux_write_attention_kv_length(kv_length);
  xopennpux_write_tensor_aux_source_address((uint32_t)(uintptr_t)gate);
  xopennpux_tattention_fp32(destination, query, kv_state);
  xopennpux_tfence();
}

static inline void xopennpux_recurrent_fp32(
    void* destination, const void* qkv, const void* alpha, const void* beta,
    void* state, const void* a_log, const void* dt_bias, uint32_t rows,
    uint32_t key_heads, uint32_t value_heads, uint32_t key_dim,
    uint32_t value_dim) {
  xopennpux_configure_tensor_fp32(rows, key_heads * key_dim);
  xopennpux_write_recurrent_heads((key_heads & 0xffffu) |
                                  ((value_heads & 0xffffu) << 16));
  xopennpux_write_recurrent_dims((key_dim & 0xffffu) |
                                 ((value_dim & 0xffffu) << 16));
  xopennpux_write_recurrent_beta_address((uint32_t)(uintptr_t)beta);
  xopennpux_write_tensor_aux_destination_address((uint32_t)(uintptr_t)state);
  xopennpux_write_recurrent_a_log_address((uint32_t)(uintptr_t)a_log);
  xopennpux_write_recurrent_dt_bias_address((uint32_t)(uintptr_t)dt_bias);
  xopennpux_trecurrent_fp32(destination, qkv, alpha);
  xopennpux_tfence();
}

static inline void xopennpux_conv2d_nhwc_ohwi_fp32(
    void* destination, const void* input, const void* weights,
    const void* bias, uint32_t batches, uint32_t input_height,
    uint32_t input_width, uint32_t input_channels, uint32_t output_height,
    uint32_t output_width, uint32_t output_channels, uint32_t kernel_height,
    uint32_t kernel_width, uint32_t stride_height, uint32_t stride_width,
    uint32_t padding_top, uint32_t padding_left, uint32_t padding_bottom,
    uint32_t padding_right, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups) {
  xopennpux_configure_tensor_fp32(batches, input_channels);
  xopennpux_write_conv_input_hw((input_height & 0xffffu) |
                                ((input_width & 0xffffu) << 16));
  xopennpux_write_conv_output_hw((output_height & 0xffffu) |
                                 ((output_width & 0xffffu) << 16));
  xopennpux_write_conv_channels_groups((output_channels & 0xffffu) |
                                       ((groups & 0xffffu) << 16));
  xopennpux_write_conv_kernel_hw((kernel_height & 0xffffu) |
                                 ((kernel_width & 0xffffu) << 16));
  xopennpux_write_conv_stride_hw((stride_height & 0xffffu) |
                                 ((stride_width & 0xffffu) << 16));
  xopennpux_write_conv_padding_tl((padding_top & 0xffffu) |
                                  ((padding_left & 0xffffu) << 16));
  xopennpux_write_conv_padding_br((padding_bottom & 0xffffu) |
                                  ((padding_right & 0xffffu) << 16));
  xopennpux_write_conv_dilation_hw((dilation_height & 0xffffu) |
                                   ((dilation_width & 0xffffu) << 16));
  xopennpux_write_conv_bias_address((uint32_t)(uintptr_t)bias);
  xopennpux_tconv_fp32(destination, input, weights);
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

// A null indices destination selects the legacy packed values/indices layout.
// Equal values are ordered by ascending source index.
static inline void xopennpux_topk_fp32(void* values_destination,
                                       void* indices_destination,
                                       const void* input, uint32_t rows,
                                       uint32_t features, uint32_t k) {
  xopennpux_configure_tensor_fp32(rows, features);
  xopennpux_write_scalar_param0(k);
  xopennpux_write_tensor_aux_destination_address(
      (uint32_t)(uintptr_t)indices_destination);
  xopennpux_ttopk_fp32(values_destination, input);
  xopennpux_tfence();
}

#endif  // HW_SIM_GEM5_BRIDGE_XOPENNPUX_OPS_H_
