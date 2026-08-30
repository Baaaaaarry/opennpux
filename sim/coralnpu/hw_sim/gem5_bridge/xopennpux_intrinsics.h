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

static inline void xopennpux_write_quant_qzeros_address(uint32_t value) {
  __asm__ volatile("csrw 0x810, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_quant_scales_address(uint32_t value) {
  __asm__ volatile("csrw 0x811, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_quant_g_idx_address(uint32_t value) {
  __asm__ volatile("csrw 0x812, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_quant_config(uint32_t value) {
  __asm__ volatile("csrw 0x813, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_quant_qweight_stride(uint32_t value) {
  __asm__ volatile("csrw 0x814, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_quant_qzeros_stride(uint32_t value) {
  __asm__ volatile("csrw 0x815, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_quant_scales_stride(uint32_t value) {
  __asm__ volatile("csrw 0x816, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_quant_group_range(uint32_t value) {
  __asm__ volatile("csrw 0x817, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_tensor_aux_source_address(uint32_t value) {
  __asm__ volatile("csrw 0x818, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_tensor_aux_destination_address(
    uint32_t value) {
  __asm__ volatile("csrw 0x819, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_attention_heads(uint32_t value) {
  __asm__ volatile("csrw 0x81a, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_attention_head_dim_flags(uint32_t value) {
  __asm__ volatile("csrw 0x81b, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_attention_kv_length(uint32_t value) {
  __asm__ volatile("csrw 0x81c, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_recurrent_heads(uint32_t value) {
  __asm__ volatile("csrw 0x81d, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_recurrent_dims(uint32_t value) {
  __asm__ volatile("csrw 0x81e, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_recurrent_beta_address(uint32_t value) {
  __asm__ volatile("csrw 0x81f, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_recurrent_a_log_address(uint32_t value) {
  __asm__ volatile("csrw 0x820, %0" : : "r"(value) : "memory");
}

static inline void xopennpux_write_recurrent_dt_bias_address(uint32_t value) {
  __asm__ volatile("csrw 0x821, %0" : : "r"(value) : "memory");
}

#define XOPENNPUX_CONV_CSR_WRITER(name, csr)                         \
  static inline void xopennpux_write_conv_##name(uint32_t value) {  \
    __asm__ volatile("csrw " #csr ", %0" : : "r"(value) : "memory"); \
  }
XOPENNPUX_CONV_CSR_WRITER(input_hw, 0x822)
XOPENNPUX_CONV_CSR_WRITER(output_hw, 0x823)
XOPENNPUX_CONV_CSR_WRITER(channels_groups, 0x824)
XOPENNPUX_CONV_CSR_WRITER(kernel_hw, 0x825)
XOPENNPUX_CONV_CSR_WRITER(stride_hw, 0x826)
XOPENNPUX_CONV_CSR_WRITER(padding_tl, 0x827)
XOPENNPUX_CONV_CSR_WRITER(padding_br, 0x828)
XOPENNPUX_CONV_CSR_WRITER(dilation_hw, 0x829)
XOPENNPUX_CONV_CSR_WRITER(bias_address, 0x82a)
#undef XOPENNPUX_CONV_CSR_WRITER

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

static inline void xopennpux_tsoftmax_fp32(void* destination,
                                           const void* input) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 2, 0x32, %0, %1, x0"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)input)
                   : "memory");
}

static inline void xopennpux_trope_fp32(void* destination, const void* input,
                                        const void* cos_sin_table) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 2, 0x33, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)input),
                     "r"((uintptr_t)cos_sin_table)
                   : "memory");
}

static inline void xopennpux_tsilu_fp32(void* destination,
                                        const void* input) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 2, 0x46, %0, %1, x0"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)input)
                   : "memory");
}

static inline void xopennpux_tgather_fp32(void* destination,
                                          const void* table,
                                          const uint32_t* indices) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 3, 0x10, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)table),
                     "r"((uintptr_t)indices)
                   : "memory");
}

static inline void xopennpux_tdequant_int4_fp32(void* destination,
                                                const void* qweight) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 3, 0x11, %0, %1, x0"
                   :
                   : "r"((uintptr_t)destination),
                     "r"((uintptr_t)qweight)
                   : "memory");
}

static inline void xopennpux_tdma_fp32(void* destination,
                                       const void* source) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 3, 0x12, %0, %1, x0"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)source)
                   : "memory");
}

static inline void xopennpux_tcausalconv_fp32(void* destination,
                                              const void* input,
                                              const void* weight) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 0, 0x20, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)input),
                     "r"((uintptr_t)weight)
                   : "memory");
}

static inline void xopennpux_tattention_fp32(void* destination,
                                             const void* query,
                                             const void* kv_state) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 0, 0x21, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)query),
                     "r"((uintptr_t)kv_state)
                   : "memory");
}

static inline void xopennpux_trecurrent_fp32(void* destination,
                                             const void* qkv,
                                             const void* alpha) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 0, 0x22, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)qkv),
                     "r"((uintptr_t)alpha)
                   : "memory");
}

static inline void xopennpux_ttopk_fp32(void* destination,
                                        const void* input) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 4, 0, %0, %1, x0"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)input)
                   : "memory");
}

static inline void xopennpux_tconv_fp32(void* destination,
                                         const void* input,
                                         const void* weights) {
  __asm__ volatile("fence rw, rw" : : : "memory");
  __asm__ volatile(".insn r 0x7b, 0, 0x23, %0, %1, %2"
                   :
                   : "r"((uintptr_t)destination), "r"((uintptr_t)input),
                     "r"((uintptr_t)weights)
                   : "memory");
}

static inline void xopennpux_tfence(void) {
  __asm__ volatile(".insn r 0x7b, 6, 0, x0, x0, x0" : : : "memory");
  // Prevent scalar loads from observing memory before NPU writeback.
  __asm__ volatile("fence rw, rw" : : : "memory");
}

#endif  // HW_SIM_GEM5_BRIDGE_XOPENNPUX_INTRINSICS_H_
