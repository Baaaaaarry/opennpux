#include <stddef.h>
#include <stdint.h>

#include "hw_sim/gem5_bridge/xopennpux_ops.h"

enum {
  kExtmemBase = 0x20000000,
  kCase0LhsOffset = 0x000,
  kCase0RhsOffset = 0x100,
  kCase0DstOffset = 0x200,
  kCase1LhsOffset = 0x400,
  kCase1RhsOffset = 0x500,
  kCase1DstOffset = 0x600,
  kCase2LhsOffset = 0x800,
  kCase2RhsOffset = 0x900,
  kCase2DstOffset = 0xa00,
  kResultOffset = 0xc00,
  kAddLhsOffset = 0xd00,
  kAddRhsOffset = 0xe00,
  kAddDstOffset = 0xf00,
  kMulLhsOffset = 0x1000,
  kMulRhsOffset = 0x1100,
  kMulDstOffset = 0x1200,
  kRmsInputOffset = 0x1300,
  kRmsWeightOffset = 0x1400,
  kRmsDstOffset = 0x1500,
  kSiluInputOffset = 0x1600,
  kSiluDstOffset = 0x1700,
  kSoftmaxInputOffset = 0x1800,
  kSoftmaxDstOffset = 0x1900,
  kGatherTableOffset = 0x1a00,
  kGatherIndicesOffset = 0x1b00,
  kGatherDstOffset = 0x1c00,
  kRopeInputOffset = 0x1d00,
  kRopeTableOffset = 0x1e00,
  kRopeDstOffset = 0x1f00,
  kTopKInputOffset = 0x2000,
  kTopKDstOffset = 0x2100,
  kDequantQweightOffset = 0x2200,
  kDequantQzerosOffset = 0x2300,
  kDequantScalesOffset = 0x2400,
  kDequantScratchOffset = 0x2500,
  kDequantInputOffset = 0x2600,
  kDequantOutputOffset = 0x2700,
  kResultMagic = 0x544d4545,
};

static volatile uint32_t* Extmem(uint32_t offset) {
  return (volatile uint32_t*)(uintptr_t)(kExtmemBase + offset);
}

static void WriteWords(volatile uint32_t* destination, const uint32_t* source,
                       size_t count) {
  for (size_t index = 0; index < count; ++index) {
    destination[index] = source[index];
  }
}

static int WordsEqual(const volatile uint32_t* actual,
                      const uint32_t* expected, size_t count) {
  for (size_t index = 0; index < count; ++index) {
    if (actual[index] != expected[index]) {
      return 0;
    }
  }
  return 1;
}

static void RunFp32Tmma(uint32_t m, uint32_t n, uint32_t k,
                        volatile uint32_t* destination,
                        const volatile uint32_t* lhs,
                        const volatile uint32_t* rhs) {
  xopennpux_matmul_fp32((void*)destination, (const void*)lhs,
                        (const void*)rhs, m, n, k);
}

int main(void) {
  static const uint32_t kCase0Lhs[] = {
      0x3f800000, 0x40000000, 0x40400000,
      0x40800000, 0x40a00000, 0x40c00000,
  };
  static const uint32_t kCase0Rhs[] = {
      0x40e00000, 0x41000000, 0x41100000,
      0x41200000, 0x41300000, 0x41400000,
  };
  static const uint32_t kCase0Expected[] = {
      0x42680000, 0x42800000, 0x430b0000, 0x431a0000,
  };

  static const uint32_t kCase1Lhs[] = {
      0x3f800000, 0xc0000000, 0x40400000,
      0x40800000, 0xbf800000, 0x40a00000,
  };
  static const uint32_t kCase1Rhs[] = {
      0x40000000, 0x00000000, 0xbf800000, 0x40400000,
      0x40800000, 0x3f800000, 0x40000000, 0xc0000000,
  };
  static const uint32_t kCase1Expected[] = {
      0xc0c00000, 0xc0000000, 0xc0a00000, 0x40e00000,
      0x41b00000, 0x40800000, 0x40a00000, 0x3f800000,
      0x41900000, 0x40a00000, 0x41300000, 0xc1500000,
  };

  static const uint32_t kCase2Lhs[] = {
      0x40000000, 0xbf800000, 0x3f000000, 0x40800000,
  };
  static const uint32_t kCase2Rhs[] = {
      0x3f800000, 0x40000000, 0x40400000,
      0x40800000, 0x40a00000, 0x40c00000,
      0x40000000, 0xc0000000, 0x3f800000,
      0x3f000000, 0x3f800000, 0xbf800000,
  };
  static const uint32_t kCase2Expected[] = {
      0x3f800000, 0x40000000, 0xc0600000,
  };
  static const uint32_t kAddLhs[] = {
      0x3f800000, 0xc0000000, 0x40400000, 0x40800000,
      0x40a00000, 0xc0c00000, 0x40e00000, 0x41000000,
  };
  static const uint32_t kAddRhs[] = {
      0x40000000, 0x3f800000, 0xc0400000, 0x40800000,
      0xbf800000, 0x40c00000, 0x3f000000, 0xc0000000,
  };
  static const uint32_t kAddExpected[] = {
      0x40400000, 0xbf800000, 0x00000000, 0x41000000,
      0x40800000, 0x00000000, 0x40f00000, 0x40c00000,
  };
  static const uint32_t kMulExpected[] = {
      0x40000000, 0xc0000000, 0xc1100000, 0x41800000,
      0xc0a00000, 0xc2100000, 0x40600000, 0xc1800000,
  };
  static const uint32_t kRmsInput[] = {
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
  };
  static const uint32_t kRmsWeight[] = {
      0x3f800000, 0x40000000, 0x40400000, 0x40800000,
  };
  static const uint32_t kRmsExpected[] = {
      0x3f000000, 0x3f800000, 0x3fc00000, 0x40000000,
      0xbf000000, 0xbf800000, 0xbfc00000, 0xc0000000,
  };
  static const uint32_t kSiluInput[] = {
      0xc0000000, 0xbf800000, 0x00000000, 0x3f800000,
      0x40000000, 0x40800000, 0x41000000, 0x41800000,
  };
  static const uint32_t kSiluExpected[] = {
      0xbe7420a9, 0xbe89b2b1, 0x00000000, 0x3f3b26a8,
      0x3fe17bea, 0x407b6541, 0x40ffea06, 0x417ffffe,
  };
  static const uint32_t kSoftmaxInput[] = {
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0xc0000000, 0xc0000000, 0xc0000000, 0xc0000000,
  };
  static const uint32_t kSoftmaxExpected[] = {
      0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000,
      0x3e800000, 0x3e800000, 0x3e800000, 0x3e800000,
  };
  static const uint32_t kGatherTable[] = {
      0x3f800000, 0x40000000, 0x40400000,
      0x40800000, 0x40a00000, 0x40c00000,
      0x40e00000, 0x41000000, 0x41100000,
      0x41200000, 0x41300000, 0x41400000,
  };
  static const uint32_t kGatherIndices[] = {2, 0};
  static const uint32_t kGatherExpected[] = {
      0x40e00000, 0x41000000, 0x41100000,
      0x3f800000, 0x40000000, 0x40400000,
  };
  static const uint32_t kRopeInput[] = {
      0x3f800000, 0x40000000, 0x40400000, 0x40800000,
      0x40a00000, 0x40c00000, 0x40e00000, 0x41000000,
  };
  // The table is contiguous [cos rows][sin rows]. Row 0 is identity and row
  // 1 applies a 90-degree half-split rotation.
  static const uint32_t kRopeTable[] = {
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000,
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
  };
  static const uint32_t kRopeExpected[] = {
      0x3f800000, 0x40000000, 0x40400000, 0x40800000,
      0xc0e00000, 0xc1000000, 0x40a00000, 0x40c00000,
  };
  static const uint32_t kTopKInput[] = {
      0x3f800000, 0x40a00000, 0x40400000, 0x40a00000, 0x40000000,
      0xbf800000, 0x00000000, 0x40800000, 0x40000000, 0x40400000,
  };
  static const uint32_t kTopKExpected[] = {
      0x40a00000, 0x40a00000, 0x40800000, 0x40400000,
      1, 3, 2, 4,
  };
  static const uint32_t kDequantQweight[] = {
      0x99999999, 0xaaaaaaaa,
  };
  static const uint32_t kDequantQzeros[] = {
      0x00000077, 0x00000077,
  };
  static const uint32_t kDequantScales[] = {
      0x3f000000, 0x3f800000, 0x3f000000, 0x3f800000,
  };
  static const uint32_t kDequantInput[] = {
      0x3f800000, 0x40000000, 0x40400000, 0x40800000,
      0x40a00000, 0x40c00000, 0x40e00000, 0x41000000,
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
      0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000,
  };
  static const uint32_t kDequantExpected[] = {
      0x41900000, 0x42900000, 0x40800000, 0x41800000,
  };

  volatile uint32_t* case0_lhs = Extmem(kCase0LhsOffset);
  volatile uint32_t* case0_rhs = Extmem(kCase0RhsOffset);
  volatile uint32_t* case0_dst = Extmem(kCase0DstOffset);
  volatile uint32_t* case1_lhs = Extmem(kCase1LhsOffset);
  volatile uint32_t* case1_rhs = Extmem(kCase1RhsOffset);
  volatile uint32_t* case1_dst = Extmem(kCase1DstOffset);
  volatile uint32_t* case2_lhs = Extmem(kCase2LhsOffset);
  volatile uint32_t* case2_rhs = Extmem(kCase2RhsOffset);
  volatile uint32_t* case2_dst = Extmem(kCase2DstOffset);
  volatile uint32_t* result = Extmem(kResultOffset);
  volatile uint32_t* add_lhs = Extmem(kAddLhsOffset);
  volatile uint32_t* add_rhs = Extmem(kAddRhsOffset);
  volatile uint32_t* add_dst = Extmem(kAddDstOffset);
  volatile uint32_t* mul_lhs = Extmem(kMulLhsOffset);
  volatile uint32_t* mul_rhs = Extmem(kMulRhsOffset);
  volatile uint32_t* mul_dst = Extmem(kMulDstOffset);
  volatile uint32_t* rms_input = Extmem(kRmsInputOffset);
  volatile uint32_t* rms_weight = Extmem(kRmsWeightOffset);
  volatile uint32_t* rms_dst = Extmem(kRmsDstOffset);
  volatile uint32_t* silu_input = Extmem(kSiluInputOffset);
  volatile uint32_t* silu_dst = Extmem(kSiluDstOffset);
  volatile uint32_t* softmax_input = Extmem(kSoftmaxInputOffset);
  volatile uint32_t* softmax_dst = Extmem(kSoftmaxDstOffset);
  volatile uint32_t* gather_table = Extmem(kGatherTableOffset);
  volatile uint32_t* gather_indices = Extmem(kGatherIndicesOffset);
  volatile uint32_t* gather_dst = Extmem(kGatherDstOffset);
  volatile uint32_t* rope_input = Extmem(kRopeInputOffset);
  volatile uint32_t* rope_table = Extmem(kRopeTableOffset);
  volatile uint32_t* rope_dst = Extmem(kRopeDstOffset);
  volatile uint32_t* topk_input = Extmem(kTopKInputOffset);
  volatile uint32_t* topk_dst = Extmem(kTopKDstOffset);
  volatile uint32_t* dequant_qweight = Extmem(kDequantQweightOffset);
  volatile uint32_t* dequant_qzeros = Extmem(kDequantQzerosOffset);
  volatile uint32_t* dequant_scales = Extmem(kDequantScalesOffset);
  volatile uint32_t* dequant_scratch = Extmem(kDequantScratchOffset);
  volatile uint32_t* dequant_input = Extmem(kDequantInputOffset);
  volatile uint32_t* dequant_output = Extmem(kDequantOutputOffset);

  WriteWords(case0_lhs, kCase0Lhs, 6);
  WriteWords(case0_rhs, kCase0Rhs, 6);
  RunFp32Tmma(2, 2, 3, case0_dst, case0_lhs, case0_rhs);

  WriteWords(case1_lhs, kCase1Lhs, 6);
  WriteWords(case1_rhs, kCase1Rhs, 8);
  RunFp32Tmma(3, 4, 2, case1_dst, case1_lhs, case1_rhs);

  WriteWords(case2_lhs, kCase2Lhs, 4);
  WriteWords(case2_rhs, kCase2Rhs, 12);
  RunFp32Tmma(1, 3, 4, case2_dst, case2_lhs, case2_rhs);

  WriteWords(add_lhs, kAddLhs, 8);
  WriteWords(add_rhs, kAddRhs, 8);
  xopennpux_add_fp32((void*)add_dst, (const void*)add_lhs,
                     (const void*)add_rhs, 2, 2, 2);

  WriteWords(mul_lhs, kAddLhs, 8);
  WriteWords(mul_rhs, kAddRhs, 8);
  xopennpux_mul_fp32((void*)mul_dst, (const void*)mul_lhs,
                     (const void*)mul_rhs, 2, 2, 2);

  WriteWords(rms_input, kRmsInput, 8);
  WriteWords(rms_weight, kRmsWeight, 4);
  xopennpux_rmsnorm_fp32((void*)rms_dst, (const void*)rms_input,
                         (const void*)rms_weight, 2, 4, 3.0f);

  WriteWords(silu_input, kSiluInput, 8);
  xopennpux_silu_fp32((void*)silu_dst, (const void*)silu_input, 2, 4);

  WriteWords(softmax_input, kSoftmaxInput, 8);
  xopennpux_softmax_fp32((void*)softmax_dst, (const void*)softmax_input, 2, 4);

  WriteWords(gather_table, kGatherTable, 12);
  WriteWords(gather_indices, kGatherIndices, 2);
  xopennpux_gather_fp32((void*)gather_dst, (const void*)gather_table,
                        (const uint32_t*)gather_indices, 2, 3, 4);

  WriteWords(rope_input, kRopeInput, 8);
  WriteWords(rope_table, kRopeTable, 16);
  xopennpux_rope_fp32((void*)rope_dst, (const void*)rope_input,
                       (const void*)rope_table, 2, 4,
                       XOPENNPUX_ROPE_HALF_SPLIT);

  WriteWords(topk_input, kTopKInput, 10);
  xopennpux_topk_fp32((void*)topk_dst, (const void*)topk_input, 2, 5, 2);

  WriteWords(dequant_qweight, kDequantQweight, 2);
  WriteWords(dequant_qzeros, kDequantQzeros, 2);
  WriteWords(dequant_scales, kDequantScales, 4);
  WriteWords(dequant_input, kDequantInput, 16);
  xopennpux_dequant_int4_fp32(
      (void*)dequant_scratch, (const void*)dequant_qweight,
      (const void*)dequant_qzeros, (const void*)dequant_scales, NULL, 2, 8,
      4, 1, 2, 8, 4, 8, 0, 2);
  RunFp32Tmma(2, 2, 8, dequant_output, dequant_input, dequant_scratch);

  const uint32_t failure_mask =
      (WordsEqual(case0_dst, kCase0Expected, 4) ? 0u : 1u) |
      (WordsEqual(case1_dst, kCase1Expected, 12) ? 0u : 2u) |
      (WordsEqual(case2_dst, kCase2Expected, 3) ? 0u : 4u) |
      (WordsEqual(add_dst, kAddExpected, 8) ? 0u : 8u) |
      (WordsEqual(mul_dst, kMulExpected, 8) ? 0u : 16u) |
      (WordsEqual(rms_dst, kRmsExpected, 8) ? 0u : 32u) |
      (WordsEqual(silu_dst, kSiluExpected, 8) ? 0u : 64u) |
      (WordsEqual(softmax_dst, kSoftmaxExpected, 8) ? 0u : 128u) |
      (WordsEqual(gather_dst, kGatherExpected, 6) ? 0u : 256u) |
      (WordsEqual(rope_dst, kRopeExpected, 8) ? 0u : 512u) |
      (WordsEqual(topk_dst, kTopKExpected, 8) ? 0u : 1024u) |
      (WordsEqual(dequant_output, kDequantExpected, 4) ? 0u : 2048u);
  result[0] = failure_mask == 0 ? kResultMagic : 0;
  result[1] = failure_mask;
  result[2] = 12;

  if (failure_mask != 0) {
    __asm__ volatile("ebreak");
  }
  return 0;
}
