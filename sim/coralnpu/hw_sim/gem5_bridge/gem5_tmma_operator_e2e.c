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

  const uint32_t failure_mask =
      (WordsEqual(case0_dst, kCase0Expected, 4) ? 0u : 1u) |
      (WordsEqual(case1_dst, kCase1Expected, 12) ? 0u : 2u) |
      (WordsEqual(case2_dst, kCase2Expected, 3) ? 0u : 4u) |
      (WordsEqual(add_dst, kAddExpected, 8) ? 0u : 8u) |
      (WordsEqual(mul_dst, kMulExpected, 8) ? 0u : 16u) |
      (WordsEqual(rms_dst, kRmsExpected, 8) ? 0u : 32u);
  result[0] = failure_mask == 0 ? kResultMagic : 0;
  result[1] = failure_mask;
  result[2] = 6;

  if (failure_mask != 0) {
    __asm__ volatile("ebreak");
  }
  return 0;
}
