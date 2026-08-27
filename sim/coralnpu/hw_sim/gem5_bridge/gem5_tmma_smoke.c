#include <stdint.h>

#include "hw_sim/gem5_bridge/xopennpux_intrinsics.h"

enum {
  kExtmemBase = 0x20000000,
  kLhsOffset = 0x000,
  kRhsOffset = 0x100,
  kDstOffset = 0x200,
  kResultOffset = 0x300,
};

int main(void) {
  volatile uint32_t* lhs =
      (volatile uint32_t*)(uintptr_t)(kExtmemBase + kLhsOffset);
  volatile uint32_t* rhs =
      (volatile uint32_t*)(uintptr_t)(kExtmemBase + kRhsOffset);
  volatile uint32_t* dst =
      (volatile uint32_t*)(uintptr_t)(kExtmemBase + kDstOffset);
  volatile uint32_t* result =
      (volatile uint32_t*)(uintptr_t)(kExtmemBase + kResultOffset);

  lhs[0] = 0x3f800000;  // 1.0
  lhs[1] = 0x40000000;  // 2.0
  lhs[2] = 0x40400000;  // 3.0
  lhs[3] = 0x40800000;  // 4.0
  rhs[0] = 0x40a00000;  // 5.0
  rhs[1] = 0x40c00000;  // 6.0
  rhs[2] = 0x40e00000;  // 7.0
  rhs[3] = 0x41000000;  // 8.0

  xopennpux_write_mma_shape((2u << 20) | (2u << 10) | 2u);
  xopennpux_write_mma_data_type((2u << 8) | (2u << 4) | 2u);
  xopennpux_tmma_fp32((void*)dst, (const void*)lhs, (const void*)rhs);
  xopennpux_tfence();

  result[0] = dst[0] == 0x41980000 && dst[1] == 0x41b00000 &&
                      dst[2] == 0x422c0000 && dst[3] == 0x42480000
                  ? 0x544d4d41
                  : 0;
  return result[0] == 0x544d4d41 ? 0 : 1;
}
