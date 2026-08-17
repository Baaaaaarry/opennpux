#ifndef HW_SIM_GEM5_BRIDGE_GEM5_GPTQ_KERNELS_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_GPTQ_KERNELS_H_

#include <cstddef>
#include <cstdint>

enum Gem5GptqScaleDataType : uint32_t {
  kGem5GptqScaleFloat16 = 4,
  kGem5GptqScaleBfloat16 = 5,
  kGem5GptqScaleFloat32 = 6,
};

struct Gem5GptqMatMulConfig {
  uint32_t rows;
  uint32_t input_columns;
  uint32_t output_columns;
  uint32_t group_size;
  uint32_t zero_bias;
  uint32_t scale_data_type;
};

struct Gem5GptqKernelStats {
  uint64_t operations;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t modeled_cycles;
};

uint32_t Gem5GptqScaleElementSize(uint32_t data_type);

// AutoGPTQ layout: qweight packs the K axis, qzeros packs the N axis, and
// scales is group-major. g_idx is optional; otherwise k / group_size is used.
bool RunGem5GptqInt4MatMul(
    const Gem5GptqMatMulConfig& config, const float* input,
    const uint32_t* qweight, const uint32_t* qzeros, const void* scales,
    const uint32_t* g_idx, float* output, Gem5GptqKernelStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_GPTQ_KERNELS_H_
