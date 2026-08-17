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

enum Gem5GptqComponent : uint32_t {
  kGem5GptqQweight = 0,
  kGem5GptqQzeros = 1,
  kGem5GptqScales = 2,
  kGem5GptqGIdx = 3,
};

using Gem5GptqRead = bool (*)(void* opaque, Gem5GptqComponent component,
                              uint64_t offset, void* destination,
                              size_t size);

uint32_t Gem5GptqScaleElementSize(uint32_t data_type);

// AutoGPTQ layout: qweight packs the K axis, qzeros packs the N axis, and
// scales is group-major. g_idx is optional; otherwise k / group_size is used.
//
// The float32 accumulation order is part of the contract: the host reference in
// runtime/host/src/npu_gptq_reference.c reproduces it to gate the full-system
// projection test on an exact output checksum. Do not build either side with
// multiply-add contraction enabled.
bool RunGem5GptqInt4MatMul(
    const Gem5GptqMatMulConfig& config, const float* input,
    const uint32_t* qweight, const uint32_t* qzeros, const void* scales,
    const uint32_t* g_idx, float* output, Gem5GptqKernelStats* stats);

// Executes the same numerical kernel while fetching bounded output-channel
// tiles through a caller-owned reader. The reader may be backed by the shared
// page cache, so matrices larger than Coral EXTMEM never need to be contiguous.
bool RunGem5GptqInt4MatMulStreamed(
    const Gem5GptqMatMulConfig& config, const float* input,
    Gem5GptqRead reader, void* reader_opaque, uint32_t output_tile_columns,
    bool has_g_idx, float* output, Gem5GptqKernelStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_GPTQ_KERNELS_H_
