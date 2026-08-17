#ifndef HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_GPTQ_EXECUTOR_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_GPTQ_EXECUTOR_H_

#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/gem5_gptq_kernels.h"
#include "hw_sim/gem5_bridge/npu_submission.h"
#include "hw_sim/gem5_bridge/npu_weight_queue.h"

struct Gem5GenericConstBuffer {
  const void* data;
  size_t size;
};

struct Gem5GenericMutableBuffer {
  void* data;
  size_t size;
};

struct Gem5GenericGptqOperands {
  Gem5GenericConstBuffer input;
  Gem5GenericConstBuffer qweight;
  Gem5GenericConstBuffer qzeros;
  Gem5GenericConstBuffer scales;
  Gem5GenericConstBuffer g_idx;
  Gem5GenericMutableBuffer output;
};

struct Gem5GenericGptqWeights {
  Gem5GenericConstBuffer qweight;
  Gem5GenericConstBuffer qzeros;
  Gem5GenericConstBuffer scales;
  Gem5GenericConstBuffer g_idx;
};

struct Gem5GenericGptqExpertOperands {
  Gem5GenericConstBuffer input;
  Gem5GenericGptqWeights gate;
  Gem5GenericGptqWeights up;
  Gem5GenericGptqWeights down;
  Gem5GenericMutableBuffer gate_output;
  Gem5GenericMutableBuffer up_output;
  Gem5GenericMutableBuffer activated;
  Gem5GenericMutableBuffer output;
};

struct Gem5GenericGptqPageSpan {
  Gem5GptqComponent component;
  uint64_t tensor_offset;
  const void* data;
  size_t size;
};

struct Gem5GenericGptqPageReader {
  const Gem5GenericGptqPageSpan* spans;
  size_t span_count;
};

// Gem5GptqRead adapter over semantic ranges currently resident in cache slots.
// Every requested byte must be covered exactly once; gaps and overlaps fail.
bool ReadGem5GenericGptqPageSpans(
    void* opaque, Gem5GptqComponent component, uint64_t offset,
    void* destination, size_t size);

bool BuildGem5GenericGptqPageSpan(
    const opennpux_npu_page_fault& fault, const void* cache,
    size_t cache_size, Gem5GenericGptqPageSpan* span);

bool RunGem5GenericGptqMatMul(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    const Gem5GenericGptqOperands& operands, Gem5GptqKernelStats* stats);

bool RunGem5GenericGptqMatMulStreamed(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    Gem5GenericConstBuffer input, Gem5GptqRead reader, void* reader_opaque,
    uint32_t output_tile_columns, bool has_g_idx,
    Gem5GenericMutableBuffer output, Gem5GptqKernelStats* stats);

// Executes a generic SwiGLU-style GPTQ expert: down(SiLU(gate(x)) * up(x)).
// The operator parameters come directly from an executable EXPERT command.
bool RunGem5GenericGptqExpert(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    const Gem5GenericGptqExpertOperands& operands,
    Gem5GptqKernelStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_GPTQ_EXECUTOR_H_
