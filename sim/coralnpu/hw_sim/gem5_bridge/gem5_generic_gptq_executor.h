#ifndef HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_GPTQ_EXECUTOR_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_GPTQ_EXECUTOR_H_

#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/gem5_gptq_kernels.h"
#include "hw_sim/gem5_bridge/npu_submission.h"

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

bool RunGem5GenericGptqMatMul(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    const Gem5GenericGptqOperands& operands, Gem5GptqKernelStats* stats);

// Executes a generic SwiGLU-style GPTQ expert: down(SiLU(gate(x)) * up(x)).
// The operator parameters come directly from an executable EXPERT command.
bool RunGem5GenericGptqExpert(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    const Gem5GenericGptqExpertOperands& operands,
    Gem5GptqKernelStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_GENERIC_GPTQ_EXECUTOR_H_
