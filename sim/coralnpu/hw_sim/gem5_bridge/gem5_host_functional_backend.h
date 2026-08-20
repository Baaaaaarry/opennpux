#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HOST_FUNCTIONAL_BACKEND_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HOST_FUNCTIONAL_BACKEND_H_

#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"
#include "hw_sim/gem5_bridge/gem5_transformer_kernels.h"

enum class Gem5HostFunctionalStatus {
  kComplete,
  kInvalid,
  kUnsupported,
  kExecutionError,
};

struct Gem5HostFunctionalRequest {
  uint32_t opcode;
  const float* input;
  const uint32_t* input_indices;
  const float* secondary;
  const float* tertiary;
  const float* weight;
  const uint32_t* positions;
  size_t rows;
  size_t features;
  size_t heads;
  size_t head_dim;
  size_t kv_heads;
  size_t kv_length;
  size_t top_k;
  size_t vocabulary_size;
  float epsilon;
  float rope_theta;
  float* output;
  float* output_secondary;
  float* output_tertiary;
  uint32_t* output_indices;
  size_t output_indices_count;
  const opennpux_npu_operator_parameters* operator_parameters;
  const Gem5GenericGptqOperands* gptq_operands;
  const Gem5GenericGptqOperands* q_gptq_operands;
  const Gem5GenericGptqOperands* k_gptq_operands;
  const Gem5GenericGptqOperands* v_gptq_operands;
  const Gem5GenericGptqExpertOperands* gptq_expert_operands;
};

struct Gem5HostFunctionalResult {
  Gem5HostFunctionalStatus status;
  Gem5TransformerKernelStats stats;
};

// Model-independent functional execution backend. It performs real tensor
// computation on the simulation host while preserving the same command opcode
// boundary that timing-model and RTL engines will implement.
class Gem5HostFunctionalBackend {
 public:
  Gem5HostFunctionalResult Execute(
      const Gem5HostFunctionalRequest& request) const;
  bool Supports(uint32_t opcode) const;
};

const char* Gem5HostFunctionalStatusName(Gem5HostFunctionalStatus status);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HOST_FUNCTIONAL_BACKEND_H_
