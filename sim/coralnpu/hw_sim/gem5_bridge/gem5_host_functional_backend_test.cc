#include "hw_sim/gem5_bridge/gem5_host_functional_backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

#include "hw_sim/gem5_bridge/npu_submission.h"

int main() {
  Gem5HostFunctionalBackend backend;
  assert(backend.Supports(OPENNPUX_NPU_OP_EMBED));
  assert(backend.Supports(OPENNPUX_NPU_OP_MATMUL));
  assert(backend.Supports(OPENNPUX_NPU_OP_NORMALIZE));
  assert(backend.Supports(OPENNPUX_NPU_OP_ATTENTION));
  assert(backend.Supports(OPENNPUX_NPU_OP_ROUTER));
  assert(backend.Supports(OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION));

  const float input[] = {3.0f, 4.0f};
  const float weight[] = {1.0f, 2.0f};
  float output[2] = {};
  Gem5HostFunctionalRequest request = {};
  request.opcode = OPENNPUX_NPU_OP_NORMALIZE;
  request.input = input;
  request.weight = weight;
  request.rows = 1;
  request.features = 2;
  request.output = output;
  Gem5HostFunctionalResult result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kComplete);
  assert(std::fabs(output[0] - 0.8485281f) < 1.0e-5f);
  assert(std::fabs(output[1] - 2.2627417f) < 1.0e-5f);
  assert(result.stats.operations != 0 && result.stats.modeled_cycles != 0);

  const uint32_t token_ids[] = {1};
  const float embedding_table[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float embedding_output[2] = {};
  request = {};
  request.opcode = OPENNPUX_NPU_OP_EMBED;
  request.input_indices = token_ids;
  request.weight = embedding_table;
  request.rows = 1;
  request.features = 2;
  request.vocabulary_size = 2;
  request.output = embedding_output;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kComplete);
  assert(embedding_output[0] == 3.0f && embedding_output[1] == 4.0f);

  const float attention_state[] = {
      1.0f, 0.0f, 0.0f, 1.0f,
      2.0f, 0.0f, 0.0f, 4.0f,
  };
  request = {};
  request.opcode = OPENNPUX_NPU_OP_ATTENTION;
  request.input = input;
  request.secondary = attention_state;
  request.rows = 1;
  request.features = 2;
  request.heads = 1;
  request.kv_heads = 1;
  request.head_dim = 2;
  request.kv_length = 2;
  request.output = output;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kComplete);
  assert(result.stats.operations != 0);

  opennpux_npu_operator_parameters convolution_parameters = {};
  convolution_parameters.intermediate_features = 2;
  const float convolution_input[] = {1.0f, 10.0f, 2.0f, 20.0f,
                                     3.0f, 30.0f};
  const float convolution_weight[] = {2.0f, 3.0f, 4.0f, 5.0f};
  float convolution_output[6] = {};
  request = {};
  request.opcode = OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION;
  request.input = convolution_input;
  request.weight = convolution_weight;
  request.rows = 3;
  request.features = 2;
  request.operator_parameters = &convolution_parameters;
  request.output = convolution_output;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kComplete);
  assert(convolution_output[0] == 4.0f &&
         convolution_output[1] == 50.0f &&
         convolution_output[2] == 10.0f &&
         convolution_output[3] == 130.0f &&
         convolution_output[4] == 16.0f &&
         convolution_output[5] == 210.0f);

  const float router_logits[] = {0.5f, 2.0f, 2.0f, -1.0f};
  float router_weights[2] = {};
  uint32_t router_indices[2] = {};
  request = {};
  request.opcode = OPENNPUX_NPU_OP_ROUTER;
  request.input = router_logits;
  request.rows = 1;
  request.features = 4;
  request.top_k = 2;
  request.output = router_weights;
  request.output_indices = router_indices;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kComplete);
  assert(router_indices[0] == 1 && router_indices[1] == 2);
  assert(router_weights[0] == 2.0f && router_weights[1] == 2.0f);
  assert(result.stats.operations != 0);

  request.opcode = OPENNPUX_NPU_OP_ADD;
  request.input = input;
  request.output = output;
  request.secondary = nullptr;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kExecutionError);

  opennpux_npu_operator_parameters parameters = {};
  parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters.struct_size = sizeof(parameters);
  parameters.opcode = OPENNPUX_NPU_OP_MATMUL;
  parameters.flags = OPENNPUX_NPU_PARAMETER_GPTQ;
  parameters.input_features = 2;
  parameters.output_features = 1;
  parameters.quantization_bits = 4;
  parameters.quantization_group_size = 2;
  parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
  const float matmul_input[] = {2.0f, 3.0f};
  const uint32_t qweight[] = {UINT32_C(0x21)};
  const uint32_t qzeros[] = {0};
  const float scales[] = {0.5f};
  float matmul_output[] = {0.0f};
  const Gem5GenericGptqOperands operands = {
      {matmul_input, sizeof(matmul_input)}, {qweight, sizeof(qweight)},
      {qzeros, sizeof(qzeros)},             {scales, sizeof(scales)},
      {nullptr, 0},                         {matmul_output, sizeof(matmul_output)},
  };
  request = {};
  request.opcode = OPENNPUX_NPU_OP_MATMUL;
  request.rows = 1;
  request.operator_parameters = &parameters;
  request.gptq_operands = &operands;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kComplete);
  assert(matmul_output[0] == 4.0f);
  assert(result.stats.operations == 4);

  float q_output[] = {0.0f};
  float k_output[] = {0.0f};
  float v_output[] = {0.0f};
  const Gem5GenericGptqOperands q_operands = {
      {matmul_input, sizeof(matmul_input)}, {qweight, sizeof(qweight)},
      {qzeros, sizeof(qzeros)},             {scales, sizeof(scales)},
      {nullptr, 0},                         {q_output, sizeof(q_output)},
  };
  const Gem5GenericGptqOperands k_operands = {
      {matmul_input, sizeof(matmul_input)}, {qweight, sizeof(qweight)},
      {qzeros, sizeof(qzeros)},             {scales, sizeof(scales)},
      {nullptr, 0},                         {k_output, sizeof(k_output)},
  };
  const Gem5GenericGptqOperands v_operands = {
      {matmul_input, sizeof(matmul_input)}, {qweight, sizeof(qweight)},
      {qzeros, sizeof(qzeros)},             {scales, sizeof(scales)},
      {nullptr, 0},                         {v_output, sizeof(v_output)},
  };
  parameters.output_features = 3;
  parameters.head_count = 1;
  parameters.kv_head_count = 1;
  parameters.head_dim = 1;
  request.output_secondary = k_output;
  request.output_tertiary = v_output;
  request.heads = 1;
  request.kv_heads = 1;
  request.head_dim = 1;
  request.q_gptq_operands = &q_operands;
  request.k_gptq_operands = &k_operands;
  request.v_gptq_operands = &v_operands;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kComplete);
  assert(q_output[0] == 4.0f && k_output[0] == 4.0f &&
         v_output[0] == 4.0f);
  assert(result.stats.operations == 12);

  parameters = {};
  parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters.struct_size = sizeof(parameters);
  parameters.opcode = OPENNPUX_NPU_OP_MATMUL;
  parameters.flags = OPENNPUX_NPU_PARAMETER_GPTQ;
  parameters.input_features = 2;
  parameters.output_features = 2;
  const float dense_weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float dense_output[2] = {};
  request = {};
  request.opcode = OPENNPUX_NPU_OP_MATMUL;
  request.input = matmul_input;
  request.weight = dense_weight;
  request.rows = 1;
  request.output = dense_output;
  request.operator_parameters = &parameters;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kComplete);
  assert(dense_output[0] == 11.0f && dense_output[1] == 16.0f);
  std::printf("mixed_precision_matmul_operations=%llu\n",
              static_cast<unsigned long long>(result.stats.operations));
  assert(result.stats.operations == 8);
  std::puts("gem5_host_functional_mixed_precision_matmul=PASS");

  assert(std::string(Gem5HostFunctionalStatusName(
             Gem5HostFunctionalStatus::kUnsupported)) == "unsupported");
  std::puts("gem5_host_functional_backend=PASS");
  return 0;
}
