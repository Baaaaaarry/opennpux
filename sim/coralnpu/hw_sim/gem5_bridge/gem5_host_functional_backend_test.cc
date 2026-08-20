#include "hw_sim/gem5_bridge/gem5_host_functional_backend.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

#include "hw_sim/gem5_bridge/npu_submission.h"

int main() {
  Gem5HostFunctionalBackend backend;
  assert(backend.Supports(OPENNPUX_NPU_OP_NORMALIZE));
  assert(!backend.Supports(OPENNPUX_NPU_OP_ATTENTION));

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

  request.opcode = OPENNPUX_NPU_OP_ATTENTION;
  result = backend.Execute(request);
  assert(result.status == Gem5HostFunctionalStatus::kUnsupported);
  assert(result.stats.operations == 0);

  request.opcode = OPENNPUX_NPU_OP_ADD;
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

  assert(std::string(Gem5HostFunctionalStatusName(
             Gem5HostFunctionalStatus::kUnsupported)) == "unsupported");
  std::puts("gem5_host_functional_backend=PASS");
  return 0;
}
