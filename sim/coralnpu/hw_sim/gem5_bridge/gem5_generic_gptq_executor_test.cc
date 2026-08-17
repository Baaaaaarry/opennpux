#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"

#include <cassert>

namespace {

opennpux_npu_operator_parameters Parameters() {
  opennpux_npu_operator_parameters parameters = {};
  parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters.struct_size = sizeof(parameters);
  parameters.opcode = 2;
  parameters.flags = 1;
  parameters.input_features = 2;
  parameters.output_features = 1;
  parameters.quantization_bits = 4;
  parameters.quantization_group_size = 2;
  parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
  return parameters;
}

}  // namespace

int main() {
  const float input[] = {2.0f, 3.0f};
  const uint32_t qweight[] = {UINT32_C(0x21)};
  const uint32_t qzeros[] = {0};
  const float scales[] = {0.5f};
  float output[] = {0.0f};
  const Gem5GenericGptqOperands operands = {
      {input, sizeof(input)},       {qweight, sizeof(qweight)},
      {qzeros, sizeof(qzeros)},    {scales, sizeof(scales)},
      {nullptr, 0},                {output, sizeof(output)},
  };
  Gem5GptqKernelStats stats = {};
  auto parameters = Parameters();
  assert(RunGem5GenericGptqMatMul(parameters, 1, operands, &stats));
  assert(output[0] == 4.0f);
  assert(stats.operations == 4);

  const uint16_t half_scales[] = {UINT16_C(0x3800)};
  auto half_operands = operands;
  half_operands.scales = {half_scales, sizeof(half_scales)};
  parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT16;
  output[0] = 0.0f;
  assert(RunGem5GenericGptqMatMul(
      parameters, 1, half_operands, &stats));
  assert(output[0] == 4.0f);

  auto truncated = operands;
  truncated.qweight.size = 0;
  parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
  assert(!RunGem5GenericGptqMatMul(parameters, 1, truncated, &stats));
  parameters.quantization_bits = 8;
  assert(!RunGem5GenericGptqMatMul(parameters, 1, operands, &stats));
  return 0;
}
