#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"

#include <cassert>
#include <cmath>
#include <cstring>

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
  parameters.quantized_zero_bias = 0;
  return parameters;
}

Gem5GenericGptqWeights IdentityWeights(const uint32_t* qweight,
                                       const uint32_t* qzeros,
                                       const float* scales) {
  return {{qweight, sizeof(uint32_t) * 2}, {qzeros, sizeof(uint32_t)},
          {scales, sizeof(float) * 2}, {nullptr, 0}};
}

struct StreamedWeights {
  const void* components[4];
  size_t sizes[4];
};

bool ReadStreamedWeight(void* opaque, Gem5GptqComponent component,
                        uint64_t offset, void* destination, size_t size) {
  auto* weights = static_cast<StreamedWeights*>(opaque);
  const size_t index = static_cast<size_t>(component);
  if (index >= 4 || offset > weights->sizes[index] ||
      size > weights->sizes[index] - offset) {
    return false;
  }
  std::memcpy(destination,
              static_cast<const uint8_t*>(weights->components[index]) + offset,
              size);
  return true;
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

  parameters.output_features = 8;
  const uint32_t streamed_qweight[8] = {
      UINT32_C(0x21), UINT32_C(0x21), UINT32_C(0x21), UINT32_C(0x21),
      UINT32_C(0x21), UINT32_C(0x21), UINT32_C(0x21), UINT32_C(0x21)};
  const uint32_t streamed_qzeros[1] = {};
  const float streamed_scales[8] = {
      0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
  float streamed_output[8] = {};
  StreamedWeights streamed_weights = {
      {streamed_qweight, streamed_qzeros, streamed_scales, nullptr},
      {sizeof(streamed_qweight), sizeof(streamed_qzeros),
       sizeof(streamed_scales), 0}};
  assert(RunGem5GenericGptqMatMulStreamed(
      parameters, 1, {input, sizeof(input)}, ReadStreamedWeight,
      &streamed_weights, 8, false,
      {streamed_output, sizeof(streamed_output)}, &stats));
  for (float value : streamed_output) {
    assert(value == 4.0f);
  }
  parameters.output_features = 1;

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

  auto expert_parameters = Parameters();
  expert_parameters.opcode = 13;
  expert_parameters.input_features = 2;
  expert_parameters.output_features = 2;
  expert_parameters.intermediate_features = 2;
  const float expert_input[] = {1.0f, 2.0f};
  const uint32_t identity_qweight[] = {
      UINT32_C(0x00000001), UINT32_C(0x00000010)};
  const uint32_t sum_qweight[] = {
      UINT32_C(0x00000011), UINT32_C(0x00000011)};
  const uint32_t expert_qzeros[] = {0};
  const float expert_scales[] = {1.0f, 1.0f};
  float gate_output[2] = {};
  float up_output[2] = {};
  float activated[2] = {};
  float expert_output[2] = {};
  const Gem5GenericGptqExpertOperands expert_operands = {
      {expert_input, sizeof(expert_input)},
      IdentityWeights(identity_qweight, expert_qzeros, expert_scales),
      IdentityWeights(sum_qweight, expert_qzeros, expert_scales),
      IdentityWeights(identity_qweight, expert_qzeros, expert_scales),
      {gate_output, sizeof(gate_output)},
      {up_output, sizeof(up_output)},
      {activated, sizeof(activated)},
      {expert_output, sizeof(expert_output)},
  };
  assert(RunGem5GenericGptqExpert(
      expert_parameters, 1, expert_operands, &stats));
  assert(gate_output[0] == 1.0f && gate_output[1] == 2.0f);
  assert(up_output[0] == 3.0f && up_output[1] == 3.0f);
  assert(std::fabs(expert_output[0] -
                   3.0f / (1.0f + std::exp(-1.0f))) < 1e-6f);
  assert(std::fabs(expert_output[1] -
                   6.0f / (1.0f + std::exp(-2.0f))) < 1e-6f);
  assert(stats.operations == 36);
  assert(stats.modeled_cycles > 0);
  expert_parameters.intermediate_features = 0;
  assert(!RunGem5GenericGptqExpert(
      expert_parameters, 1, expert_operands, &stats));
  return 0;
}
