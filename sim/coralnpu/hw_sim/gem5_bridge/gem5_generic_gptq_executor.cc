#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"

#include <limits>

namespace {

constexpr uint32_t kMatMulOpcode = 2;
constexpr uint32_t kGptqFlag = 1;

bool ProductSize(uint64_t first, uint64_t second, uint64_t third,
                 uint64_t* result) {
  if (result == nullptr ||
      (second != 0 && first > std::numeric_limits<uint64_t>::max() / second)) {
    return false;
  }
  const uint64_t pair = first * second;
  if (third != 0 && pair > std::numeric_limits<uint64_t>::max() / third) {
    return false;
  }
  *result = pair * third;
  return true;
}

bool BufferFits(const Gem5GenericConstBuffer& buffer, uint64_t required) {
  return buffer.data != nullptr && required <= buffer.size;
}

}  // namespace

bool RunGem5GenericGptqMatMul(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    const Gem5GenericGptqOperands& operands, Gem5GptqKernelStats* stats) {
  if (parameters.magic != OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC ||
      parameters.version != OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION ||
      parameters.struct_size != sizeof(parameters) ||
      parameters.opcode != kMatMulOpcode ||
      (parameters.flags & kGptqFlag) == 0 ||
      parameters.quantization_bits != 4 || rows == 0 ||
      parameters.input_features == 0 || parameters.output_features == 0 ||
      parameters.quantization_group_size == 0 || stats == nullptr ||
      operands.output.data == nullptr) {
    return false;
  }

  const uint64_t input_columns = parameters.input_features;
  const uint64_t output_columns = parameters.output_features;
  const uint64_t groups =
      (input_columns + parameters.quantization_group_size - 1) /
      parameters.quantization_group_size;
  const uint64_t weight_rows = (input_columns + 7) / 8;
  const uint64_t zero_columns = (output_columns + 7) / 8;
  uint64_t input_bytes = 0;
  uint64_t qweight_bytes = 0;
  uint64_t qzeros_bytes = 0;
  uint64_t scales_bytes = 0;
  uint64_t g_idx_bytes = 0;
  uint64_t output_bytes = 0;
  if (!ProductSize(rows, input_columns, sizeof(float), &input_bytes) ||
      !ProductSize(weight_rows, output_columns, sizeof(uint32_t),
                   &qweight_bytes) ||
      !ProductSize(groups, zero_columns, sizeof(uint32_t), &qzeros_bytes) ||
      !ProductSize(groups, output_columns, sizeof(float), &scales_bytes) ||
      !ProductSize(input_columns, 1, sizeof(uint32_t), &g_idx_bytes) ||
      !ProductSize(rows, output_columns, sizeof(float), &output_bytes) ||
      !BufferFits(operands.input, input_bytes) ||
      !BufferFits(operands.qweight, qweight_bytes) ||
      !BufferFits(operands.qzeros, qzeros_bytes) ||
      !BufferFits(operands.scales, scales_bytes) ||
      (operands.g_idx.data != nullptr &&
       !BufferFits(operands.g_idx, g_idx_bytes)) ||
      output_bytes > operands.output.size) {
    return false;
  }

  const Gem5GptqMatMulConfig config = {
      rows, parameters.input_features, parameters.output_features,
      parameters.quantization_group_size, 0};
  return RunGem5GptqInt4MatMul(
      config, static_cast<const float*>(operands.input.data),
      static_cast<const uint32_t*>(operands.qweight.data),
      static_cast<const uint32_t*>(operands.qzeros.data),
      static_cast<const float*>(operands.scales.data),
      static_cast<const uint32_t*>(operands.g_idx.data),
      static_cast<float*>(operands.output.data), stats);
}
