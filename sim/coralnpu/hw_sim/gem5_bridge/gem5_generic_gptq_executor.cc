#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"

#include <cmath>
#include <limits>

namespace {

constexpr uint32_t kMatMulOpcode = 2;
constexpr uint32_t kExpertOpcode = 13;
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

uint32_t ScaleElementSize(uint32_t data_type) {
  if (data_type == OPENNPUX_NPU_DTYPE_FLOAT16 ||
      data_type == OPENNPUX_NPU_DTYPE_BFLOAT16) {
    return sizeof(uint16_t);
  }
  return data_type == OPENNPUX_NPU_DTYPE_FLOAT32 ? sizeof(float) : 0;
}

bool AddStats(const Gem5GptqKernelStats& source,
              Gem5GptqKernelStats* destination) {
  if (destination == nullptr ||
      source.operations > UINT64_MAX - destination->operations ||
      source.bytes_read > UINT64_MAX - destination->bytes_read ||
      source.bytes_written > UINT64_MAX - destination->bytes_written ||
      source.modeled_cycles > UINT64_MAX - destination->modeled_cycles) {
    return false;
  }
  destination->operations += source.operations;
  destination->bytes_read += source.bytes_read;
  destination->bytes_written += source.bytes_written;
  destination->modeled_cycles += source.modeled_cycles;
  return true;
}

bool RunProjection(const opennpux_npu_operator_parameters& base,
                   uint32_t rows, uint32_t input_features,
                   uint32_t output_features,
                   const Gem5GenericConstBuffer& input,
                   const Gem5GenericGptqWeights& weights,
                   const Gem5GenericMutableBuffer& output,
                   Gem5GptqKernelStats* stats) {
  auto parameters = base;
  parameters.opcode = kMatMulOpcode;
  parameters.input_features = input_features;
  parameters.output_features = output_features;
  return RunGem5GenericGptqMatMul(
      parameters, rows,
      {input, weights.qweight, weights.qzeros, weights.scales, weights.g_idx,
       output},
      stats);
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
      parameters.quantization_group_size == 0 ||
      ScaleElementSize(parameters.scale_data_type) == 0 || stats == nullptr ||
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
      !ProductSize(groups, output_columns,
                   ScaleElementSize(parameters.scale_data_type),
                   &scales_bytes) ||
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
      parameters.quantization_group_size, parameters.quantized_zero_bias,
      parameters.scale_data_type};
  return RunGem5GptqInt4MatMul(
      config, static_cast<const float*>(operands.input.data),
      static_cast<const uint32_t*>(operands.qweight.data),
      static_cast<const uint32_t*>(operands.qzeros.data),
      operands.scales.data,
      static_cast<const uint32_t*>(operands.g_idx.data),
      static_cast<float*>(operands.output.data), stats);
}

bool RunGem5GenericGptqMatMulStreamed(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    Gem5GenericConstBuffer input, Gem5GptqRead reader, void* reader_opaque,
    uint32_t output_tile_columns, bool has_g_idx,
    Gem5GenericMutableBuffer output, Gem5GptqKernelStats* stats) {
  if (parameters.magic != OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC ||
      parameters.version != OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION ||
      parameters.struct_size != sizeof(parameters) ||
      parameters.opcode != kMatMulOpcode ||
      (parameters.flags & kGptqFlag) == 0 ||
      parameters.quantization_bits != 4 || rows == 0 ||
      parameters.input_features == 0 || parameters.output_features == 0 ||
      parameters.quantization_group_size == 0 ||
      ScaleElementSize(parameters.scale_data_type) == 0 || reader == nullptr ||
      stats == nullptr || output.data == nullptr) {
    return false;
  }
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  if (!ProductSize(rows, parameters.input_features, sizeof(float),
                   &input_bytes) ||
      !ProductSize(rows, parameters.output_features, sizeof(float),
                   &output_bytes) ||
      !BufferFits(input, input_bytes) || output_bytes > output.size) {
    return false;
  }
  const Gem5GptqMatMulConfig config = {
      rows, parameters.input_features, parameters.output_features,
      parameters.quantization_group_size, parameters.quantized_zero_bias,
      parameters.scale_data_type};
  return RunGem5GptqInt4MatMulStreamed(
      config, static_cast<const float*>(input.data), reader, reader_opaque,
      output_tile_columns, has_g_idx, static_cast<float*>(output.data), stats);
}

bool RunGem5GenericGptqExpert(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    const Gem5GenericGptqExpertOperands& operands,
    Gem5GptqKernelStats* stats) {
  if (parameters.magic != OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC ||
      parameters.version != OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION ||
      parameters.struct_size != sizeof(parameters) ||
      parameters.opcode != kExpertOpcode ||
      (parameters.flags & kGptqFlag) == 0 || rows == 0 ||
      parameters.input_features == 0 ||
      parameters.output_features != parameters.input_features ||
      parameters.intermediate_features == 0 || stats == nullptr ||
      operands.gate_output.data == nullptr ||
      operands.up_output.data == nullptr || operands.activated.data == nullptr ||
      operands.output.data == nullptr) {
    return false;
  }

  uint64_t intermediate_count = 0;
  uint64_t intermediate_bytes = 0;
  if (!ProductSize(rows, parameters.intermediate_features, 1,
                   &intermediate_count) ||
      !ProductSize(intermediate_count, sizeof(float), 1,
                   &intermediate_bytes) ||
      intermediate_bytes > operands.gate_output.size ||
      intermediate_bytes > operands.up_output.size ||
      intermediate_bytes > operands.activated.size) {
    return false;
  }

  Gem5GptqKernelStats gate_stats = {};
  Gem5GptqKernelStats up_stats = {};
  Gem5GptqKernelStats down_stats = {};
  if (!RunProjection(parameters, rows, parameters.input_features,
                     parameters.intermediate_features, operands.input,
                     operands.gate, operands.gate_output, &gate_stats) ||
      !RunProjection(parameters, rows, parameters.input_features,
                     parameters.intermediate_features, operands.input,
                     operands.up, operands.up_output, &up_stats)) {
    return false;
  }

  const auto* gate = static_cast<const float*>(operands.gate_output.data);
  const auto* up = static_cast<const float*>(operands.up_output.data);
  auto* activated = static_cast<float*>(operands.activated.data);
  for (uint64_t index = 0; index < intermediate_count; ++index) {
    const float sigmoid = 1.0f / (1.0f + std::exp(-gate[index]));
    activated[index] = gate[index] * sigmoid * up[index];
    if (!std::isfinite(activated[index])) {
      return false;
    }
  }
  if (!RunProjection(parameters, rows, parameters.intermediate_features,
                     parameters.output_features,
                     {activated, static_cast<size_t>(intermediate_bytes)},
                     operands.down, operands.output, &down_stats) ||
      intermediate_count > UINT64_MAX / 6 ||
      intermediate_bytes > UINT64_MAX / 3) {
    return false;
  }

  const uint64_t activation_operations = intermediate_count * 6;
  const uint64_t activation_bytes_read = intermediate_bytes * 2;
  const uint64_t activation_bytes_written = intermediate_bytes;
  const Gem5GptqKernelStats activation_stats = {
      activation_operations, activation_bytes_read,
      activation_bytes_written,
      (activation_operations + 15) / 16 +
          (activation_bytes_read + activation_bytes_written + 15) / 16};
  *stats = {};
  return AddStats(gate_stats, stats) && AddStats(up_stats, stats) &&
         AddStats(activation_stats, stats) && AddStats(down_stats, stats);
}
