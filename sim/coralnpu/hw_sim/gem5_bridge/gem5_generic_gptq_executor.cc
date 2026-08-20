#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
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

bool ReadGem5GenericGptqPageSpans(
    void* opaque, Gem5GptqComponent component, uint64_t offset,
    void* destination, size_t size) {
  auto* reader = static_cast<Gem5GenericGptqPageReader*>(opaque);
  if (reader == nullptr || reader->spans == nullptr || destination == nullptr ||
      size == 0 || offset > UINT64_MAX - size) {
    return false;
  }
  auto* output = static_cast<uint8_t*>(destination);
  const uint64_t end = offset + size;
  uint64_t cursor = offset;
  while (cursor < end) {
    const Gem5GenericGptqPageSpan* match = nullptr;
    for (size_t index = 0; index < reader->span_count; ++index) {
      const auto& span = reader->spans[index];
      if (span.component != component || span.data == nullptr ||
          span.tensor_offset > UINT64_MAX - span.size ||
          cursor < span.tensor_offset ||
          cursor >= span.tensor_offset + span.size) {
        continue;
      }
      if (match != nullptr) {
        return false;
      }
      match = &span;
    }
    if (match == nullptr) {
      return false;
    }
    const uint64_t span_end = match->tensor_offset + match->size;
    const uint64_t copy_end = span_end < end ? span_end : end;
    const size_t copy_size = static_cast<size_t>(copy_end - cursor);
    std::memcpy(output + static_cast<size_t>(cursor - offset),
                static_cast<const uint8_t*>(match->data) +
                    static_cast<size_t>(cursor - match->tensor_offset),
                copy_size);
    cursor = copy_end;
  }
  return true;
}

bool BuildGem5GenericGptqPageSpan(
    const opennpux_npu_page_fault& fault, const void* cache,
    size_t cache_size, Gem5GenericGptqPageSpan* span) {
  if (cache == nullptr || span == nullptr || fault.page_size == 0 ||
      fault.range_size == 0 || fault.file_offset > UINT64_MAX - fault.page_size ||
      fault.range_file_offset > UINT64_MAX - fault.range_size ||
      fault.cache_slot > SIZE_MAX / fault.page_size) {
    return false;
  }
  Gem5GptqComponent component;
  switch (fault.component_id) {
    case OPENNPUX_NPU_WEIGHT_COMPONENT_QWEIGHT:
      component = kGem5GptqQweight;
      break;
    case OPENNPUX_NPU_WEIGHT_COMPONENT_QZEROS:
      component = kGem5GptqQzeros;
      break;
    case OPENNPUX_NPU_WEIGHT_COMPONENT_SCALES:
      component = kGem5GptqScales;
      break;
    case OPENNPUX_NPU_WEIGHT_COMPONENT_G_IDX:
      component = kGem5GptqGIdx;
      break;
    default:
      return false;
  }
  const uint64_t page_end = fault.file_offset + fault.page_size;
  const uint64_t range_end = fault.range_file_offset + fault.range_size;
  const uint64_t valid_start =
      fault.file_offset > fault.range_file_offset ? fault.file_offset :
                                                    fault.range_file_offset;
  const uint64_t valid_end = page_end < range_end ? page_end : range_end;
  if (valid_start >= valid_end) {
    return false;
  }
  const size_t slot_offset =
      static_cast<size_t>(fault.cache_slot) * fault.page_size;
  const uint64_t page_data_offset = valid_start - fault.file_offset;
  const uint64_t valid_size = valid_end - valid_start;
  if (slot_offset > cache_size || page_data_offset > cache_size - slot_offset ||
      valid_size > cache_size - slot_offset - page_data_offset) {
    return false;
  }
  span->component = component;
  span->tensor_offset = valid_start - fault.range_file_offset;
  span->data = static_cast<const uint8_t*>(cache) + slot_offset +
      static_cast<size_t>(page_data_offset);
  span->size = static_cast<size_t>(valid_size);
  return true;
}

bool BuildGem5GenericGptqResidentSpans(
    const opennpux_npu_weight_residency_header* residency,
    size_t residency_size, uint32_t command_id, uint32_t role_id,
    uint64_t expert_id, const void* cache, size_t cache_size,
    Gem5GenericGptqPageSpan* spans, size_t span_capacity,
    size_t* span_count) {
  if (residency == nullptr || cache == nullptr || spans == nullptr ||
      span_count == nullptr || residency->magic !=
          OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC ||
      residency->version != OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION ||
      residency->header_size != sizeof(*residency) ||
      residency->record_size !=
          sizeof(opennpux_npu_weight_residency_record) ||
      residency_size < sizeof(*residency) ||
      residency->capacity >
          (residency_size - sizeof(*residency)) /
              sizeof(opennpux_npu_weight_residency_record)) {
    return false;
  }
  const auto* records =
      reinterpret_cast<const opennpux_npu_weight_residency_record*>(
          residency + 1);
  size_t count = 0;
  for (uint32_t index = 0; index < residency->capacity; ++index) {
    const auto& record = records[index];
    if ((record.flags & OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID) == 0 ||
        record.command_id != command_id || record.role_id != role_id ||
        record.expert_id != expert_id) {
      continue;
    }
    if (count >= span_capacity) {
      return false;
    }
    opennpux_npu_page_fault fault = {};
    fault.command_id = record.command_id;
    fault.shard_index = record.shard_index;
    fault.file_offset = record.page_file_offset;
    fault.expert_id = record.expert_id;
    fault.role_id = record.role_id;
    fault.component_id = record.component_id;
    fault.range_file_offset = record.range_file_offset;
    fault.range_size = record.range_size;
    fault.cache_slot = record.cache_slot;
    fault.page_size = record.page_size;
    if (!BuildGem5GenericGptqPageSpan(
            fault, cache, cache_size, &spans[count])) {
      return false;
    }
    ++count;
  }
  if (count == 0) {
    return false;
  }
  *span_count = count;
  return true;
}

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

bool RunGem5RoutedGptqExperts(
    const opennpux_npu_operator_parameters& parameters, uint32_t rows,
    Gem5GenericConstBuffer input, const uint64_t* expert_ids,
    const float* route_weights, uint32_t active_experts,
    Gem5GptqExpertWeightsProvider provider, void* provider_opaque,
    Gem5GenericMutableBuffer output, Gem5GptqKernelStats* stats) {
  if (rows == 0 || parameters.output_features == 0 ||
      parameters.intermediate_features == 0 || input.data == nullptr ||
      expert_ids == nullptr || route_weights == nullptr ||
      active_experts == 0 || provider == nullptr || output.data == nullptr ||
      stats == nullptr) {
    return false;
  }
  uint64_t output_count = 0;
  uint64_t intermediate_count = 0;
  if (!ProductSize(rows, parameters.output_features, 1, &output_count) ||
      !ProductSize(rows, parameters.intermediate_features, 1,
                   &intermediate_count) ||
      output_count > output.size / sizeof(float) ||
      intermediate_count > SIZE_MAX / sizeof(float)) {
    return false;
  }

  std::vector<float> gate(intermediate_count);
  std::vector<float> up(intermediate_count);
  std::vector<float> activated(intermediate_count);
  std::vector<float> expert_output(output_count);
  auto* combined = static_cast<float*>(output.data);
  std::fill(combined, combined + output_count, 0.0f);
  *stats = {};

  for (uint32_t route = 0; route < active_experts; ++route) {
    if (!std::isfinite(route_weights[route])) {
      return false;
    }
    Gem5GenericGptqWeights gate_weights = {};
    Gem5GenericGptqWeights up_weights = {};
    Gem5GenericGptqWeights down_weights = {};
    if (!provider(provider_opaque, expert_ids[route], &gate_weights,
                  &up_weights, &down_weights)) {
      return false;
    }
    std::fill(expert_output.begin(), expert_output.end(), 0.0f);
    const Gem5GenericGptqExpertOperands operands = {
        input,
        gate_weights,
        up_weights,
        down_weights,
        {gate.data(), gate.size() * sizeof(float)},
        {up.data(), up.size() * sizeof(float)},
        {activated.data(), activated.size() * sizeof(float)},
        {expert_output.data(), expert_output.size() * sizeof(float)},
    };
    Gem5GptqKernelStats expert_stats = {};
    if (!RunGem5GenericGptqExpert(parameters, rows, operands, &expert_stats) ||
        !AddStats(expert_stats, stats)) {
      return false;
    }
    for (uint64_t index = 0; index < output_count; ++index) {
      combined[index] += route_weights[route] * expert_output[index];
      if (!std::isfinite(combined[index])) {
        return false;
      }
    }
  }

  const uint64_t combine_operations = output_count * active_experts * 2;
  const uint64_t combine_bytes_read =
      output_count * active_experts * sizeof(float) * 2;
  const uint64_t combine_bytes_written = output_count * sizeof(float);
  const Gem5GptqKernelStats combine_stats = {
      combine_operations,
      combine_bytes_read,
      combine_bytes_written,
      (combine_operations + 15) / 16 +
          (combine_bytes_read + combine_bytes_written + 15) / 16,
  };
  return AddStats(combine_stats, stats);
}
