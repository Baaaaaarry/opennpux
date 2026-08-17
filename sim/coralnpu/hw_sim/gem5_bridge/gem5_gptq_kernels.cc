#include "hw_sim/gem5_bridge/gem5_gptq_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace {

uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
  return value / divisor + (value % divisor != 0 ? 1 : 0);
}

bool MultiplyFits(uint64_t lhs, uint64_t rhs) {
  return rhs == 0 || lhs <= std::numeric_limits<uint64_t>::max() / rhs;
}

bool ProductFits(uint64_t first, uint64_t second, uint64_t third) {
  return MultiplyFits(first, second) &&
      MultiplyFits(first * second, third);
}

uint32_t ScaleElementSize(uint32_t data_type) {
  if (data_type == kGem5GptqScaleFloat16 ||
      data_type == kGem5GptqScaleBfloat16) {
    return sizeof(uint16_t);
  }
  return data_type == kGem5GptqScaleFloat32 ? sizeof(float) : 0;
}

float BitsToFloat(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float Float16ToFloat(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000) << 16;
  const uint32_t exponent = (value >> 10) & 0x1f;
  uint32_t mantissa = value & 0x3ff;
  if (exponent == 0) {
    if (mantissa == 0) {
      return BitsToFloat(sign);
    }
    uint32_t normalized_exponent = 113;
    while ((mantissa & 0x400) == 0) {
      mantissa <<= 1;
      --normalized_exponent;
    }
    return BitsToFloat(sign | (normalized_exponent << 23) |
                       ((mantissa & 0x3ff) << 13));
  }
  if (exponent == 0x1f) {
    return BitsToFloat(sign | UINT32_C(0x7f800000) | (mantissa << 13));
  }
  return BitsToFloat(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

float ReadScale(const void* scales, size_t index, uint32_t data_type) {
  if (data_type == kGem5GptqScaleFloat16) {
    return Float16ToFloat(static_cast<const uint16_t*>(scales)[index]);
  }
  if (data_type == kGem5GptqScaleBfloat16) {
    return BitsToFloat(
        static_cast<uint32_t>(static_cast<const uint16_t*>(scales)[index]) <<
        16);
  }
  return static_cast<const float*>(scales)[index];
}

}  // namespace

bool RunGem5GptqInt4MatMul(
    const Gem5GptqMatMulConfig& config, const float* input,
    const uint32_t* qweight, const uint32_t* qzeros, const void* scales,
    const uint32_t* g_idx, float* output, Gem5GptqKernelStats* stats) {
  if (input == nullptr || qweight == nullptr || qzeros == nullptr ||
      scales == nullptr || output == nullptr || stats == nullptr ||
      config.rows == 0 || config.input_columns == 0 ||
      config.output_columns == 0 || config.group_size == 0 ||
      config.zero_bias > 15 || ScaleElementSize(config.scale_data_type) == 0) {
    return false;
  }
  const uint32_t weight_rows = CeilDiv(config.input_columns, 8);
  const uint32_t groups = CeilDiv(config.input_columns, config.group_size);
  const uint32_t zero_columns = CeilDiv(config.output_columns, 8);
  if (!ProductFits(config.rows, config.input_columns, sizeof(float)) ||
      !ProductFits(weight_rows, config.output_columns, sizeof(uint32_t)) ||
      !ProductFits(groups, config.output_columns,
                   ScaleElementSize(config.scale_data_type)) ||
      !ProductFits(groups, zero_columns, sizeof(uint32_t)) ||
      !ProductFits(config.rows, config.output_columns, sizeof(float)) ||
      !ProductFits(config.rows, config.input_columns,
                   config.output_columns) ||
      (g_idx != nullptr &&
       !MultiplyFits(config.input_columns, sizeof(uint32_t)))) {
    return false;
  }

  for (uint32_t row = 0; row < config.rows; ++row) {
    for (uint32_t column = 0; column < config.output_columns; ++column) {
      float accumulator = 0.0f;
      for (uint32_t k = 0; k < config.input_columns; ++k) {
        const uint32_t group =
            g_idx == nullptr ? k / config.group_size : g_idx[k];
        if (group >= groups) {
          return false;
        }
        const uint32_t packed_weight =
            qweight[static_cast<size_t>(k / 8) * config.output_columns +
                    column];
        const uint32_t quantized = (packed_weight >> (4 * (k % 8))) & 0xf;
        const uint32_t packed_zero =
            qzeros[static_cast<size_t>(group) * zero_columns + column / 8];
        const uint32_t stored_zero =
            (packed_zero >> (4 * (column % 8))) & 0xf;
        const uint32_t zero = std::min(stored_zero + config.zero_bias, 15u);
        const float scale = ReadScale(
            scales, static_cast<size_t>(group) * config.output_columns + column,
            config.scale_data_type);
        if (!std::isfinite(scale)) {
          return false;
        }
        const float weight =
            (static_cast<int32_t>(quantized) - static_cast<int32_t>(zero)) *
            scale;
        accumulator +=
            input[static_cast<size_t>(row) * config.input_columns + k] * weight;
      }
      output[static_cast<size_t>(row) * config.output_columns + column] =
          accumulator;
    }
  }

  const uint64_t input_bytes = static_cast<uint64_t>(config.rows) *
      config.input_columns * sizeof(float);
  const uint64_t weight_bytes = static_cast<uint64_t>(weight_rows) *
      config.output_columns * sizeof(uint32_t);
  const uint64_t zero_bytes = static_cast<uint64_t>(groups) * zero_columns *
      sizeof(uint32_t);
  const uint64_t scale_bytes = static_cast<uint64_t>(groups) *
      config.output_columns * ScaleElementSize(config.scale_data_type);
  const uint64_t g_idx_bytes = g_idx == nullptr ? 0 :
      static_cast<uint64_t>(config.input_columns) * sizeof(uint32_t);
  const uint64_t output_bytes = static_cast<uint64_t>(config.rows) *
      config.output_columns * sizeof(float);
  const uint64_t multiply_accumulates = static_cast<uint64_t>(config.rows) *
      config.input_columns * config.output_columns;
  if (!MultiplyFits(multiply_accumulates, 2)) {
    return false;
  }
  stats->operations = multiply_accumulates * 2;
  stats->bytes_read = input_bytes + weight_bytes + zero_bytes + scale_bytes +
      g_idx_bytes;
  stats->bytes_written = output_bytes;
  stats->modeled_cycles = stats->operations / 2 +
      (stats->bytes_read + stats->bytes_written + 15) / 16;
  return true;
}
