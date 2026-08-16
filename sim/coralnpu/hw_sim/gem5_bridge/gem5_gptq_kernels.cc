#include "hw_sim/gem5_bridge/gem5_gptq_kernels.h"

#include <algorithm>
#include <cmath>
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

}  // namespace

bool RunGem5GptqInt4MatMul(
    const Gem5GptqMatMulConfig& config, const float* input,
    const uint32_t* qweight, const uint32_t* qzeros, const float* scales,
    const uint32_t* g_idx, float* output, Gem5GptqKernelStats* stats) {
  if (input == nullptr || qweight == nullptr || qzeros == nullptr ||
      scales == nullptr || output == nullptr || stats == nullptr ||
      config.rows == 0 || config.input_columns == 0 ||
      config.output_columns == 0 || config.group_size == 0 ||
      config.zero_bias > 15) {
    return false;
  }
  const uint32_t weight_rows = CeilDiv(config.input_columns, 8);
  const uint32_t groups = CeilDiv(config.input_columns, config.group_size);
  const uint32_t zero_columns = CeilDiv(config.output_columns, 8);
  if (!ProductFits(config.rows, config.input_columns, sizeof(float)) ||
      !ProductFits(weight_rows, config.output_columns, sizeof(uint32_t)) ||
      !ProductFits(groups, config.output_columns, sizeof(float)) ||
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
        const float scale =
            scales[static_cast<size_t>(group) * config.output_columns + column];
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
      config.output_columns * sizeof(float);
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
