#include "hw_sim/gem5_bridge/gem5_gptq_kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

enum class AccumulationMode {
  kFloat32,
  kFloat64,
  kGroupedFloat32,
  kWna16Grouped,
  kMma16,
  kMma16Bfloat16,
};

bool ReadAccumulationMode(AccumulationMode* mode) {
  const char* value = std::getenv("OPENNPUX_GPTQ_ACCUMULATION");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "fp32") == 0) {
    *mode = AccumulationMode::kFloat32;
    return true;
  }
  if (std::strcmp(value, "fp64") == 0) {
    *mode = AccumulationMode::kFloat64;
    return true;
  }
  if (std::strcmp(value, "grouped") == 0) {
    *mode = AccumulationMode::kGroupedFloat32;
    return true;
  }
  if (std::strcmp(value, "wna16") == 0) {
    *mode = AccumulationMode::kWna16Grouped;
    return true;
  }
  if (std::strcmp(value, "mma16") == 0) {
    *mode = AccumulationMode::kMma16;
    return true;
  }
  if (std::strcmp(value, "mma16-bf16") == 0) {
    *mode = AccumulationMode::kMma16Bfloat16;
    return true;
  }
  return false;
}

float RoundBfloat16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000)) {
    bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1U);
    bits &= UINT32_C(0xffff0000);
    std::memcpy(&value, &bits, sizeof(bits));
  }
  return value;
}

float ReduceMmaTile(std::array<float, 16>* values, uint32_t count) {
  while (count > 1) {
    uint32_t write = 0;
    for (uint32_t read = 0; read < count; read += 2) {
      (*values)[write++] = read + 1 < count
          ? (*values)[read] + (*values)[read + 1]
          : (*values)[read];
    }
    count = write;
  }
  return count == 0 ? 0.0f : (*values)[0];
}

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

uint32_t Gem5GptqScaleElementSize(uint32_t data_type) {
  return ScaleElementSize(data_type);
}

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
  AccumulationMode accumulation_mode = AccumulationMode::kFloat32;
  if (!ReadAccumulationMode(&accumulation_mode)) {
    return false;
  }

  for (uint32_t row = 0; row < config.rows; ++row) {
    for (uint32_t column = 0; column < config.output_columns; ++column) {
      float accumulator = 0.0f;
      double wide_accumulator = 0.0;
      float group_accumulator = 0.0f;
      float active_scale = 0.0f;
      uint32_t active_group = UINT32_MAX;
      std::array<float, 16> mma_products = {};
      uint32_t mma_product_count = 0;
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
        // GPTQ stores (zero_point - bias) in four bits. A stored nibble of 15
        // with bias 1 therefore represents zero point 16, not 15.
        const uint32_t zero = stored_zero + config.zero_bias;
        const float scale = ReadScale(
            scales, static_cast<size_t>(group) * config.output_columns + column,
            config.scale_data_type);
        if (!std::isfinite(scale)) {
          return false;
        }
        const int32_t centered_weight =
            static_cast<int32_t>(quantized) - static_cast<int32_t>(zero);
        float weight = static_cast<float>(centered_weight) * scale;
        if (accumulation_mode == AccumulationMode::kMma16Bfloat16) {
          weight = RoundBfloat16(weight);
        }
        const float product =
            input[static_cast<size_t>(row) * config.input_columns + k] * weight;
        if (accumulation_mode == AccumulationMode::kFloat64) {
          wide_accumulator += static_cast<double>(product);
        } else if (accumulation_mode == AccumulationMode::kGroupedFloat32) {
          if (active_group != UINT32_MAX && active_group != group) {
            accumulator += group_accumulator;
            group_accumulator = 0.0f;
          }
          active_group = group;
          group_accumulator += product;
        } else if (accumulation_mode == AccumulationMode::kWna16Grouped) {
          if (active_group != UINT32_MAX && active_group != group) {
            accumulator += group_accumulator * active_scale;
            group_accumulator = 0.0f;
          }
          active_group = group;
          active_scale = scale;
          group_accumulator +=
              input[static_cast<size_t>(row) * config.input_columns + k] *
              static_cast<float>(centered_weight);
        } else if (accumulation_mode == AccumulationMode::kMma16 ||
                   accumulation_mode == AccumulationMode::kMma16Bfloat16) {
          mma_products[mma_product_count++] = product;
          if (mma_product_count == mma_products.size()) {
            accumulator += ReduceMmaTile(&mma_products, mma_product_count);
            mma_product_count = 0;
          }
        } else {
          accumulator += product;
        }
      }
      if (accumulation_mode == AccumulationMode::kFloat64) {
        accumulator = static_cast<float>(wide_accumulator);
      } else if (accumulation_mode == AccumulationMode::kGroupedFloat32) {
        accumulator += group_accumulator;
      } else if (accumulation_mode == AccumulationMode::kWna16Grouped) {
        accumulator += group_accumulator * active_scale;
      } else if (accumulation_mode == AccumulationMode::kMma16 ||
                 accumulation_mode == AccumulationMode::kMma16Bfloat16) {
        accumulator += ReduceMmaTile(&mma_products, mma_product_count);
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

bool RunGem5GptqInt4MatMulStreamed(
    const Gem5GptqMatMulConfig& config, const float* input,
    Gem5GptqRead reader, void* reader_opaque, uint32_t output_tile_columns,
    bool has_g_idx, float* output, Gem5GptqKernelStats* stats) {
  if (input == nullptr || reader == nullptr || output == nullptr ||
      stats == nullptr || config.rows == 0 || config.input_columns == 0 ||
      config.output_columns == 0 || output_tile_columns == 0 ||
      output_tile_columns % 8 != 0 || config.group_size == 0 ||
      config.zero_bias > 15 || ScaleElementSize(config.scale_data_type) == 0) {
    return false;
  }
  const uint32_t weight_rows = CeilDiv(config.input_columns, 8);
  const uint32_t groups = CeilDiv(config.input_columns, config.group_size);
  const uint32_t global_zero_columns = CeilDiv(config.output_columns, 8);
  std::vector<uint32_t> g_idx(has_g_idx ? config.input_columns : 0);
  if (has_g_idx && !reader(reader_opaque, kGem5GptqGIdx, 0, g_idx.data(),
                           g_idx.size() * sizeof(g_idx[0]))) {
    return false;
  }
  *stats = {};
  for (uint32_t column_base = 0; column_base < config.output_columns;
       column_base += output_tile_columns) {
    const uint32_t tile_columns = std::min(
        output_tile_columns, config.output_columns - column_base);
    const uint32_t tile_zero_columns = CeilDiv(tile_columns, 8);
    std::vector<uint32_t> qweight(
        static_cast<size_t>(weight_rows) * tile_columns);
    std::vector<uint32_t> qzeros(
        static_cast<size_t>(groups) * tile_zero_columns);
    std::vector<uint8_t> scales(
        static_cast<size_t>(groups) * tile_columns *
        ScaleElementSize(config.scale_data_type));
    std::vector<float> tile_output(
        static_cast<size_t>(config.rows) * tile_columns);
    for (uint32_t packed_k = 0; packed_k < weight_rows; ++packed_k) {
      const uint64_t offset =
          (static_cast<uint64_t>(packed_k) * config.output_columns +
           column_base) * sizeof(uint32_t);
      if (!reader(reader_opaque, kGem5GptqQweight, offset,
                  qweight.data() + static_cast<size_t>(packed_k) * tile_columns,
                  static_cast<size_t>(tile_columns) * sizeof(uint32_t))) {
        return false;
      }
    }
    for (uint32_t group = 0; group < groups; ++group) {
      const uint64_t zero_offset =
          (static_cast<uint64_t>(group) * global_zero_columns +
           column_base / 8) * sizeof(uint32_t);
      if (!reader(reader_opaque, kGem5GptqQzeros, zero_offset,
                  qzeros.data() + static_cast<size_t>(group) * tile_zero_columns,
                  static_cast<size_t>(tile_zero_columns) * sizeof(uint32_t))) {
        return false;
      }
      const uint32_t scale_size = ScaleElementSize(config.scale_data_type);
      const uint64_t scale_offset =
          (static_cast<uint64_t>(group) * config.output_columns + column_base) *
          scale_size;
      if (!reader(reader_opaque, kGem5GptqScales, scale_offset,
                  scales.data() + static_cast<size_t>(group) * tile_columns *
                      scale_size,
                  static_cast<size_t>(tile_columns) * scale_size)) {
        return false;
      }
    }
    const Gem5GptqMatMulConfig tile_config = {
        config.rows, config.input_columns, tile_columns, config.group_size,
        config.zero_bias, config.scale_data_type};
    Gem5GptqKernelStats tile_stats = {};
    if (!RunGem5GptqInt4MatMul(
            tile_config, input, qweight.data(), qzeros.data(), scales.data(),
            has_g_idx ? g_idx.data() : nullptr, tile_output.data(),
            &tile_stats)) {
      return false;
    }
    for (uint32_t row = 0; row < config.rows; ++row) {
      std::copy_n(tile_output.data() + static_cast<size_t>(row) * tile_columns,
                  tile_columns,
                  output + static_cast<size_t>(row) * config.output_columns +
                      column_base);
    }
    if (tile_stats.operations > UINT64_MAX - stats->operations ||
        tile_stats.bytes_read > UINT64_MAX - stats->bytes_read ||
        tile_stats.bytes_written > UINT64_MAX - stats->bytes_written ||
        tile_stats.modeled_cycles > UINT64_MAX - stats->modeled_cycles) {
      return false;
    }
    stats->operations += tile_stats.operations;
    stats->bytes_read += tile_stats.bytes_read;
    stats->bytes_written += tile_stats.bytes_written;
    stats->modeled_cycles += tile_stats.modeled_cycles;
  }
  return true;
}
