#include "hw_sim/gem5_bridge/gem5_transformer_kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace {

constexpr uint64_t kOperationsPerCycle = 16;
constexpr uint64_t kBytesPerCycle = 16;

bool ProductFits(size_t first, size_t second, size_t* result) {
  if (result == nullptr ||
      (second != 0 && first > std::numeric_limits<size_t>::max() / second)) {
    return false;
  }
  *result = first * second;
  return true;
}

uint64_t DivCeil(uint64_t value, uint64_t divisor) {
  return value / divisor + (value % divisor != 0);
}

bool FinishStats(uint64_t operations, uint64_t elements_read,
                 uint64_t elements_written, Gem5TransformerKernelStats* stats) {
  if (stats == nullptr ||
      elements_read > UINT64_MAX / sizeof(float) ||
      elements_written > UINT64_MAX / sizeof(float)) {
    return false;
  }
  stats->operations = operations;
  stats->bytes_read = elements_read * sizeof(float);
  stats->bytes_written = elements_written * sizeof(float);
  stats->modeled_cycles = std::max<uint64_t>(
      DivCeil(operations, kOperationsPerCycle),
      DivCeil(stats->bytes_read + stats->bytes_written, kBytesPerCycle));
  return true;
}

bool FinishByteStats(uint64_t operations, uint64_t bytes_read,
                     uint64_t bytes_written,
                     Gem5TransformerKernelStats* stats) {
  if (stats == nullptr) {
    return false;
  }
  stats->operations = operations;
  stats->bytes_read = bytes_read;
  stats->bytes_written = bytes_written;
  stats->modeled_cycles = std::max<uint64_t>(
      DivCeil(operations, kOperationsPerCycle),
      DivCeil(bytes_read + bytes_written, kBytesPerCycle));
  return true;
}

bool ValidBuffers(const float* input, size_t count, float* output,
                  Gem5TransformerKernelStats* stats) {
  return input != nullptr && count != 0 && output != nullptr && stats != nullptr;
}

}  // namespace

bool RunGem5EmbeddingF32(const uint32_t* token_ids, size_t token_count,
                         const float* table, size_t vocabulary_size,
                         size_t features, float* output,
                         Gem5TransformerKernelStats* stats) {
  size_t output_elements = 0;
  size_t table_elements = 0;
  if (token_ids == nullptr || token_count == 0 || table == nullptr ||
      vocabulary_size == 0 || features == 0 || output == nullptr ||
      stats == nullptr ||
      !ProductFits(token_count, features, &output_elements) ||
      !ProductFits(vocabulary_size, features, &table_elements) ||
      output_elements > UINT64_MAX / sizeof(float) ||
      token_count > UINT64_MAX / sizeof(uint32_t)) {
    return false;
  }
  for (size_t token = 0; token < token_count; ++token) {
    if (token_ids[token] >= vocabulary_size) {
      return false;
    }
    std::copy_n(table + static_cast<size_t>(token_ids[token]) * features,
                features, output + token * features);
  }
  return FinishByteStats(
      output_elements,
      token_count * sizeof(uint32_t) + output_elements * sizeof(float),
      output_elements * sizeof(float), stats);
}

bool RunGem5MatMulF32(const float* input, const float* weight, size_t rows,
                      size_t input_features, size_t output_features,
                      float* output, Gem5TransformerKernelStats* stats) {
  size_t input_elements = 0;
  size_t weight_elements = 0;
  size_t output_elements = 0;
  if (!ProductFits(rows, input_features, &input_elements) ||
      !ProductFits(input_features, output_features, &weight_elements) ||
      !ProductFits(rows, output_features, &output_elements) ||
      !ValidBuffers(input, input_elements, output, stats) ||
      weight == nullptr || output_features == 0 ||
      output_elements > UINT64_MAX / 2 ||
      output_elements * 2 > UINT64_MAX / input_features) {
    return false;
  }
  for (size_t row = 0; row < rows; ++row) {
    for (size_t column = 0; column < output_features; ++column) {
      float sum = 0.0f;
      for (size_t inner = 0; inner < input_features; ++inner) {
        sum += input[row * input_features + inner] *
               weight[column * input_features + inner];
      }
      output[row * output_features + column] = sum;
    }
  }
  return FinishStats(static_cast<uint64_t>(output_elements) *
                         input_features * 2,
                     input_elements + weight_elements, output_elements,
                     stats);
}

bool RunGem5AddF32(const float* lhs, const float* rhs, size_t count,
                   float* output, Gem5TransformerKernelStats* stats) {
  if (!ValidBuffers(lhs, count, output, stats) || rhs == nullptr ||
      count > UINT64_MAX / 2) {
    return false;
  }
  for (size_t index = 0; index < count; ++index) {
    output[index] = lhs[index] + rhs[index];
  }
  return FinishStats(count, count * 2, count, stats);
}

bool RunGem5MulF32(const float* lhs, const float* rhs, size_t count,
                   float* output, Gem5TransformerKernelStats* stats) {
  if (!ValidBuffers(lhs, count, output, stats) || rhs == nullptr ||
      count > UINT64_MAX / 2) {
    return false;
  }
  for (size_t index = 0; index < count; ++index) {
    output[index] = lhs[index] * rhs[index];
  }
  return FinishStats(count, count * 2, count, stats);
}

bool RunGem5SiluF32(const float* input, size_t count, float* output,
                    Gem5TransformerKernelStats* stats) {
  if (!ValidBuffers(input, count, output, stats)) {
    return false;
  }
  for (size_t index = 0; index < count; ++index) {
    output[index] = input[index] / (1.0f + std::exp(-input[index]));
    if (!std::isfinite(output[index])) {
      return false;
    }
  }
  return FinishStats(count * 4, count, count, stats);
}

bool RunGem5RmsNormF32(const float* input, const float* weight, size_t rows,
                       size_t features, float epsilon, float* output,
                       Gem5TransformerKernelStats* stats,
                       bool weight_offset) {
  size_t count = 0;
  if (!ProductFits(rows, features, &count) ||
      !ValidBuffers(input, count, output, stats) || weight == nullptr ||
      !std::isfinite(epsilon) || epsilon < 0.0f || count > UINT64_MAX / 4) {
    return false;
  }
  for (size_t row = 0; row < rows; ++row) {
    double sum_squares = 0.0;
    for (size_t column = 0; column < features; ++column) {
      const float value = input[row * features + column];
      sum_squares += static_cast<double>(value) * value;
    }
    const float inverse_rms = 1.0f / std::sqrt(
        static_cast<float>(sum_squares / features) + epsilon);
    if (!std::isfinite(inverse_rms)) {
      return false;
    }
    for (size_t column = 0; column < features; ++column) {
      const float scale = weight[column] + (weight_offset ? 1.0f : 0.0f);
      output[row * features + column] =
          input[row * features + column] * inverse_rms * scale;
    }
  }
  const uint64_t operations = static_cast<uint64_t>(count) * 4 + rows * 2;
  return FinishStats(operations, count * 2, count, stats);
}

bool RunGem5GatedRmsNormF32(
    const float* input, const float* gate, const float* weight, size_t rows,
    size_t heads, size_t head_dim, float epsilon, float* output,
    Gem5TransformerKernelStats* stats) {
  size_t features = 0;
  size_t count = 0;
  if (!ProductFits(heads, head_dim, &features) ||
      !ProductFits(rows, features, &count) ||
      !ValidBuffers(input, count, output, stats) || gate == nullptr ||
      weight == nullptr || !std::isfinite(epsilon) || epsilon < 0.0f) {
    return false;
  }
  for (size_t row = 0; row < rows; ++row) {
    for (size_t head = 0; head < heads; ++head) {
      const size_t base = row * features + head * head_dim;
      double sum_squares = 0.0;
      for (size_t column = 0; column < head_dim; ++column) {
        const float value = input[base + column];
        sum_squares += static_cast<double>(value) * value;
      }
      const float inverse_rms = 1.0f / std::sqrt(
          static_cast<float>(sum_squares / head_dim) + epsilon);
      if (!std::isfinite(inverse_rms)) {
        return false;
      }
      for (size_t column = 0; column < head_dim; ++column) {
        const float gate_value = gate[base + column];
        const float silu_gate = gate_value / (1.0f + std::exp(-gate_value));
        output[base + column] = input[base + column] * inverse_rms *
                                weight[column] * silu_gate;
        if (!std::isfinite(output[base + column])) {
          return false;
        }
      }
    }
  }
  return FinishStats(static_cast<uint64_t>(count) * 8 + rows * heads * 2,
                     count * 3 + head_dim, count, stats);
}

bool RunGem5SoftmaxF32(const float* input, size_t rows, size_t features,
                       float* output, Gem5TransformerKernelStats* stats) {
  size_t count = 0;
  if (!ProductFits(rows, features, &count) ||
      !ValidBuffers(input, count, output, stats) || count > UINT64_MAX / 4) {
    return false;
  }
  for (size_t row = 0; row < rows; ++row) {
    const float* row_input = input + row * features;
    float* row_output = output + row * features;
    const float maximum =
        *std::max_element(row_input, row_input + features);
    double sum = 0.0;
    for (size_t column = 0; column < features; ++column) {
      row_output[column] = std::exp(row_input[column] - maximum);
      sum += row_output[column];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
      return false;
    }
    for (size_t column = 0; column < features; ++column) {
      row_output[column] /= static_cast<float>(sum);
    }
  }
  return FinishStats(static_cast<uint64_t>(count) * 4, count, count, stats);
}

bool RunGem5RopeF32(const float* input, const uint32_t* positions, size_t rows,
                    size_t heads, size_t head_dim, size_t rotary_dim,
                    float theta, float* output,
                    Gem5TransformerKernelStats* stats) {
  size_t row_elements = 0;
  size_t count = 0;
  if (!ProductFits(heads, head_dim, &row_elements) ||
      !ProductFits(rows, row_elements, &count) ||
      !ValidBuffers(input, count, output, stats) || positions == nullptr ||
      head_dim == 0 || rotary_dim == 0 || rotary_dim > head_dim ||
      rotary_dim % 2 != 0 || !(theta > 0.0f) ||
      !std::isfinite(theta) || count > UINT64_MAX / 3) {
    return false;
  }
  for (size_t row = 0; row < rows; ++row) {
    for (size_t head = 0; head < heads; ++head) {
      const size_t base = row * row_elements + head * head_dim;
      const size_t half = rotary_dim / 2;
      for (size_t pair = 0; pair < half; ++pair) {
        const float exponent =
            -2.0f * static_cast<float>(pair) / static_cast<float>(rotary_dim);
        const float angle = static_cast<float>(positions[row]) *
                            std::pow(theta, exponent);
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        const float first = input[base + pair];
        const float second = input[base + half + pair];
        output[base + pair] = first * cosine - second * sine;
        output[base + half + pair] = second * cosine + first * sine;
      }
      std::copy(input + base + rotary_dim, input + base + head_dim,
                output + base + rotary_dim);
    }
  }
  return FinishStats(static_cast<uint64_t>(count) * 3, count + rows, count,
                     stats);
}

bool RunGem5CausalDepthwiseConvF32(
    const float* input, const float* weight, size_t rows, size_t features,
    size_t kernel_width, float* output, Gem5TransformerKernelStats* stats,
    bool silu_activation) {
  size_t count = 0;
  size_t weight_count = 0;
  if (!ProductFits(rows, features, &count) ||
      !ProductFits(kernel_width, features, &weight_count) ||
      !ValidBuffers(input, count, output, stats) || weight == nullptr ||
      kernel_width == 0 || count > UINT64_MAX / kernel_width / 2) {
    return false;
  }
  for (size_t row = 0; row < rows; ++row) {
    for (size_t feature = 0; feature < features; ++feature) {
      float sum = 0.0f;
      for (size_t tap = 0; tap < kernel_width; ++tap) {
        if (tap > row) {
          continue;
        }
        sum += input[(row - tap) * features + feature] *
               weight[(kernel_width - 1 - tap) * features + feature];
      }
      output[row * features + feature] =
          silu_activation ? sum / (1.0f + std::exp(-sum)) : sum;
    }
  }
  return FinishStats(static_cast<uint64_t>(count) * kernel_width * 2,
                     count + weight_count, count, stats);
}

bool RunGem5TopKF32(const float* input, size_t count, size_t k,
                    float* output_values, uint32_t* output_indices,
                    Gem5TransformerKernelStats* stats) {
  if (input == nullptr || count == 0 || k == 0 || k > count ||
      count > UINT32_MAX || output_values == nullptr ||
      output_indices == nullptr || stats == nullptr) {
    return false;
  }
  std::vector<uint32_t> indices(count);
  std::iota(indices.begin(), indices.end(), 0);
  const auto order = [input](uint32_t lhs, uint32_t rhs) {
    if (input[lhs] == input[rhs]) {
      return lhs < rhs;
    }
    return input[lhs] > input[rhs];
  };
  std::partial_sort(indices.begin(), indices.begin() + k, indices.end(), order);
  for (size_t index = 0; index < k; ++index) {
    output_indices[index] = indices[index];
    output_values[index] = input[indices[index]];
  }
  if (!FinishStats(static_cast<uint64_t>(count) * k, count, k, stats)) {
    return false;
  }
  stats->bytes_written += k * sizeof(uint32_t);
  stats->modeled_cycles = std::max<uint64_t>(
      stats->modeled_cycles,
      DivCeil(stats->bytes_read + stats->bytes_written, kBytesPerCycle));
  return true;
}

bool RunGem5KvCacheUpdateF32(
    const float* key, const float* value, size_t token_count,
    size_t kv_heads, size_t head_dim, size_t kv_length, float* state,
    Gem5TransformerKernelStats* stats) {
  size_t token_elements = 0;
  size_t plane_elements = 0;
  if (!ProductFits(kv_heads, head_dim, &token_elements) ||
      !ProductFits(kv_length, token_elements, &plane_elements) ||
      key == nullptr || value == nullptr || state == nullptr ||
      token_count == 0 || token_count > kv_length || stats == nullptr) {
    return false;
  }
  const size_t destination = (kv_length - token_count) * token_elements;
  const size_t copy_elements = token_count * token_elements;
  std::copy_n(key, copy_elements, state + destination);
  std::copy_n(value, copy_elements, state + plane_elements + destination);
  return FinishStats(copy_elements * 2, copy_elements * 2,
                     copy_elements * 2, stats);
}

bool RunGem5AttentionF32(
    const float* query, const float* state, size_t query_rows,
    size_t heads, size_t kv_heads, size_t head_dim, size_t kv_length,
    float* output, Gem5TransformerKernelStats* stats) {
  size_t query_features = 0;
  size_t kv_features = 0;
  size_t query_elements = 0;
  size_t plane_elements = 0;
  if (!ProductFits(heads, head_dim, &query_features) ||
      !ProductFits(kv_heads, head_dim, &kv_features) ||
      !ProductFits(query_rows, query_features, &query_elements) ||
      !ProductFits(kv_length, kv_features, &plane_elements) ||
      query == nullptr || state == nullptr || output == nullptr ||
      stats == nullptr || heads == 0 || kv_heads == 0 || head_dim == 0 ||
      kv_length == 0 || query_rows > kv_length || heads % kv_heads != 0) {
    return false;
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> scores(kv_length);
  const float* keys = state;
  const float* values = state + plane_elements;
  const size_t query_start = kv_length - query_rows;
  const size_t query_heads_per_kv_head = heads / kv_heads;
  for (size_t row = 0; row < query_rows; ++row) {
    const size_t visible_positions = query_start + row + 1;
    for (size_t head = 0; head < heads; ++head) {
      // repeat_kv expands each KV head into one contiguous group of query
      // heads. A modulo mapping incorrectly interleaves those groups.
      const size_t kv_head = head / query_heads_per_kv_head;
      const float* q = query + row * query_features + head * head_dim;
      float maximum = -std::numeric_limits<float>::infinity();
      for (size_t position = 0; position < visible_positions; ++position) {
        const float* k = keys + position * kv_features + kv_head * head_dim;
        float score = 0.0f;
        for (size_t column = 0; column < head_dim; ++column) {
          score += q[column] * k[column];
        }
        scores[position] = score * scale;
        maximum = std::max(maximum, scores[position]);
      }
      float sum = 0.0f;
      for (size_t position = 0; position < visible_positions; ++position) {
        scores[position] = std::exp(scores[position] - maximum);
        sum += scores[position];
      }
      if (!(sum > 0.0f) || !std::isfinite(sum)) {
        return false;
      }
      float* destination =
          output + row * query_features + head * head_dim;
      std::fill_n(destination, head_dim, 0.0f);
      for (size_t position = 0; position < visible_positions; ++position) {
        const float probability = scores[position] / sum;
        const float* v =
            values + position * kv_features + kv_head * head_dim;
        for (size_t column = 0; column < head_dim; ++column) {
          destination[column] += probability * v[column];
        }
      }
    }
  }
  const uint64_t visible_position_count =
      static_cast<uint64_t>(query_rows) * (query_start + 1) +
      static_cast<uint64_t>(query_rows) * (query_rows - 1) / 2;
  const uint64_t operations = visible_position_count * heads * head_dim * 4;
  return FinishStats(operations, query_elements + plane_elements * 2,
                     query_elements, stats);
}

bool RunGem5RecurrentUpdateF32(
    const float* input, size_t rows, size_t features, float* output,
    float* state,
    Gem5TransformerKernelStats* stats) {
  size_t count = 0;
  if (!ProductFits(rows, features, &count) ||
      !ValidBuffers(input, count, output, stats) || state == nullptr) {
    return false;
  }
  std::copy_n(input, count, output);
  std::copy_n(input + (rows - 1) * features, features, state);
  return FinishStats(count + features, count, count + features, stats);
}

bool RunGem5GatedDeltaNetF32(
    const float* qkv, const float* alpha, const float* beta,
    const float* a_log, const float* dt_bias, size_t rows,
    size_t key_heads, size_t value_heads, size_t key_dim, size_t value_dim,
    float* output, float* state, Gem5TransformerKernelStats* stats) {
  if (qkv == nullptr || alpha == nullptr || beta == nullptr ||
      a_log == nullptr || dt_bias == nullptr || output == nullptr ||
      state == nullptr || stats == nullptr || rows == 0 || key_heads == 0 ||
      value_heads == 0 || key_dim == 0 || value_dim == 0 ||
      value_heads % key_heads != 0) {
    return false;
  }
  size_t key_features = 0;
  size_t value_features = 0;
  size_t state_per_head = 0;
  size_t qkv_count = 0;
  size_t output_count = 0;
  size_t state_count = 0;
  if (!ProductFits(key_heads, key_dim, &key_features) ||
      !ProductFits(value_heads, value_dim, &value_features) ||
      !ProductFits(key_dim, value_dim, &state_per_head) ||
      key_features >
          (std::numeric_limits<size_t>::max() - value_features) / 2) {
    return false;
  }
  const size_t qkv_features = 2 * key_features + value_features;
  if (!ProductFits(rows, qkv_features, &qkv_count) ||
      !ProductFits(rows, value_features, &output_count) ||
      !ProductFits(value_heads, state_per_head, &state_count)) {
    return false;
  }
  const size_t key_head_repeat = value_heads / key_heads;
  const float query_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
  std::vector<float> normalized_q(key_dim);
  std::vector<float> normalized_k(key_dim);
  std::vector<float> delta(value_dim);
  uint64_t operations = 0;

  for (size_t row = 0; row < rows; ++row) {
    const float* row_q = qkv + row * qkv_features;
    const float* row_k = row_q + key_features;
    const float* row_v = row_k + key_features;
    for (size_t value_head = 0; value_head < value_heads; ++value_head) {
      const size_t key_head = value_head / key_head_repeat;
      const float* q = row_q + key_head * key_dim;
      const float* k = row_k + key_head * key_dim;
      const float* v = row_v + value_head * value_dim;
      double q_sum = 0.0;
      double k_sum = 0.0;
      for (size_t index = 0; index < key_dim; ++index) {
        q_sum += static_cast<double>(q[index]) * q[index];
        k_sum += static_cast<double>(k[index]) * k[index];
      }
      const float q_norm = 1.0f / std::sqrt(static_cast<float>(q_sum) + 1e-6f);
      const float k_norm = 1.0f / std::sqrt(static_cast<float>(k_sum) + 1e-6f);
      for (size_t index = 0; index < key_dim; ++index) {
        normalized_q[index] = q[index] * q_norm * query_scale;
        normalized_k[index] = k[index] * k_norm;
      }
      const float beta_value =
          1.0f / (1.0f + std::exp(-beta[row * value_heads + value_head]));
      const float alpha_value = alpha[row * value_heads + value_head] +
                                dt_bias[value_head];
      const float softplus = alpha_value > 20.0f
                                 ? alpha_value
                                 : std::log1p(std::exp(alpha_value));
      const float decay = std::exp(-std::exp(a_log[value_head]) * softplus);
      float* head_state = state + value_head * state_per_head;
      for (size_t value_index = 0; value_index < value_dim; ++value_index) {
        float memory_value = 0.0f;
        for (size_t key_index = 0; key_index < key_dim; ++key_index) {
          float& cell = head_state[key_index * value_dim + value_index];
          cell *= decay;
          memory_value += cell * normalized_k[key_index];
        }
        delta[value_index] = (v[value_index] - memory_value) * beta_value;
      }
      for (size_t key_index = 0; key_index < key_dim; ++key_index) {
        for (size_t value_index = 0; value_index < value_dim; ++value_index) {
          head_state[key_index * value_dim + value_index] +=
              normalized_k[key_index] * delta[value_index];
        }
      }
      float* head_output = output + row * value_features +
                           value_head * value_dim;
      for (size_t value_index = 0; value_index < value_dim; ++value_index) {
        float value = 0.0f;
        for (size_t key_index = 0; key_index < key_dim; ++key_index) {
          value += head_state[key_index * value_dim + value_index] *
                   normalized_q[key_index];
        }
        head_output[value_index] = value;
      }
      operations += key_dim * 4 + value_dim * key_dim * 6 + value_dim * 3 + 8;
    }
  }
  return FinishStats(operations,
                     qkv_count + rows * value_heads * 2 + value_heads * 2 +
                         state_count,
                     output_count + state_count, stats);
}

bool RunGem5CombineF32(
    const float* routed, const float* shared, size_t count, float* output,
    Gem5TransformerKernelStats* stats) {
  return RunGem5AddF32(routed, shared, count, output, stats);
}

bool RunGem5SharedExpertF32(
    const float* input, const float* gate_weight, const float* up_weight,
    const float* down_weight, const float* router_weight, size_t rows,
    size_t input_features, size_t intermediate_features,
    size_t output_features, float* output,
    Gem5TransformerKernelStats* stats) {
  size_t intermediate_elements = 0;
  size_t output_elements = 0;
  if (input == nullptr || gate_weight == nullptr || up_weight == nullptr ||
      down_weight == nullptr || router_weight == nullptr || output == nullptr ||
      stats == nullptr || rows == 0 || input_features == 0 ||
      intermediate_features == 0 || output_features == 0 ||
      !ProductFits(rows, intermediate_features, &intermediate_elements) ||
      !ProductFits(rows, output_features, &output_elements)) {
    return false;
  }
  std::vector<float> gate(intermediate_elements);
  std::vector<float> up(intermediate_elements);
  std::vector<float> activated(intermediate_elements);
  std::vector<float> shared(output_elements);
  std::vector<float> router(rows);
  Gem5TransformerKernelStats stages[6] = {};
  if (!RunGem5MatMulF32(input, gate_weight, rows, input_features,
                        intermediate_features, gate.data(), &stages[0]) ||
      !RunGem5MatMulF32(input, up_weight, rows, input_features,
                        intermediate_features, up.data(), &stages[1]) ||
      !RunGem5SiluF32(gate.data(), intermediate_elements, gate.data(),
                      &stages[2]) ||
      !RunGem5MulF32(gate.data(), up.data(), intermediate_elements,
                     activated.data(), &stages[3]) ||
      !RunGem5MatMulF32(activated.data(), down_weight, rows,
                        intermediate_features, output_features,
                        shared.data(), &stages[4]) ||
      !RunGem5MatMulF32(input, router_weight, rows, input_features, 1,
                        router.data(), &stages[5])) {
    return false;
  }
  *stats = {};
  for (const auto& stage : stages) {
    stats->operations += stage.operations;
    stats->bytes_read += stage.bytes_read;
    stats->bytes_written += stage.bytes_written;
    stats->modeled_cycles += stage.modeled_cycles;
  }
  for (size_t row = 0; row < rows; ++row) {
    const float scale = 1.0f / (1.0f + std::exp(-router[row]));
    for (size_t column = 0; column < output_features; ++column) {
      output[row * output_features + column] =
          shared[row * output_features + column] * scale;
    }
  }
  const uint64_t scale_operations = rows * 4 + output_elements;
  const uint64_t scale_bytes = (rows + output_elements) * sizeof(float);
  stats->operations += scale_operations;
  stats->bytes_read += scale_bytes;
  stats->bytes_written += output_elements * sizeof(float);
  stats->modeled_cycles += std::max<uint64_t>(
      DivCeil(scale_operations, kOperationsPerCycle),
      DivCeil(scale_bytes + output_elements * sizeof(float), kBytesPerCycle));
  return true;
}

bool RunGem5FloatQkvF32(
    const float* input, const float* q_weight, const float* k_weight,
    const float* v_weight, const float* q_norm_weight,
    const float* k_norm_weight, size_t rows, size_t input_features,
    size_t heads, size_t kv_heads, size_t head_dim, size_t q_weight_outputs,
    float epsilon, bool norm_weight_offset, float* query, float* key,
    float* value, float* gate,
    Gem5TransformerKernelStats* stats) {
  size_t query_features = 0;
  size_t key_features = 0;
  size_t raw_query_elements = 0;
  size_t query_elements = 0;
  size_t key_elements = 0;
  if (input == nullptr || q_weight == nullptr || k_weight == nullptr ||
      v_weight == nullptr || q_norm_weight == nullptr ||
      k_norm_weight == nullptr || query == nullptr || key == nullptr ||
      value == nullptr || stats == nullptr || rows == 0 ||
      input_features == 0 || heads == 0 || kv_heads == 0 || head_dim == 0 ||
      !ProductFits(heads, head_dim, &query_features) ||
      !ProductFits(kv_heads, head_dim, &key_features) ||
      query_features > std::numeric_limits<size_t>::max() / 2 ||
      (q_weight_outputs != query_features &&
       q_weight_outputs != 2 * query_features) ||
      (q_weight_outputs == 2 * query_features && gate == nullptr) ||
      !ProductFits(rows, q_weight_outputs, &raw_query_elements) ||
      !ProductFits(rows, query_features, &query_elements) ||
      !ProductFits(rows, key_features, &key_elements)) {
    return false;
  }
  std::vector<float> raw_query(raw_query_elements);
  std::vector<float> raw_key(key_elements);
  Gem5TransformerKernelStats stages[5] = {};
  if (!RunGem5MatMulF32(input, q_weight, rows, input_features,
                        q_weight_outputs, raw_query.data(), &stages[0]) ||
      !RunGem5MatMulF32(input, k_weight, rows, input_features, key_features,
                        raw_key.data(), &stages[1]) ||
      !RunGem5MatMulF32(input, v_weight, rows, input_features, key_features,
                        value, &stages[2])) {
    return false;
  }
  if (q_weight_outputs == query_features) {
    std::copy_n(raw_query.data(), query_elements, query);
  } else {
    for (size_t row = 0; row < rows; ++row) {
      std::copy_n(raw_query.data() + row * q_weight_outputs, query_features,
                  query + row * query_features);
      std::copy_n(raw_query.data() + row * q_weight_outputs + query_features,
                  query_features, gate + row * query_features);
    }
  }
  if (!RunGem5RmsNormF32(query, q_norm_weight, rows * heads, head_dim,
                         epsilon, query, &stages[3], norm_weight_offset) ||
      !RunGem5RmsNormF32(raw_key.data(), k_norm_weight, rows * kv_heads,
                         head_dim, epsilon, key, &stages[4],
                         norm_weight_offset)) {
    return false;
  }
  *stats = {};
  for (const auto& stage : stages) {
    stats->operations += stage.operations;
    stats->bytes_read += stage.bytes_read;
    stats->bytes_written += stage.bytes_written;
    stats->modeled_cycles += stage.modeled_cycles;
  }
  return true;
}
