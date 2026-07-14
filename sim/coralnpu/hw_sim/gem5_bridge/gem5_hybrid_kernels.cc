#include "hw_sim/gem5_bridge/gem5_hybrid_kernels.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
  if (a == INT32_MIN && b == INT32_MIN) {
    return INT32_MAX;
  }
  const int64_t product = static_cast<int64_t>(a) * b;
  const int64_t nudge = product >= 0 ? (INT64_C(1) << 30) :
                                       (INT64_C(1) - (INT64_C(1) << 30));
  return static_cast<int32_t>((product + nudge) / (INT64_C(1) << 31));
}

int32_t RoundingDivideByPowerOfTwo(int32_t value, int exponent) {
  if (exponent <= 0) {
    return value;
  }
  const uint32_t mask = (UINT32_C(1) << exponent) - 1;
  const uint32_t remainder = static_cast<uint32_t>(value) & mask;
  const uint32_t threshold = (mask >> 1) + (value < 0 ? 1 : 0);
  return (value >> exponent) + (remainder > threshold ? 1 : 0);
}

int32_t MultiplyByQuantizedMultiplier(
    int32_t value, int32_t multiplier, int32_t shift) {
  const int left_shift = std::max(shift, 0);
  const int right_shift = std::max(-shift, 0);
  const int64_t shifted = static_cast<int64_t>(value) *
      (INT64_C(1) << left_shift);
  const int32_t bounded = static_cast<int32_t>(std::max<int64_t>(
      INT32_MIN, std::min<int64_t>(INT32_MAX, shifted)));
  return RoundingDivideByPowerOfTwo(
      SaturatingRoundingDoublingHighMul(bounded, multiplier), right_shift);
}

uint64_t ElementCount(const coral_operator_tensor& tensor) {
  if (tensor.rank == 0 || tensor.rank > CORAL_OPERATOR_MAX_DIMS) return 0;
  uint64_t count = 1;
  for (uint32_t i = 0; i < tensor.rank; ++i) {
    const uint32_t dim = tensor.dimensions[i];
    if (dim == 0 || count > UINT64_MAX / dim) {
      return 0;
    }
    count *= dim;
  }
  return count;
}

template <class T>
T* TensorData(uint8_t* extmem, uint32_t base,
              const coral_operator_tensor& tensor) {
  return reinterpret_cast<T*>(extmem + (tensor.address - base));
}

const float* TensorFloatData(
    uint8_t* extmem, uint32_t base, const coral_operator_tensor& tensor) {
  return reinterpret_cast<const float*>(extmem + (tensor.address - base));
}

float* MutableTensorFloatData(
    uint8_t* extmem, uint32_t base, const coral_operator_tensor& tensor) {
  return reinterpret_cast<float*>(extmem + (tensor.address - base));
}

bool FloatBits(uint32_t bits, float* value) {
  if (value == nullptr) return false;
  std::memcpy(value, &bits, sizeof(*value));
  return std::isfinite(*value);
}

bool TensorBytesMatch(const coral_operator_tensor& tensor,
                      uint32_t element_size) {
  const uint64_t count = ElementCount(tensor);
  return count != 0 && count <= UINT32_MAX / element_size &&
         tensor.size == count * element_size;
}

bool SameShape(const coral_operator_tensor& lhs,
               const coral_operator_tensor& rhs) {
  if (lhs.rank != rhs.rank) return false;
  for (uint32_t i = 0; i < lhs.rank; ++i) {
    if (lhs.dimensions[i] != rhs.dimensions[i]) return false;
  }
  return true;
}

bool CommonConvValid(const coral_operator_descriptor& descriptor,
                     bool depthwise) {
  if (descriptor.tensor_count != 4 || descriptor.quantization_count == 0 ||
      descriptor.stride_height == 0 || descriptor.stride_width == 0) {
    return false;
  }
  const auto& input = descriptor.tensors[0];
  const auto& filter = descriptor.tensors[1];
  const auto& bias = descriptor.tensors[2];
  const auto& output = descriptor.tensors[3];
  if (input.element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      filter.element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      bias.element_type != CORAL_OPERATOR_ELEMENT_INT32 ||
      output.element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      input.rank != 4 || filter.rank != 4 || bias.rank != 1 ||
      output.rank != 4 ||
      input.size != ElementCount(input) ||
      filter.size != ElementCount(filter) ||
      bias.size != ElementCount(bias) * sizeof(int32_t) ||
      output.size != ElementCount(output) || input.dimensions[0] == 0 ||
      output.dimensions[0] != input.dimensions[0] ||
      descriptor.quantization_count != output.dimensions[3]) {
    return false;
  }
  if (descriptor.activation_min < INT8_MIN ||
      descriptor.activation_max > INT8_MAX ||
      descriptor.activation_min > descriptor.activation_max) {
    return false;
  }
  if (depthwise) {
    return filter.dimensions[0] == 1 &&
           filter.dimensions[3] == output.dimensions[3] &&
           output.dimensions[3] % input.dimensions[3] == 0;
  }
  return filter.dimensions[0] == output.dimensions[3] &&
         filter.dimensions[3] == input.dimensions[3];
}

}  // namespace

bool RunGem5HybridConv2D(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  (void)extmem_size;
  if (!CommonConvValid(*descriptor, false)) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto& input = descriptor->tensors[0];
  const auto& filter = descriptor->tensors[1];
  const auto& bias = descriptor->tensors[2];
  const auto& output = descriptor->tensors[3];
  const int8_t* input_data = TensorData<int8_t>(extmem, extmem_base, input);
  const int8_t* filter_data = TensorData<int8_t>(extmem, extmem_base, filter);
  const int32_t* bias_data = TensorData<int32_t>(extmem, extmem_base, bias);
  int8_t* output_data = TensorData<int8_t>(extmem, extmem_base, output);
  const int32_t* multipliers = reinterpret_cast<const int32_t*>(
      extmem + descriptor->multiplier_address - extmem_base);
  const int32_t* shifts = reinterpret_cast<const int32_t*>(
      extmem + descriptor->shift_address - extmem_base);
  const uint32_t batches = input.dimensions[0];
  const uint32_t input_h = input.dimensions[1];
  const uint32_t input_w = input.dimensions[2];
  const uint32_t input_c = input.dimensions[3];
  const uint32_t output_h = output.dimensions[1];
  const uint32_t output_w = output.dimensions[2];
  const uint32_t output_c = output.dimensions[3];
  const uint32_t filter_h = filter.dimensions[1];
  const uint32_t filter_w = filter.dimensions[2];
  for (uint32_t i = 0; i < output_c; ++i) {
    if (shifts[i] < -31 || shifts[i] > 30) return false;
  }

  for (uint32_t batch = 0; batch < batches; ++batch) {
    for (uint32_t out_y = 0; out_y < output_h; ++out_y) {
      for (uint32_t out_x = 0; out_x < output_w; ++out_x) {
        for (uint32_t out_c = 0; out_c < output_c; ++out_c) {
          int64_t accumulator = bias_data[out_c];
          for (uint32_t filter_y = 0; filter_y < filter_h; ++filter_y) {
            const int32_t in_y = out_y * descriptor->stride_height +
                filter_y - descriptor->padding_height;
            if (in_y < 0 || in_y >= static_cast<int32_t>(input_h)) continue;
            for (uint32_t filter_x = 0; filter_x < filter_w; ++filter_x) {
              const int32_t in_x = out_x * descriptor->stride_width +
                  filter_x - descriptor->padding_width;
              if (in_x < 0 || in_x >= static_cast<int32_t>(input_w)) continue;
              for (uint32_t in_c = 0; in_c < input_c; ++in_c) {
                const size_t input_index =
                    ((batch * input_h + in_y) * input_w + in_x) * input_c + in_c;
                const size_t filter_index =
                    ((out_c * filter_h + filter_y) * filter_w + filter_x) *
                        input_c + in_c;
                accumulator +=
                    (static_cast<int32_t>(input_data[input_index]) -
                     input.zero_point) *
                    (static_cast<int32_t>(filter_data[filter_index]) -
                     filter.zero_point);
              }
            }
          }
          const int32_t bounded_accumulator = static_cast<int32_t>(
              std::max<int64_t>(INT32_MIN,
                  std::min<int64_t>(INT32_MAX, accumulator)));
          int32_t scaled = MultiplyByQuantizedMultiplier(
              bounded_accumulator, multipliers[out_c], shifts[out_c]);
          scaled += descriptor->output_zero_point;
          scaled = std::max(descriptor->activation_min,
                            std::min(descriptor->activation_max, scaled));
          const size_t output_index =
              ((batch * output_h + out_y) * output_w + out_x) * output_c + out_c;
          output_data[output_index] = static_cast<int8_t>(scaled);
        }
      }
    }
  }
  descriptor->operation_count = static_cast<uint64_t>(batches) * output_h *
      output_w * output_c * filter_h * filter_w * input_c;
  descriptor->bytes_read = input.size + filter.size + bias.size +
      UINT64_C(2) * descriptor->quantization_count * sizeof(int32_t);
  descriptor->bytes_written = output.size;
  descriptor->host_elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}

bool RunGem5HybridMatMulInt8(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  (void)extmem_size;
  if (descriptor == nullptr || descriptor->tensor_count != 3 ||
      descriptor->tensors[0].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[1].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[2].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[0].rank != 2 || descriptor->tensors[1].rank != 2 ||
      descriptor->tensors[2].rank != 2 ||
      !TensorBytesMatch(descriptor->tensors[0], sizeof(int8_t)) ||
      !TensorBytesMatch(descriptor->tensors[1], sizeof(int8_t)) ||
      !TensorBytesMatch(descriptor->tensors[2], sizeof(int8_t))) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto& lhs = descriptor->tensors[0];
  const auto& rhs = descriptor->tensors[1];
  const auto& output = descriptor->tensors[2];
  const uint32_t m = lhs.dimensions[0];
  const uint32_t k = lhs.dimensions[1];
  const uint32_t n = rhs.dimensions[1];
  if (rhs.dimensions[0] != k || output.dimensions[0] != m ||
      output.dimensions[1] != n || descriptor->activation_min < INT8_MIN ||
      descriptor->activation_max > INT8_MAX ||
      descriptor->activation_min > descriptor->activation_max) {
    return false;
  }
  const int8_t* lhs_data = TensorData<int8_t>(extmem, extmem_base, lhs);
  const int8_t* rhs_data = TensorData<int8_t>(extmem, extmem_base, rhs);
  int8_t* output_data = TensorData<int8_t>(extmem, extmem_base, output);
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t col = 0; col < n; ++col) {
      int32_t accumulator = 0;
      for (uint32_t inner = 0; inner < k; ++inner) {
        accumulator +=
            (static_cast<int32_t>(lhs_data[row * k + inner]) -
             lhs.zero_point) *
            (static_cast<int32_t>(rhs_data[inner * n + col]) -
             rhs.zero_point);
      }
      accumulator += descriptor->output_zero_point;
      accumulator = std::max(descriptor->activation_min,
                             std::min(descriptor->activation_max,
                                      accumulator));
      output_data[row * n + col] = static_cast<int8_t>(accumulator);
    }
  }
  descriptor->operation_count = static_cast<uint64_t>(m) * n * k;
  descriptor->bytes_read = lhs.size + rhs.size;
  descriptor->bytes_written = output.size;
  descriptor->host_elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}

bool RunGem5HybridFullyConnectedInt8(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  (void)extmem_size;
  if (descriptor == nullptr || descriptor->tensor_count < 3 ||
      descriptor->tensor_count > 4 ||
      descriptor->tensors[0].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[1].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[descriptor->tensor_count - 1].element_type !=
          CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[0].rank != 2 || descriptor->tensors[1].rank != 2 ||
      descriptor->tensors[descriptor->tensor_count - 1].rank != 2 ||
      !TensorBytesMatch(descriptor->tensors[0], sizeof(int8_t)) ||
      !TensorBytesMatch(descriptor->tensors[1], sizeof(int8_t)) ||
      !TensorBytesMatch(descriptor->tensors[descriptor->tensor_count - 1],
                        sizeof(int8_t))) {
    return false;
  }
  const bool has_bias = descriptor->tensor_count == 4;
  if (has_bias &&
      (descriptor->tensors[2].element_type != CORAL_OPERATOR_ELEMENT_INT32 ||
       descriptor->tensors[2].rank != 1 ||
       !TensorBytesMatch(descriptor->tensors[2], sizeof(int32_t)))) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto& input = descriptor->tensors[0];
  const auto& weights = descriptor->tensors[1];
  const auto& output = descriptor->tensors[descriptor->tensor_count - 1];
  const uint32_t batches = input.dimensions[0];
  const uint32_t input_depth = input.dimensions[1];
  const uint32_t output_depth = weights.dimensions[0];
  if (weights.dimensions[1] != input_depth ||
      output.dimensions[0] != batches ||
      output.dimensions[1] != output_depth ||
      descriptor->activation_min < INT8_MIN ||
      descriptor->activation_max > INT8_MAX ||
      descriptor->activation_min > descriptor->activation_max ||
      (has_bias && descriptor->tensors[2].dimensions[0] != output_depth)) {
    return false;
  }
  const int8_t* input_data = TensorData<int8_t>(extmem, extmem_base, input);
  const int8_t* weights_data =
      TensorData<int8_t>(extmem, extmem_base, weights);
  const int32_t* bias_data = has_bias ?
      TensorData<int32_t>(extmem, extmem_base, descriptor->tensors[2]) :
      nullptr;
  int8_t* output_data = TensorData<int8_t>(extmem, extmem_base, output);
  for (uint32_t batch = 0; batch < batches; ++batch) {
    for (uint32_t out = 0; out < output_depth; ++out) {
      int32_t accumulator = has_bias ? bias_data[out] : 0;
      for (uint32_t in = 0; in < input_depth; ++in) {
        accumulator +=
            (static_cast<int32_t>(input_data[batch * input_depth + in]) -
             input.zero_point) *
            (static_cast<int32_t>(weights_data[out * input_depth + in]) -
             weights.zero_point);
      }
      accumulator += descriptor->output_zero_point;
      accumulator = std::max(descriptor->activation_min,
                             std::min(descriptor->activation_max,
                                      accumulator));
      output_data[batch * output_depth + out] =
          static_cast<int8_t>(accumulator);
    }
  }
  descriptor->operation_count =
      static_cast<uint64_t>(batches) * output_depth * input_depth;
  descriptor->bytes_read = input.size + weights.size +
      (has_bias ? descriptor->tensors[2].size : 0);
  descriptor->bytes_written = output.size;
  descriptor->host_elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}

bool RunGem5HybridAddInt8(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  (void)extmem_size;
  if (descriptor == nullptr || descriptor->tensor_count != 3 ||
      descriptor->tensors[0].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[1].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[2].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      !SameShape(descriptor->tensors[0], descriptor->tensors[1]) ||
      !SameShape(descriptor->tensors[0], descriptor->tensors[2]) ||
      !TensorBytesMatch(descriptor->tensors[0], sizeof(int8_t)) ||
      !TensorBytesMatch(descriptor->tensors[1], sizeof(int8_t)) ||
      !TensorBytesMatch(descriptor->tensors[2], sizeof(int8_t)) ||
      descriptor->activation_min < INT8_MIN ||
      descriptor->activation_max > INT8_MAX ||
      descriptor->activation_min > descriptor->activation_max) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto& lhs = descriptor->tensors[0];
  const auto& rhs = descriptor->tensors[1];
  const auto& output = descriptor->tensors[2];
  const int8_t* lhs_data = TensorData<int8_t>(extmem, extmem_base, lhs);
  const int8_t* rhs_data = TensorData<int8_t>(extmem, extmem_base, rhs);
  int8_t* output_data = TensorData<int8_t>(extmem, extmem_base, output);
  const uint64_t count = ElementCount(output);
  for (uint64_t i = 0; i < count; ++i) {
    int32_t value =
        (static_cast<int32_t>(lhs_data[i]) - lhs.zero_point) +
        (static_cast<int32_t>(rhs_data[i]) - rhs.zero_point) +
        descriptor->output_zero_point;
    value = std::max(descriptor->activation_min,
                     std::min(descriptor->activation_max, value));
    output_data[i] = static_cast<int8_t>(value);
  }
  descriptor->operation_count = count;
  descriptor->bytes_read = lhs.size + rhs.size;
  descriptor->bytes_written = output.size;
  descriptor->host_elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}

bool RunGem5HybridSoftmax(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  (void)extmem_size;
  if (descriptor == nullptr || descriptor->tensor_count != 2 ||
      descriptor->tensors[0].rank == 0 ||
      !SameShape(descriptor->tensors[0], descriptor->tensors[1])) {
    return false;
  }
  const auto& input = descriptor->tensors[0];
  const auto& output = descriptor->tensors[1];
  const uint32_t depth = input.dimensions[input.rank - 1];
  const uint64_t total = ElementCount(input);
  if (depth == 0 || total == 0 || total % depth != 0) return false;
  const uint64_t rows = total / depth;
  const auto start = std::chrono::steady_clock::now();
  if (input.element_type == CORAL_OPERATOR_ELEMENT_FLOAT32 &&
      output.element_type == CORAL_OPERATOR_ELEMENT_FLOAT32 &&
      TensorBytesMatch(input, sizeof(float)) &&
      TensorBytesMatch(output, sizeof(float))) {
    const float* input_data = TensorFloatData(extmem, extmem_base, input);
    float* output_data = MutableTensorFloatData(extmem, extmem_base, output);
    for (uint64_t row = 0; row < rows; ++row) {
      const float* row_input = input_data + row * depth;
      float* row_output = output_data + row * depth;
      float max_value = row_input[0];
      for (uint32_t i = 1; i < depth; ++i) {
        max_value = std::max(max_value, row_input[i]);
      }
      float sum = 0.0f;
      for (uint32_t i = 0; i < depth; ++i) {
        row_output[i] = std::exp(row_input[i] - max_value);
        sum += row_output[i];
      }
      if (sum == 0.0f || !std::isfinite(sum)) return false;
      for (uint32_t i = 0; i < depth; ++i) {
        row_output[i] /= sum;
      }
    }
  } else if (input.element_type == CORAL_OPERATOR_ELEMENT_INT8 &&
             output.element_type == CORAL_OPERATOR_ELEMENT_INT8 &&
             TensorBytesMatch(input, sizeof(int8_t)) &&
             TensorBytesMatch(output, sizeof(int8_t))) {
    const int8_t* input_data = TensorData<int8_t>(extmem, extmem_base, input);
    int8_t* output_data = TensorData<int8_t>(extmem, extmem_base, output);
    for (uint64_t row = 0; row < rows; ++row) {
      float max_value = static_cast<float>(input_data[row * depth]);
      for (uint32_t i = 1; i < depth; ++i) {
        max_value =
            std::max(max_value, static_cast<float>(input_data[row * depth + i]));
      }
      float sum = 0.0f;
      for (uint32_t i = 0; i < depth; ++i) {
        sum += std::exp(static_cast<float>(input_data[row * depth + i]) -
                        max_value);
      }
      if (sum == 0.0f || !std::isfinite(sum)) return false;
      for (uint32_t i = 0; i < depth; ++i) {
        const float probability =
            std::exp(static_cast<float>(input_data[row * depth + i]) -
                     max_value) / sum;
        int32_t quantized = static_cast<int32_t>(
            std::lround(probability * 255.0f)) - 128;
        quantized = std::max<int32_t>(INT8_MIN,
                                      std::min<int32_t>(INT8_MAX, quantized));
        output_data[row * depth + i] = static_cast<int8_t>(quantized);
      }
    }
  } else {
    return false;
  }
  descriptor->operation_count = total;
  descriptor->bytes_read = input.size;
  descriptor->bytes_written = output.size;
  descriptor->host_elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}

bool RunGem5HybridLayerNorm(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  (void)extmem_size;
  if (descriptor == nullptr || descriptor->tensor_count != 4 ||
      descriptor->tensors[0].element_type != CORAL_OPERATOR_ELEMENT_FLOAT32 ||
      descriptor->tensors[1].element_type != CORAL_OPERATOR_ELEMENT_FLOAT32 ||
      descriptor->tensors[2].element_type != CORAL_OPERATOR_ELEMENT_FLOAT32 ||
      descriptor->tensors[3].element_type != CORAL_OPERATOR_ELEMENT_FLOAT32 ||
      descriptor->tensors[0].rank == 0 ||
      !SameShape(descriptor->tensors[0], descriptor->tensors[3]) ||
      descriptor->tensors[1].rank != 1 || descriptor->tensors[2].rank != 1 ||
      !TensorBytesMatch(descriptor->tensors[0], sizeof(float)) ||
      !TensorBytesMatch(descriptor->tensors[1], sizeof(float)) ||
      !TensorBytesMatch(descriptor->tensors[2], sizeof(float)) ||
      !TensorBytesMatch(descriptor->tensors[3], sizeof(float))) {
    return false;
  }
  const auto& input = descriptor->tensors[0];
  const auto& scale = descriptor->tensors[1];
  const auto& bias = descriptor->tensors[2];
  const auto& output = descriptor->tensors[3];
  const uint32_t depth = input.dimensions[input.rank - 1];
  const uint64_t total = ElementCount(input);
  if (depth == 0 || total == 0 || total % depth != 0 ||
      scale.dimensions[0] != depth || bias.dimensions[0] != depth) {
    return false;
  }
  float epsilon = 1.0e-5f;
  if (descriptor->reserved[0] != 0 &&
      (!FloatBits(descriptor->reserved[0], &epsilon) || epsilon <= 0.0f)) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  const float* input_data = TensorFloatData(extmem, extmem_base, input);
  const float* scale_data = TensorFloatData(extmem, extmem_base, scale);
  const float* bias_data = TensorFloatData(extmem, extmem_base, bias);
  float* output_data = MutableTensorFloatData(extmem, extmem_base, output);
  const uint64_t rows = total / depth;
  for (uint64_t row = 0; row < rows; ++row) {
    const float* row_input = input_data + row * depth;
    float mean = 0.0f;
    for (uint32_t i = 0; i < depth; ++i) mean += row_input[i];
    mean /= static_cast<float>(depth);
    float variance = 0.0f;
    for (uint32_t i = 0; i < depth; ++i) {
      const float centered = row_input[i] - mean;
      variance += centered * centered;
    }
    variance /= static_cast<float>(depth);
    const float inv_stddev = 1.0f / std::sqrt(variance + epsilon);
    for (uint32_t i = 0; i < depth; ++i) {
      output_data[row * depth + i] =
          (row_input[i] - mean) * inv_stddev * scale_data[i] + bias_data[i];
    }
  }
  descriptor->operation_count = total * 5;
  descriptor->bytes_read = input.size + scale.size + bias.size;
  descriptor->bytes_written = output.size;
  descriptor->host_elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}

bool RunGem5HybridDepthwiseConv2D(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  (void)extmem_size;
  if (!CommonConvValid(*descriptor, true)) {
    return false;
  }
  const auto start = std::chrono::steady_clock::now();
  const auto& input = descriptor->tensors[0];
  const auto& filter = descriptor->tensors[1];
  const auto& bias = descriptor->tensors[2];
  const auto& output = descriptor->tensors[3];
  const int8_t* input_data = TensorData<int8_t>(extmem, extmem_base, input);
  const int8_t* filter_data = TensorData<int8_t>(extmem, extmem_base, filter);
  const int32_t* bias_data = TensorData<int32_t>(extmem, extmem_base, bias);
  int8_t* output_data = TensorData<int8_t>(extmem, extmem_base, output);
  const int32_t* multipliers = reinterpret_cast<const int32_t*>(
      extmem + descriptor->multiplier_address - extmem_base);
  const int32_t* shifts = reinterpret_cast<const int32_t*>(
      extmem + descriptor->shift_address - extmem_base);
  const uint32_t batches = input.dimensions[0];
  const uint32_t input_h = input.dimensions[1];
  const uint32_t input_w = input.dimensions[2];
  const uint32_t input_c = input.dimensions[3];
  const uint32_t output_h = output.dimensions[1];
  const uint32_t output_w = output.dimensions[2];
  const uint32_t output_c = output.dimensions[3];
  const uint32_t filter_h = filter.dimensions[1];
  const uint32_t filter_w = filter.dimensions[2];
  const uint32_t depth_multiplier = output_c / input_c;
  for (uint32_t i = 0; i < output_c; ++i) {
    if (shifts[i] < -31 || shifts[i] > 30) return false;
  }

  for (uint32_t batch = 0; batch < batches; ++batch) {
    for (uint32_t out_y = 0; out_y < output_h; ++out_y) {
      for (uint32_t out_x = 0; out_x < output_w; ++out_x) {
        for (uint32_t out_c = 0; out_c < output_c; ++out_c) {
          const uint32_t in_c = out_c / depth_multiplier;
          int64_t accumulator = bias_data[out_c];
          for (uint32_t filter_y = 0; filter_y < filter_h; ++filter_y) {
            const int32_t in_y = out_y * descriptor->stride_height +
                filter_y - descriptor->padding_height;
            if (in_y < 0 || in_y >= static_cast<int32_t>(input_h)) continue;
            for (uint32_t filter_x = 0; filter_x < filter_w; ++filter_x) {
              const int32_t in_x = out_x * descriptor->stride_width +
                  filter_x - descriptor->padding_width;
              if (in_x < 0 || in_x >= static_cast<int32_t>(input_w)) continue;
              const size_t input_index =
                  ((batch * input_h + in_y) * input_w + in_x) * input_c + in_c;
              const size_t filter_index =
                  (filter_y * filter_w + filter_x) * output_c + out_c;
              accumulator +=
                  (static_cast<int32_t>(input_data[input_index]) -
                   input.zero_point) *
                  (static_cast<int32_t>(filter_data[filter_index]) -
                   filter.zero_point);
            }
          }
          const int32_t bounded_accumulator = static_cast<int32_t>(
              std::max<int64_t>(INT32_MIN,
                  std::min<int64_t>(INT32_MAX, accumulator)));
          int32_t scaled = MultiplyByQuantizedMultiplier(
              bounded_accumulator, multipliers[out_c], shifts[out_c]);
          scaled += descriptor->output_zero_point;
          scaled = std::max(descriptor->activation_min,
                            std::min(descriptor->activation_max, scaled));
          const size_t output_index =
              ((batch * output_h + out_y) * output_w + out_x) * output_c + out_c;
          output_data[output_index] = static_cast<int8_t>(scaled);
        }
      }
    }
  }
  descriptor->operation_count = static_cast<uint64_t>(batches) * output_h *
      output_w * output_c * filter_h * filter_w;
  descriptor->bytes_read = input.size + filter.size + bias.size +
      UINT64_C(2) * descriptor->quantization_count * sizeof(int32_t);
  descriptor->bytes_written = output.size;
  descriptor->host_elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}
