#include "hw_sim/gem5_bridge/gem5_hybrid_kernels.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>

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
