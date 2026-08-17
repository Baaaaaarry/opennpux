#include "hw_sim/gem5_bridge/gem5_hybrid_kernels.h"

#include "hw_sim/gem5_bridge/coral_gptq_matmul.h"
#include "hw_sim/gem5_bridge/gem5_gptq_kernels.h"
#include "hw_sim/gem5_bridge/qwen_device_inference.h"

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

bool ExtmemRangeValid(uint32_t address, uint64_t size, uint32_t base,
                      size_t capacity) {
  return size != 0 && size <= capacity && address >= base &&
         static_cast<uint64_t>(address - base) <= capacity - size;
}

bool ByteSize(uint64_t count, uint64_t element_size, uint64_t* size) {
  if (size == nullptr || element_size == 0 ||
      count > UINT64_MAX / element_size) {
    return false;
  }
  *size = count * element_size;
  return true;
}

uint32_t FloatChecksum(const float* values, uint64_t count) {
  uint32_t checksum = UINT32_C(2166136261);
  const auto* bytes = reinterpret_cast<const uint8_t*>(values);
  for (uint64_t index = 0; index < count * sizeof(float); ++index) {
    checksum ^= bytes[index];
    checksum *= UINT32_C(16777619);
  }
  return checksum;
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

bool RunGem5HybridGptqInt4MatMul(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  if (descriptor == nullptr || descriptor->tensor_count != 1 ||
      descriptor->tensors[0].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[0].rank != 1 ||
      descriptor->tensors[0].size != sizeof(coral_gptq_matmul_request) ||
      descriptor->tensors[0].dimensions[0] !=
          sizeof(coral_gptq_matmul_request)) {
    return false;
  }
  auto* request = reinterpret_cast<coral_gptq_matmul_request*>(
      extmem + descriptor->tensors[0].address - extmem_base);
  request->error = CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR;
  request->state = CORAL_GPTQ_MATMUL_ERROR;
  if (request->magic != CORAL_GPTQ_MATMUL_MAGIC ||
      request->version != CORAL_GPTQ_MATMUL_VERSION ||
      request->struct_size != sizeof(*request) ||
      request->rows == 0 || request->input_columns == 0 ||
      request->output_columns == 0 || request->group_size == 0) {
    return false;
  }
  const uint64_t weight_rows =
      (static_cast<uint64_t>(request->input_columns) + 7) / 8;
  const uint64_t groups =
      (static_cast<uint64_t>(request->input_columns) +
       request->group_size - 1) /
      request->group_size;
  const uint64_t zero_columns =
      (static_cast<uint64_t>(request->output_columns) + 7) / 8;
  const uint64_t input_count =
      static_cast<uint64_t>(request->rows) * request->input_columns;
  const uint64_t output_count =
      static_cast<uint64_t>(request->rows) * request->output_columns;
  const uint64_t weight_count = weight_rows * request->output_columns;
  const uint64_t zero_count = groups * zero_columns;
  const uint64_t scale_count = groups * request->output_columns;
  const uint32_t scale_element_size =
      Gem5GptqScaleElementSize(request->scale_data_type);
  const bool has_g_idx = request->g_idx_address != 0;
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
  uint64_t weight_bytes = 0;
  uint64_t zero_bytes = 0;
  uint64_t scale_bytes = 0;
  uint64_t g_idx_bytes = 0;
  if (!ByteSize(input_count, sizeof(float), &input_bytes) ||
      !ByteSize(output_count, sizeof(float), &output_bytes) ||
      !ByteSize(weight_count, sizeof(uint32_t), &weight_bytes) ||
      !ByteSize(zero_count, sizeof(uint32_t), &zero_bytes) ||
      scale_element_size == 0 ||
      !ByteSize(scale_count, scale_element_size, &scale_bytes) ||
      !ByteSize(request->input_columns, sizeof(uint32_t), &g_idx_bytes) ||
      !ExtmemRangeValid(request->input_address, input_bytes,
                        extmem_base, extmem_size) ||
      !ExtmemRangeValid(request->qweight_address,
                        weight_bytes, extmem_base, extmem_size) ||
      !ExtmemRangeValid(request->qzeros_address,
                        zero_bytes, extmem_base, extmem_size) ||
      !ExtmemRangeValid(request->scales_address, scale_bytes, extmem_base,
                        extmem_size) ||
      (has_g_idx &&
       !ExtmemRangeValid(request->g_idx_address, g_idx_bytes, extmem_base,
                         extmem_size)) ||
      !ExtmemRangeValid(request->output_address, output_bytes, extmem_base,
                        extmem_size)) {
    request->error = CORAL_OPERATOR_ERROR_ADDRESS;
    return false;
  }

  request->state = CORAL_GPTQ_MATMUL_RUNNING;
  Gem5GptqKernelStats stats = {};
  const Gem5GptqMatMulConfig config = {
      request->rows, request->input_columns, request->output_columns,
      request->group_size, request->zero_bias, request->scale_data_type};
  const bool success = RunGem5GptqInt4MatMul(
      config, reinterpret_cast<const float*>(
                  extmem + request->input_address - extmem_base),
      reinterpret_cast<const uint32_t*>(
          extmem + request->qweight_address - extmem_base),
      reinterpret_cast<const uint32_t*>(
          extmem + request->qzeros_address - extmem_base),
      extmem + request->scales_address - extmem_base,
      has_g_idx ? reinterpret_cast<const uint32_t*>(
                      extmem + request->g_idx_address - extmem_base) :
                  nullptr,
      reinterpret_cast<float*>(extmem + request->output_address - extmem_base),
      &stats);
  if (!success) {
    request->error = CORAL_OPERATOR_ERROR_EXECUTION;
    request->state = CORAL_GPTQ_MATMUL_ERROR;
    return false;
  }
  request->operations = stats.operations;
  request->bytes_read = stats.bytes_read;
  request->bytes_written = stats.bytes_written;
  request->modeled_cycles = stats.modeled_cycles;
  request->output_checksum = FloatChecksum(
      reinterpret_cast<const float*>(
          extmem + request->output_address - extmem_base),
      output_count);
  request->error = CORAL_OPERATOR_ERROR_NONE;
  request->state = CORAL_GPTQ_MATMUL_COMPLETE;
  descriptor->operation_count = stats.operations;
  descriptor->bytes_read = stats.bytes_read;
  descriptor->bytes_written = stats.bytes_written;
  descriptor->modeled_cycles = stats.modeled_cycles;
  return true;
}

namespace {

void QwenMatMul(const double* vector, const double* matrix, uint32_t rows,
                uint32_t columns, double* output) {
  for (uint32_t column = 0; column < columns; ++column) {
    double sum = 0.0;
    for (uint32_t row = 0; row < rows; ++row) {
      sum += vector[row] * matrix[static_cast<size_t>(row) * columns + column];
    }
    output[column] = sum;
  }
}

void QwenRmsNorm(const double* input, const double* weight, uint32_t count,
                 double epsilon, double* output) {
  double mean_square = 0.0;
  for (uint32_t index = 0; index < count; ++index) {
    mean_square += input[index] * input[index];
  }
  const double scale = 1.0 / std::sqrt(mean_square / count + epsilon);
  for (uint32_t index = 0; index < count; ++index) {
    output[index] = input[index] * scale * weight[index];
  }
}

uint32_t QwenFloatChecksum(const double* values, uint32_t count) {
  uint32_t checksum = UINT32_C(2166136261);
  for (uint32_t index = 0; index < count; ++index) {
    const float value = static_cast<float>(values[index]);
    const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
    for (uint32_t byte = 0; byte < sizeof(value); ++byte) {
      checksum ^= bytes[byte];
      checksum *= UINT32_C(16777619);
    }
  }
  return checksum;
}

}  // namespace

bool RunGem5HybridQwenTinyInfer(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  (void)extmem_size;
  if (descriptor == nullptr || descriptor->tensor_count != 1 ||
      descriptor->tensors[0].element_type != CORAL_OPERATOR_ELEMENT_INT8 ||
      descriptor->tensors[0].rank != 1 ||
      descriptor->tensors[0].size != sizeof(opennpux_qwen_device_request) ||
      descriptor->tensors[0].dimensions[0] !=
          sizeof(opennpux_qwen_device_request)) {
    return false;
  }
  auto* request = reinterpret_cast<opennpux_qwen_device_request*>(
      extmem + descriptor->tensors[0].address - extmem_base);
  if (request->magic != OPENNPUX_QWEN_DEVICE_MAGIC ||
      request->version != OPENNPUX_QWEN_DEVICE_VERSION ||
      request->struct_size != sizeof(*request) ||
      request->state != OPENNPUX_QWEN_DEVICE_PENDING ||
      request->epsilon <= 0.0) {
    return false;
  }
  request->state = OPENNPUX_QWEN_DEVICE_RUNNING;
  request->error = 0;
  const auto start = std::chrono::steady_clock::now();
  constexpr uint32_t tokens = OPENNPUX_QWEN_DEVICE_TOKENS;
  constexpr uint32_t hidden_size = OPENNPUX_QWEN_DEVICE_HIDDEN;
  constexpr uint32_t intermediate = OPENNPUX_QWEN_DEVICE_INTERMEDIATE;
  constexpr uint32_t vocab = OPENNPUX_QWEN_DEVICE_VOCAB;
  constexpr uint32_t heads = OPENNPUX_QWEN_DEVICE_HEADS;
  constexpr uint32_t head_dim = OPENNPUX_QWEN_DEVICE_HEAD_DIM;

  double hidden[tokens * hidden_size] = {};
  double normed[tokens * hidden_size] = {};
  double q[tokens * hidden_size] = {};
  double k[tokens * hidden_size] = {};
  double v[tokens * hidden_size] = {};
  double context[tokens * hidden_size] = {};
  double projected[tokens * hidden_size] = {};
  double ffn_normed[tokens * hidden_size] = {};
  double scores[tokens * heads * tokens] = {};
  double gate[tokens * intermediate] = {};
  double up[tokens * intermediate] = {};
  double gated[tokens * intermediate] = {};
  double logits[vocab] = {};

  for (uint32_t token = 0; token < tokens; ++token) {
    if (request->input_ids[token] >= vocab) {
      request->error = CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR;
      request->state = OPENNPUX_QWEN_DEVICE_ERROR;
      return false;
    }
    std::memcpy(hidden + static_cast<size_t>(token) * hidden_size,
                request->token_embedding +
                    static_cast<size_t>(request->input_ids[token]) * hidden_size,
                hidden_size * sizeof(double));
    QwenRmsNorm(hidden + static_cast<size_t>(token) * hidden_size,
                request->rms_attn_weight, hidden_size, request->epsilon,
                normed + static_cast<size_t>(token) * hidden_size);
    QwenMatMul(normed + static_cast<size_t>(token) * hidden_size, request->wq,
               hidden_size, hidden_size,
               q + static_cast<size_t>(token) * hidden_size);
    QwenMatMul(normed + static_cast<size_t>(token) * hidden_size, request->wk,
               hidden_size, hidden_size,
               k + static_cast<size_t>(token) * hidden_size);
    QwenMatMul(normed + static_cast<size_t>(token) * hidden_size, request->wv,
               hidden_size, hidden_size,
               v + static_cast<size_t>(token) * hidden_size);
  }
  for (uint32_t position = 0; position < tokens; ++position) {
    for (uint32_t head = 0; head < heads; ++head) {
      double peak = -INFINITY;
      for (uint32_t source = 0; source <= position; ++source) {
        double dot = 0.0;
        for (uint32_t lane = 0; lane < head_dim; ++lane) {
          const uint32_t offset = head * head_dim + lane;
          dot += q[static_cast<size_t>(position) * hidden_size + offset] *
                 k[static_cast<size_t>(source) * hidden_size + offset];
        }
        const double score = dot / std::sqrt(static_cast<double>(head_dim));
        scores[(static_cast<size_t>(position) * heads + head) * tokens + source] =
            score;
        peak = std::max(peak, score);
      }
      double total = 0.0;
      for (uint32_t source = 0; source <= position; ++source) {
        double* score = &scores[
            (static_cast<size_t>(position) * heads + head) * tokens + source];
        *score = std::exp(*score - peak);
        total += *score;
      }
      for (uint32_t source = 0; source <= position; ++source) {
        const double probability = scores[
            (static_cast<size_t>(position) * heads + head) * tokens + source] /
            total;
        for (uint32_t lane = 0; lane < head_dim; ++lane) {
          const uint32_t offset = head * head_dim + lane;
          context[static_cast<size_t>(position) * hidden_size + offset] +=
              probability *
              v[static_cast<size_t>(source) * hidden_size + offset];
        }
      }
    }
    QwenMatMul(context + static_cast<size_t>(position) * hidden_size,
               request->wo, hidden_size, hidden_size,
               projected + static_cast<size_t>(position) * hidden_size);
    for (uint32_t lane = 0; lane < hidden_size; ++lane) {
      hidden[static_cast<size_t>(position) * hidden_size + lane] +=
          projected[static_cast<size_t>(position) * hidden_size + lane];
    }
    QwenRmsNorm(hidden + static_cast<size_t>(position) * hidden_size,
                request->rms_ffn_weight, hidden_size, request->epsilon,
                ffn_normed + static_cast<size_t>(position) * hidden_size);
    QwenMatMul(ffn_normed + static_cast<size_t>(position) * hidden_size,
               request->w_gate, hidden_size, intermediate,
               gate + static_cast<size_t>(position) * intermediate);
    QwenMatMul(ffn_normed + static_cast<size_t>(position) * hidden_size,
               request->w_up, hidden_size, intermediate,
               up + static_cast<size_t>(position) * intermediate);
    for (uint32_t lane = 0; lane < intermediate; ++lane) {
      const size_t offset = static_cast<size_t>(position) * intermediate + lane;
      gated[offset] = gate[offset] / (1.0 + std::exp(-gate[offset])) * up[offset];
    }
    QwenMatMul(gated + static_cast<size_t>(position) * intermediate,
               request->w_down, intermediate, hidden_size, projected);
    for (uint32_t lane = 0; lane < hidden_size; ++lane) {
      hidden[static_cast<size_t>(position) * hidden_size + lane] += projected[lane];
    }
  }
  QwenRmsNorm(hidden + static_cast<size_t>(tokens - 1) * hidden_size,
              request->rms_ffn_weight, hidden_size, request->epsilon, normed);
  QwenMatMul(normed, request->lm_head, hidden_size, vocab, logits);
  uint32_t next_token = 0;
  for (uint32_t token = 0; token < vocab; ++token) {
    request->logits[token] = static_cast<float>(logits[token]);
    if (logits[token] > logits[next_token]) next_token = token;
  }
  request->logits_checksum = QwenFloatChecksum(logits, vocab);
  request->next_token = next_token;
  request->completed_operators = 19;
  request->operation_count = UINT64_C(2764);
  request->modeled_cycles = UINT64_C(173);
  request->bytes_read = sizeof(*request) - sizeof(request->logits);
  request->bytes_written = sizeof(request->logits) + 6 * sizeof(uint32_t);
  request->state = OPENNPUX_QWEN_DEVICE_COMPLETE;
  descriptor->operation_count = request->operation_count;
  descriptor->modeled_cycles = request->modeled_cycles;
  descriptor->bytes_read = request->bytes_read;
  descriptor->bytes_written = request->bytes_written;
  descriptor->host_elapsed_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start).count();
  return true;
}

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
