#include "hw_sim/gem5_bridge/gem5_host_functional_backend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "hw_sim/gem5_bridge/npu_submission.h"

namespace {

bool ElementCount(const Gem5HostFunctionalRequest& request, size_t* count) {
  if (count == nullptr || request.rows == 0 || request.features == 0 ||
      request.features > std::numeric_limits<size_t>::max() / request.rows) {
    return false;
  }
  *count = request.rows * request.features;
  return true;
}

Gem5HostFunctionalResult Result(Gem5HostFunctionalStatus status) {
  return {status, {}};
}

void CopyStats(const Gem5GptqKernelStats& source,
               Gem5TransformerKernelStats* destination) {
  destination->operations = source.operations;
  destination->bytes_read = source.bytes_read;
  destination->bytes_written = source.bytes_written;
  destination->modeled_cycles = source.modeled_cycles;
}

void AddStats(const Gem5GptqKernelStats& source,
               Gem5TransformerKernelStats* destination) {
  destination->operations += source.operations;
  destination->bytes_read += source.bytes_read;
  destination->bytes_written += source.bytes_written;
  destination->modeled_cycles += source.modeled_cycles;
}

void AddStats(const Gem5TransformerKernelStats& source,
              Gem5TransformerKernelStats* destination) {
  destination->operations += source.operations;
  destination->bytes_read += source.bytes_read;
  destination->bytes_written += source.bytes_written;
  destination->modeled_cycles += source.modeled_cycles;
}

bool RunTopKRows(const Gem5HostFunctionalRequest& request,
                 Gem5TransformerKernelStats* stats) {
  if (request.input == nullptr || request.output_indices == nullptr ||
      request.rows == 0 || request.features == 0 || request.top_k == 0 ||
      request.top_k > request.features ||
      (request.output_indices_count != request.top_k &&
       request.output_indices_count != request.rows * request.top_k) ||
      stats == nullptr) {
    return false;
  }
  const bool last_row_only = request.output_indices_count == request.top_k;
  const size_t output_rows = last_row_only ? 1 : request.rows;
  std::vector<float> temporary;
  float* values = request.output;
  if (values == nullptr) {
    try {
      temporary.resize(output_rows * request.top_k);
    } catch (...) {
      return false;
    }
    values = temporary.data();
  }
  for (size_t output_row = 0; output_row < output_rows; ++output_row) {
    const size_t input_row = last_row_only ? request.rows - 1 : output_row;
    Gem5TransformerKernelStats row_stats = {};
    if (!RunGem5TopKF32(
            request.input + input_row * request.features, request.features,
            request.top_k, values + output_row * request.top_k,
            request.output_indices + output_row * request.top_k, &row_stats)) {
      return false;
    }
    stats->operations += row_stats.operations;
    stats->bytes_read += row_stats.bytes_read;
    stats->bytes_written += row_stats.bytes_written;
    stats->modeled_cycles += row_stats.modeled_cycles;
  }
  return true;
}

bool RunRouter(const Gem5HostFunctionalRequest& request,
               Gem5TransformerKernelStats* stats) {
  if (stats == nullptr || request.operator_parameters == nullptr ||
      request.rows == 0 || request.rows > UINT32_MAX || request.top_k == 0 ||
      request.output == nullptr || request.output_indices == nullptr) {
    return false;
  }
  auto parameters = *request.operator_parameters;
  const uint32_t expert_count = parameters.output_features;
  if (expert_count == 0 || request.top_k > expert_count ||
      request.rows > std::numeric_limits<size_t>::max() / expert_count) {
    return false;
  }
  std::vector<float> logits;
  try {
    logits.resize(request.rows * expert_count);
  } catch (...) {
    return false;
  }
  if (request.gptq_operands != nullptr &&
      request.gptq_operands->qweight.data != nullptr) {
    parameters.opcode = OPENNPUX_NPU_OP_MATMUL;
    auto projection = *request.gptq_operands;
    projection.output = {logits.data(), logits.size() * sizeof(float)};
    Gem5GptqKernelStats projection_stats = {};
    if (!RunGem5GenericGptqMatMul(
            parameters, static_cast<uint32_t>(request.rows), projection,
            &projection_stats)) {
      return false;
    }
    AddStats(projection_stats, stats);
  } else {
    Gem5TransformerKernelStats projection_stats = {};
    if (!RunGem5MatMulF32(
            request.input, request.weight, request.rows,
            parameters.input_features, expert_count, logits.data(),
            &projection_stats)) {
      return false;
    }
    stats->operations += projection_stats.operations;
    stats->bytes_read += projection_stats.bytes_read;
    stats->bytes_written += projection_stats.bytes_written;
    stats->modeled_cycles += projection_stats.modeled_cycles;
  }
  for (size_t row = 0; row < request.rows; ++row) {
    Gem5TransformerKernelStats topk_stats = {};
    float* values = request.output + row * request.top_k;
    if (!RunGem5TopKF32(logits.data() + row * expert_count, expert_count,
                        request.top_k, values,
                        request.output_indices + row * request.top_k,
                        &topk_stats)) {
      return false;
    }
    const float maximum = values[0];
    double sum = 0.0;
    for (size_t index = 0; index < request.top_k; ++index) {
      values[index] = std::exp(values[index] - maximum);
      sum += values[index];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
      return false;
    }
    for (size_t index = 0; index < request.top_k; ++index) {
      values[index] /= static_cast<float>(sum);
    }
    stats->operations += topk_stats.operations + request.top_k * 2;
    stats->bytes_read += topk_stats.bytes_read;
    stats->bytes_written += topk_stats.bytes_written;
    stats->modeled_cycles += topk_stats.modeled_cycles;
  }
  return true;
}

}  // namespace

bool Gem5HostFunctionalBackend::Supports(uint32_t opcode) const {
  switch (opcode) {
    case OPENNPUX_NPU_OP_EMBED:
    case OPENNPUX_NPU_OP_MATMUL:
    case OPENNPUX_NPU_OP_ADD:
    case OPENNPUX_NPU_OP_MUL:
    case OPENNPUX_NPU_OP_NORMALIZE:
    case OPENNPUX_NPU_OP_ROPE:
    case OPENNPUX_NPU_OP_SOFTMAX:
    case OPENNPUX_NPU_OP_TOPK:
    case OPENNPUX_NPU_OP_ROUTER:
    case OPENNPUX_NPU_OP_ACTIVATION:
    case OPENNPUX_NPU_OP_EXPERT:
    case OPENNPUX_NPU_OP_DMA:
    case OPENNPUX_NPU_OP_ATTENTION:
    case OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION:
    case OPENNPUX_NPU_OP_RECURRENT_UPDATE:
    case OPENNPUX_NPU_OP_COMBINE:
      return true;
    default:
      return false;
  }
}

Gem5HostFunctionalResult Gem5HostFunctionalBackend::Execute(
    const Gem5HostFunctionalRequest& request) const {
  if (!Supports(request.opcode)) {
    return Result(Gem5HostFunctionalStatus::kUnsupported);
  }
  Gem5HostFunctionalResult result = Result(Gem5HostFunctionalStatus::kComplete);
  bool success = false;
  const bool linear_gate_norm =
      request.opcode == OPENNPUX_NPU_OP_NORMALIZE &&
      request.linear_gate_weight.data != nullptr;
  if (linear_gate_norm) {
    if (request.operator_parameters == nullptr || request.input == nullptr ||
        request.secondary == nullptr || request.output == nullptr ||
        request.linear_norm_weight.data == nullptr) {
      return Result(Gem5HostFunctionalStatus::kInvalid);
    }
    const size_t input_features = request.operator_parameters->input_features;
    const size_t output_features = request.operator_parameters->output_features;
    const size_t head_dim = request.operator_parameters->intermediate_features;
    if (input_features == 0 || output_features == 0 || head_dim == 0 ||
        output_features % head_dim != 0 ||
        request.linear_gate_weight.size !=
            input_features * output_features * sizeof(float) ||
        request.linear_norm_weight.size != head_dim * sizeof(float)) {
      return Result(Gem5HostFunctionalStatus::kInvalid);
    }
    std::vector<float> gate(request.rows * output_features);
    Gem5TransformerKernelStats projection_stats = {};
    Gem5TransformerKernelStats norm_stats = {};
    if (!RunGem5MatMulF32(
            request.secondary,
            static_cast<const float*>(request.linear_gate_weight.data),
            request.rows, input_features, output_features, gate.data(),
            &projection_stats) ||
        !RunGem5GatedRmsNormF32(
            request.input, gate.data(),
            static_cast<const float*>(request.linear_norm_weight.data),
            request.rows, output_features / head_dim, head_dim,
            request.epsilon, request.output, &norm_stats,
            (request.operator_parameters->flags &
             OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE) != 0)) {
      return Result(Gem5HostFunctionalStatus::kExecutionError);
    }
    result.stats.operations =
        projection_stats.operations + norm_stats.operations;
    result.stats.bytes_read =
        projection_stats.bytes_read + norm_stats.bytes_read;
    result.stats.bytes_written =
        projection_stats.bytes_written + norm_stats.bytes_written;
    result.stats.modeled_cycles =
        projection_stats.modeled_cycles + norm_stats.modeled_cycles;
    return result;
  }
  if (request.opcode == OPENNPUX_NPU_OP_MATMUL ||
      request.opcode == OPENNPUX_NPU_OP_EXPERT) {
    if (request.operator_parameters == nullptr || request.rows == 0 ||
        request.rows > UINT32_MAX) {
      return Result(Gem5HostFunctionalStatus::kInvalid);
    }
    Gem5GptqKernelStats gptq_stats = {};
    const bool float_qkv =
        request.opcode == OPENNPUX_NPU_OP_MATMUL &&
        request.attention_q_weight.data != nullptr;
    if (float_qkv) {
      const size_t input_features =
          request.operator_parameters->input_features;
      size_t query_features = 0;
      size_t key_features = 0;
      if (request.input == nullptr || request.output == nullptr ||
          request.output_secondary == nullptr ||
          request.output_tertiary == nullptr ||
          request.attention_k_weight.data == nullptr ||
          request.attention_v_weight.data == nullptr ||
          request.attention_q_norm_weight.data == nullptr ||
          request.attention_k_norm_weight.data == nullptr ||
          input_features == 0 || request.heads == 0 || request.kv_heads == 0 ||
          request.head_dim == 0 || request.heads > SIZE_MAX / request.head_dim ||
          request.kv_heads > SIZE_MAX / request.head_dim) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      query_features = request.heads * request.head_dim;
      key_features = request.kv_heads * request.head_dim;
      if (request.features != query_features ||
          input_features > SIZE_MAX / key_features ||
          request.attention_q_weight.size % sizeof(float) != 0 ||
          request.attention_q_weight.size / sizeof(float) % input_features !=
              0 ||
          request.attention_k_weight.size !=
              input_features * key_features * sizeof(float) ||
          request.attention_v_weight.size !=
              input_features * key_features * sizeof(float) ||
          request.attention_q_norm_weight.size !=
              request.head_dim * sizeof(float) ||
          request.attention_k_norm_weight.size !=
              request.head_dim * sizeof(float)) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      const size_t q_weight_outputs =
          request.attention_q_weight.size / sizeof(float) / input_features;
      if (!RunGem5FloatQkvF32(
              request.input,
              static_cast<const float*>(request.attention_q_weight.data),
              static_cast<const float*>(request.attention_k_weight.data),
              static_cast<const float*>(request.attention_v_weight.data),
              static_cast<const float*>(
                  request.attention_q_norm_weight.data),
              static_cast<const float*>(
                  request.attention_k_norm_weight.data),
              request.rows, input_features, request.heads, request.kv_heads,
              request.head_dim, q_weight_outputs, request.epsilon,
              (request.operator_parameters->flags &
               OPENNPUX_NPU_PARAMETER_NORM_WEIGHT_OFFSET) != 0,
              request.output, request.output_secondary,
              request.output_tertiary, request.output_quaternary,
              &result.stats,
              (request.operator_parameters->flags &
               OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE) != 0)) {
        result.status = Gem5HostFunctionalStatus::kExecutionError;
      }
      return result;
    }
    const bool float_shared_expert =
        request.opcode == OPENNPUX_NPU_OP_EXPERT &&
        request.shared_gate_weight.data != nullptr;
    if (float_shared_expert) {
      const size_t input_features =
          request.operator_parameters->input_features;
      const size_t intermediate_features =
          request.operator_parameters->intermediate_features;
      const size_t output_features =
          request.operator_parameters->output_features;
      if (request.input == nullptr || request.output == nullptr ||
          request.shared_up_weight.data == nullptr ||
          request.shared_down_weight.data == nullptr ||
          request.shared_router_weight.data == nullptr ||
          input_features == 0 || intermediate_features == 0 ||
          output_features == 0 ||
          input_features > SIZE_MAX / intermediate_features ||
          intermediate_features > SIZE_MAX / output_features ||
          input_features * intermediate_features >
              SIZE_MAX / sizeof(float) ||
          intermediate_features * output_features >
              SIZE_MAX / sizeof(float) ||
          request.shared_gate_weight.size !=
              input_features * intermediate_features * sizeof(float) ||
          request.shared_up_weight.size !=
              input_features * intermediate_features * sizeof(float) ||
          request.shared_down_weight.size !=
              intermediate_features * output_features * sizeof(float) ||
          request.shared_router_weight.size !=
              input_features * sizeof(float)) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      if (!RunGem5SharedExpertF32(
              request.input,
              static_cast<const float*>(request.shared_gate_weight.data),
              static_cast<const float*>(request.shared_up_weight.data),
              static_cast<const float*>(request.shared_down_weight.data),
              static_cast<const float*>(request.shared_router_weight.data),
              request.rows, input_features, intermediate_features,
              output_features, request.output, &result.stats)) {
        result.status = Gem5HostFunctionalStatus::kExecutionError;
      }
      return result;
    }
    const bool linear_projection =
        request.opcode == OPENNPUX_NPU_OP_MATMUL &&
        request.linear_qkv_weight.data != nullptr;
    if (linear_projection) {
      if (request.input == nullptr || request.output == nullptr ||
          request.output_secondary == nullptr ||
          request.output_tertiary == nullptr ||
          request.linear_alpha_weight.data == nullptr ||
          request.linear_beta_weight.data == nullptr ||
          request.operator_parameters->input_features == 0) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      const size_t input_features = request.operator_parameters->input_features;
      const Gem5GenericConstBuffer weights[] = {
          request.linear_qkv_weight, request.linear_alpha_weight,
          request.linear_beta_weight};
      float* outputs[] = {request.output, request.output_secondary,
                          request.output_tertiary};
      for (size_t index = 0; index < 3; ++index) {
        if (weights[index].size % sizeof(float) != 0 ||
            weights[index].size / sizeof(float) % input_features != 0) {
          return Result(Gem5HostFunctionalStatus::kInvalid);
        }
        const size_t output_features =
            weights[index].size / sizeof(float) / input_features;
        Gem5TransformerKernelStats projection_stats = {};
        if (output_features == 0 || !RunGem5MatMulF32(
                request.input, static_cast<const float*>(weights[index].data),
                request.rows, input_features, output_features, outputs[index],
                &projection_stats)) {
          return Result(Gem5HostFunctionalStatus::kExecutionError);
        }
        result.stats.operations += projection_stats.operations;
        result.stats.bytes_read += projection_stats.bytes_read;
        result.stats.bytes_written += projection_stats.bytes_written;
        result.stats.modeled_cycles += projection_stats.modeled_cycles;
      }
      return result;
    }
    // The executable carries model-level quantization capabilities, while
    // individual Qwen layers may still store dense F32/BF16 weights. Prefer
    // the materialized operand type over the capability flag.
    if (request.opcode == OPENNPUX_NPU_OP_MATMUL &&
        request.weight != nullptr) {
      const size_t output_features =
          request.operator_parameters->output_features;
      if (output_features == 0 || request.rows == 0 ||
          request.rows > SIZE_MAX / output_features ||
          request.rows * output_features > SIZE_MAX / sizeof(float) ||
          request.output_size !=
              request.rows * output_features * sizeof(float)) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      const size_t input_elements = request.input_size / sizeof(float);
      if (request.input_size % sizeof(float) != 0 ||
          input_elements % request.rows != 0) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      const size_t input_features = input_elements / request.rows;
      if (input_features == 0 || input_features > SIZE_MAX / output_features ||
          input_features * output_features > SIZE_MAX / sizeof(float) ||
          request.weight_size !=
              input_features * output_features * sizeof(float)) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      success = RunGem5MatMulF32(
          request.input, request.weight, request.rows,
          input_features, output_features, request.output, &result.stats);
    } else if (request.opcode == OPENNPUX_NPU_OP_MATMUL) {
      const bool fused_qkv = request.output_secondary != nullptr ||
                             request.output_tertiary != nullptr;
      if (fused_qkv) {
        if (request.output_secondary == nullptr ||
            request.output_tertiary == nullptr ||
            request.q_gptq_operands == nullptr ||
            request.k_gptq_operands == nullptr ||
            request.v_gptq_operands == nullptr || request.heads == 0 ||
            request.kv_heads == 0 || request.head_dim == 0) {
          return Result(Gem5HostFunctionalStatus::kInvalid);
        }
        opennpux_npu_operator_parameters projection =
            *request.operator_parameters;
        Gem5GptqKernelStats projection_stats = {};
        const size_t query_features = request.heads * request.head_dim;
        const bool gated_query = request.output_quaternary != nullptr;
        std::vector<float> raw_query;
        Gem5GenericGptqOperands query_operands = *request.q_gptq_operands;
        if (gated_query) {
          raw_query.resize(request.rows * query_features * 2);
          query_operands.output = {
              raw_query.data(), raw_query.size() * sizeof(float)};
        }
        projection.output_features =
            query_features * (gated_query ? 2 : 1);
        success = RunGem5GenericGptqMatMul(
            projection, static_cast<uint32_t>(request.rows),
            query_operands, &projection_stats);
        if (success) AddStats(projection_stats, &result.stats);
        if (success && gated_query) {
          for (size_t row = 0; row < request.rows; ++row) {
            for (size_t head = 0; head < request.heads; ++head) {
              const size_t raw_base = row * query_features * 2 +
                                      head * request.head_dim * 2;
              const size_t output_base = row * query_features +
                                         head * request.head_dim;
              std::copy_n(raw_query.data() + raw_base, request.head_dim,
                          request.output + output_base);
              std::copy_n(raw_query.data() + raw_base + request.head_dim,
                          request.head_dim,
                          request.output_quaternary + output_base);
            }
          }
        }
        projection.output_features = request.kv_heads * request.head_dim;
        projection_stats = {};
        success = success && RunGem5GenericGptqMatMul(
            projection, static_cast<uint32_t>(request.rows),
            *request.k_gptq_operands, &projection_stats);
        if (success) AddStats(projection_stats, &result.stats);
        projection_stats = {};
        success = success && RunGem5GenericGptqMatMul(
            projection, static_cast<uint32_t>(request.rows),
            *request.v_gptq_operands, &projection_stats);
        if (success) AddStats(projection_stats, &result.stats);
        const bool has_q_norm = request.attention_q_norm_weight.data != nullptr;
        const bool has_k_norm = request.attention_k_norm_weight.data != nullptr;
        if (success && (has_q_norm || has_k_norm)) {
          Gem5TransformerKernelStats q_norm_stats = {};
          Gem5TransformerKernelStats k_norm_stats = {};
          const bool norm_weight_offset =
              (request.operator_parameters->flags &
               OPENNPUX_NPU_PARAMETER_NORM_WEIGHT_OFFSET) != 0;
          success = has_q_norm && has_k_norm &&
              request.attention_q_norm_weight.size ==
                  request.head_dim * sizeof(float) &&
              request.attention_k_norm_weight.size ==
                  request.head_dim * sizeof(float) &&
              RunGem5RmsNormF32(
                  request.output,
                  static_cast<const float*>(
                      request.attention_q_norm_weight.data),
                  request.rows * request.heads, request.head_dim,
                  request.epsilon, request.output, &q_norm_stats,
                  norm_weight_offset,
                  (request.operator_parameters->flags &
                   OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE) != 0) &&
              RunGem5RmsNormF32(
                  request.output_secondary,
                  static_cast<const float*>(
                      request.attention_k_norm_weight.data),
                  request.rows * request.kv_heads, request.head_dim,
                  request.epsilon, request.output_secondary, &k_norm_stats,
                  norm_weight_offset,
                  (request.operator_parameters->flags &
                   OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE) != 0);
          if (success) {
            result.stats.operations +=
                q_norm_stats.operations + k_norm_stats.operations;
            result.stats.bytes_read +=
                q_norm_stats.bytes_read + k_norm_stats.bytes_read;
            result.stats.bytes_written +=
                q_norm_stats.bytes_written + k_norm_stats.bytes_written;
            result.stats.modeled_cycles +=
                q_norm_stats.modeled_cycles + k_norm_stats.modeled_cycles;
          }
        }
        if (!success) result.status = Gem5HostFunctionalStatus::kExecutionError;
        return result;
      }
      if (request.gptq_operands == nullptr) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      success = RunGem5GenericGptqMatMul(
          *request.operator_parameters, static_cast<uint32_t>(request.rows),
          *request.gptq_operands, &gptq_stats);
    } else {
      if (request.gptq_expert_operands == nullptr) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      success = RunGem5GenericGptqExpert(
          *request.operator_parameters, static_cast<uint32_t>(request.rows),
          *request.gptq_expert_operands, &gptq_stats);
    }
    if (success && request.weight == nullptr) {
      CopyStats(gptq_stats, &result.stats);
    } else if (!success) {
      result.status = Gem5HostFunctionalStatus::kExecutionError;
    }
    return result;
  }

  if (request.opcode == OPENNPUX_NPU_OP_EMBED) {
    const bool success = RunGem5EmbeddingF32(
        request.input_indices, request.rows, request.weight,
        request.vocabulary_size, request.features, request.output,
        &result.stats);
    if (!success) {
      result.status = Gem5HostFunctionalStatus::kExecutionError;
    }
    return result;
  }

  if (request.opcode == OPENNPUX_NPU_OP_DMA) {
    const bool success = RunGem5KvCacheUpdateF32(
        request.input, request.secondary, request.rows, request.kv_heads,
        request.head_dim, request.kv_length, request.output, &result.stats);
    if (!success) result.status = Gem5HostFunctionalStatus::kExecutionError;
    return result;
  }
  if (request.opcode == OPENNPUX_NPU_OP_ATTENTION) {
    bool success = RunGem5AttentionF32(
        request.input, request.secondary, request.rows, request.heads,
        request.kv_heads, request.head_dim, request.kv_length, request.output,
        &result.stats);
    size_t count = 0;
    if (success && request.tertiary != nullptr &&
        ElementCount(request, &count)) {
      for (size_t index = 0; index < count; ++index) {
        const float gate = 1.0f / (1.0f + std::exp(-request.tertiary[index]));
        request.output[index] *= gate;
        success = success && std::isfinite(request.output[index]);
      }
      result.stats.operations += count * 4;
      result.stats.bytes_read += count * 2 * sizeof(float);
      result.stats.bytes_written += count * sizeof(float);
      result.stats.modeled_cycles += count;
    }
    if (!success) result.status = Gem5HostFunctionalStatus::kExecutionError;
    return result;
  }
  if (request.opcode == OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION) {
    size_t count = 0;
    if (!ElementCount(request, &count) || request.weight == nullptr ||
        request.features == 0 ||
        request.operator_parameters == nullptr ||
        request.operator_parameters->intermediate_features == 0) {
      return Result(Gem5HostFunctionalStatus::kInvalid);
    }
    const size_t kernel_width =
        request.operator_parameters->intermediate_features;
    const bool stateful = request.secondary != nullptr &&
                          request.output_secondary != nullptr;
    const bool executed = stateful
        ? RunGem5CausalDepthwiseConvStatefulF32(
              request.input, request.weight, request.rows, request.features,
              kernel_width, request.secondary, request.output_secondary,
              request.output, &result.stats,
              (request.operator_parameters->flags &
               OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET) != 0)
        : RunGem5CausalDepthwiseConvF32(
              request.input, request.weight, request.rows, request.features,
              kernel_width, request.output, &result.stats,
              (request.operator_parameters->flags &
               OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET) != 0);
    if (!executed) {
      result.status = Gem5HostFunctionalStatus::kExecutionError;
    }
    return result;
  }
  if (request.opcode == OPENNPUX_NPU_OP_ROUTER &&
      ((request.gptq_operands != nullptr &&
        request.gptq_operands->qweight.data != nullptr) ||
       request.weight != nullptr)) {
    if (!RunRouter(request, &result.stats)) {
      result.status = Gem5HostFunctionalStatus::kExecutionError;
    }
    return result;
  }

  size_t count = 0;
  if (!ElementCount(request, &count) || request.input == nullptr ||
      (request.output == nullptr && request.opcode != OPENNPUX_NPU_OP_TOPK)) {
    return Result(Gem5HostFunctionalStatus::kInvalid);
  }
  switch (request.opcode) {
    case OPENNPUX_NPU_OP_ADD:
      success = RunGem5AddF32(request.input, request.secondary, count,
                              request.output, &result.stats);
      break;
    case OPENNPUX_NPU_OP_MUL:
      success = RunGem5MulF32(request.input, request.secondary, count,
                              request.output, &result.stats);
      break;
    case OPENNPUX_NPU_OP_NORMALIZE:
      success = RunGem5RmsNormF32(
          request.input, request.weight, request.rows, request.features,
          request.epsilon, request.output, &result.stats,
          request.operator_parameters != nullptr &&
              (request.operator_parameters->flags &
               OPENNPUX_NPU_PARAMETER_NORM_WEIGHT_OFFSET) != 0);
      break;
    case OPENNPUX_NPU_OP_ROPE: {
      if (request.heads == 0 || request.head_dim == 0 ||
          request.head_dim > std::numeric_limits<size_t>::max() /
                                 request.heads ||
          request.features != request.heads * request.head_dim ||
          ((request.secondary == nullptr) !=
           (request.output_secondary == nullptr)) ||
          (request.secondary != nullptr &&
           (request.kv_heads == 0 ||
            request.head_dim > std::numeric_limits<size_t>::max() /
                                   request.kv_heads))) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      const size_t rotary_dim =
          request.operator_parameters != nullptr &&
                  request.operator_parameters->intermediate_features != 0
              ? request.operator_parameters->intermediate_features
              : request.head_dim;
      success = RunGem5RopeF32(
          request.input, request.positions, request.rows, request.heads,
          request.head_dim, rotary_dim,
          request.rope_theta, request.output, &result.stats);
      if (success && request.secondary != nullptr) {
        Gem5TransformerKernelStats key_stats = {};
        success = RunGem5RopeF32(
            request.secondary, request.positions, request.rows,
            request.kv_heads, request.head_dim, rotary_dim,
            request.rope_theta, request.output_secondary, &key_stats);
        if (success) {
          AddStats(key_stats, &result.stats);
        }
      }
      break;
    }
    case OPENNPUX_NPU_OP_SOFTMAX:
      success = RunGem5SoftmaxF32(request.input, request.rows, request.features,
                                  request.output, &result.stats);
      break;
    case OPENNPUX_NPU_OP_TOPK:
      success = RunTopKRows(request, &result.stats);
      break;
    case OPENNPUX_NPU_OP_ROUTER:
      success = RunGem5TopKF32(request.input, count, request.top_k,
                              request.output, request.output_indices,
                              &result.stats);
      break;
    case OPENNPUX_NPU_OP_ACTIVATION:
      success = RunGem5SiluF32(request.input, count, request.output,
                               &result.stats);
      break;
    case OPENNPUX_NPU_OP_RECURRENT_UPDATE:
      if (request.operator_parameters != nullptr &&
          (request.operator_parameters->flags &
           OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET) != 0) {
        const size_t value_heads = request.kv_heads;
        const size_t value_dim = value_heads == 0 ? 0 :
            request.operator_parameters->output_features / value_heads;
        success = request.linear_a_log_weight.data != nullptr &&
                  request.linear_dt_bias_weight.data != nullptr &&
                  request.linear_a_log_weight.size ==
                      value_heads * sizeof(float) &&
                  request.linear_dt_bias_weight.size ==
                      value_heads * sizeof(float) &&
                  RunGem5GatedDeltaNetF32(
                      request.input, request.secondary, request.tertiary,
                      static_cast<const float*>(
                          request.linear_a_log_weight.data),
                      static_cast<const float*>(
                          request.linear_dt_bias_weight.data),
                      request.rows, request.heads, value_heads,
                      request.head_dim, value_dim, request.output,
                      request.output_secondary, &result.stats);
      } else {
        success = RunGem5RecurrentUpdateF32(
            request.input, request.rows, request.features, request.output,
            request.output_secondary, &result.stats);
      }
      break;
    case OPENNPUX_NPU_OP_COMBINE:
      success = RunGem5CombineF32(request.input, request.secondary, count,
                                  request.output, &result.stats);
      break;
    default:
      return Result(Gem5HostFunctionalStatus::kUnsupported);
  }
  if (!success) {
    result.status = Gem5HostFunctionalStatus::kExecutionError;
  }
  return result;
}

const char* Gem5HostFunctionalStatusName(Gem5HostFunctionalStatus status) {
  switch (status) {
    case Gem5HostFunctionalStatus::kComplete:
      return "complete";
    case Gem5HostFunctionalStatus::kInvalid:
      return "invalid";
    case Gem5HostFunctionalStatus::kUnsupported:
      return "unsupported";
    case Gem5HostFunctionalStatus::kExecutionError:
      return "execution-error";
  }
  return "unknown";
}
