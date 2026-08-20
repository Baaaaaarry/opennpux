#include "hw_sim/gem5_bridge/gem5_host_functional_backend.h"

#include <limits>

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

}  // namespace

bool Gem5HostFunctionalBackend::Supports(uint32_t opcode) const {
  switch (opcode) {
    case OPENNPUX_NPU_OP_MATMUL:
    case OPENNPUX_NPU_OP_ADD:
    case OPENNPUX_NPU_OP_MUL:
    case OPENNPUX_NPU_OP_NORMALIZE:
    case OPENNPUX_NPU_OP_ROPE:
    case OPENNPUX_NPU_OP_SOFTMAX:
    case OPENNPUX_NPU_OP_TOPK:
    case OPENNPUX_NPU_OP_ACTIVATION:
    case OPENNPUX_NPU_OP_EXPERT:
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
  if (request.opcode == OPENNPUX_NPU_OP_MATMUL ||
      request.opcode == OPENNPUX_NPU_OP_EXPERT) {
    if (request.operator_parameters == nullptr || request.rows == 0 ||
        request.rows > UINT32_MAX) {
      return Result(Gem5HostFunctionalStatus::kInvalid);
    }
    Gem5GptqKernelStats gptq_stats = {};
    if (request.opcode == OPENNPUX_NPU_OP_MATMUL) {
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
    if (success) {
      CopyStats(gptq_stats, &result.stats);
    } else {
      result.status = Gem5HostFunctionalStatus::kExecutionError;
    }
    return result;
  }

  size_t count = 0;
  if (!ElementCount(request, &count) || request.input == nullptr ||
      request.output == nullptr) {
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
          request.epsilon, request.output, &result.stats);
      break;
    case OPENNPUX_NPU_OP_ROPE:
      if (request.heads == 0 || request.head_dim == 0 ||
          request.head_dim > std::numeric_limits<size_t>::max() /
                                 request.heads ||
          request.features != request.heads * request.head_dim) {
        return Result(Gem5HostFunctionalStatus::kInvalid);
      }
      success = RunGem5RopeF32(
          request.input, request.positions, request.rows, request.heads,
          request.head_dim, request.rope_theta, request.output, &result.stats);
      break;
    case OPENNPUX_NPU_OP_SOFTMAX:
      success = RunGem5SoftmaxF32(request.input, request.rows, request.features,
                                  request.output, &result.stats);
      break;
    case OPENNPUX_NPU_OP_TOPK:
      success = RunGem5TopKF32(
          request.input, count, request.top_k, request.output,
          request.output_indices, &result.stats);
      break;
    case OPENNPUX_NPU_OP_ACTIVATION:
      success = RunGem5SiluF32(request.input, count, request.output,
                               &result.stats);
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
