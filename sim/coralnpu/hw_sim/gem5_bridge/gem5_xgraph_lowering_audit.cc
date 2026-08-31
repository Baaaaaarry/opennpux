#include "hw_sim/gem5_bridge/gem5_xgraph_lowering_audit.h"

#include <cerrno>
#include <cstring>
#include <vector>

#include "opennpux/npu_xgraph_lowering.h"

namespace {

constexpr uint32_t kAuditExtmemBase = 0;
constexpr uint32_t kAuditExtmemSize = UINT32_MAX;
constexpr uint32_t kAuditScratchAddress = UINT32_C(0x70000000);
constexpr uint32_t kAuditScratchSize = UINT32_C(0x10000000);

const void* Translate(const Gem5FunctionalMemoryRegion* regions,
                      size_t region_count, uint32_t address, size_t bytes) {
  for (size_t index = 0; index < region_count; ++index) {
    const auto& region = regions[index];
    if (address < region.base) {
      continue;
    }
    const uint64_t offset = static_cast<uint64_t>(address) - region.base;
    if (offset <= region.size && bytes <= region.size - offset) {
      return region.data + offset;
    }
  }
  return nullptr;
}

const char* OpcodeName(uint32_t opcode) {
  switch (opcode) {
    case OPENNPUX_NPU_OP_EMBED: return "EMBED";
    case OPENNPUX_NPU_OP_MATMUL: return "MATMUL";
    case OPENNPUX_NPU_OP_ADD: return "ADD";
    case OPENNPUX_NPU_OP_MUL: return "MUL";
    case OPENNPUX_NPU_OP_NORMALIZE: return "NORMALIZE";
    case OPENNPUX_NPU_OP_ROPE: return "ROPE";
    case OPENNPUX_NPU_OP_TOPK: return "TOPK";
    case OPENNPUX_NPU_OP_ROUTER: return "ROUTER";
    case OPENNPUX_NPU_OP_CONVOLUTION: return "CONVOLUTION";
    case OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION: return "CAUSAL_CONVOLUTION";
    case OPENNPUX_NPU_OP_RECURRENT_UPDATE: return "RECURRENT_UPDATE";
    case OPENNPUX_NPU_OP_EXPERT: return "EXPERT";
    case OPENNPUX_NPU_OP_DMA: return "DMA";
    case OPENNPUX_NPU_OP_ATTENTION: return "ATTENTION";
    case OPENNPUX_NPU_OP_SOFTMAX: return "SOFTMAX";
    case OPENNPUX_NPU_OP_ACTIVATION: return "ACTIVATION";
    case OPENNPUX_NPU_OP_COMBINE: return "COMBINE";
    default: return "UNKNOWN";
  }
}

}  // namespace

void Gem5XGraphLoweringAudit::RecordFailure(uint32_t opcode, int error_code,
                                             uint32_t command_id,
                                             const opennpux_npu_functional_request*
                                                 request,
                                             const opennpux_npu_operator_parameters*
                                                 parameters) {
  for (uint32_t index = 0; index < stats_.failure_count; ++index) {
    auto& failure = stats_.failures[index];
    if (failure.opcode == opcode && failure.error_code == error_code) {
      ++failure.count;
      return;
    }
  }
  if (stats_.failure_count == stats_.failures.size()) {
    return;
  }
  auto& failure = stats_.failures[stats_.failure_count++];
  failure.opcode = opcode;
  failure.error_code = error_code;
  failure.count = 1;
  failure.first_command = command_id;
  if (request != nullptr) {
    failure.first_request = *request;
  }
  if (parameters != nullptr) {
    failure.first_parameters = *parameters;
    failure.has_parameters = true;
  }
}

void Gem5XGraphLoweringAudit::Observe(
    Gem5HostFunctionalExecutionPath path,
    const opennpux_npu_functional_request& request,
    const Gem5FunctionalMemoryRegion* regions, size_t region_count) {
  ++stats_.observed_requests;
  if (path == Gem5HostFunctionalExecutionPath::kHostFusedRoutedExpert) {
    ++stats_.host_fused_requests;
    RecordFailure(request.opcode, ENOTSUP, request.command_id, &request,
                  nullptr);
    return;
  }
  const auto* parameters = static_cast<const opennpux_npu_operator_parameters*>(
      Translate(regions, region_count, request.parameter_address,
                sizeof(opennpux_npu_operator_parameters)));
  if (parameters == nullptr || request.parameter_size != sizeof(*parameters)) {
    RecordFailure(request.opcode, EINVAL, request.command_id, &request,
                  nullptr);
    return;
  }

  opennpux_npu_xgraph_lowering_options options = {};
  options.rope_layout = OPENNPUX_NPU_XGRAPH_ROPE_HALF_SPLIT;
  options.activation = OPENNPUX_NPU_XGRAPH_ACTIVATION_SILU;
  std::vector<opennpux_xgraph_command> commands(
      OPENNPUX_XGRAPH_MAX_COMMANDS);
  std::vector<uint32_t> origins(OPENNPUX_XGRAPH_MAX_COMMANDS);
  uint32_t consumed = 0;
  uint32_t emitted = 0;
  opennpux_npu_xgraph_lowering_failure failure = {};
  errno = 0;
  const int result = opennpux_npu_xgraph_lower_batch(
      &request, parameters, &options, 1, kAuditExtmemBase,
      kAuditExtmemSize, kAuditScratchAddress, kAuditScratchSize,
      commands.data(), static_cast<uint32_t>(commands.size()), origins.data(),
      &consumed, &emitted, &failure);
  if (result != 0 || consumed != 1 || emitted == 0) {
    const int error_code = failure.error_code != 0
                               ? failure.error_code
                               : (errno != 0 ? errno : EIO);
    RecordFailure(request.opcode, error_code, request.command_id, &request,
                  parameters);
    return;
  }
  ++stats_.lowerable_requests;
  stats_.emitted_commands += emitted;
}

void Gem5XGraphLoweringAudit::Print(FILE* stream) const {
  if (stream == nullptr) {
    return;
  }
  std::fprintf(stream, "xgraph_audit_observed_requests=%u\n",
               stats_.observed_requests);
  std::fprintf(stream, "xgraph_audit_lowerable_requests=%u\n",
               stats_.lowerable_requests);
  std::fprintf(stream, "xgraph_audit_host_fused_requests=%u\n",
               stats_.host_fused_requests);
  std::fprintf(stream, "xgraph_audit_emitted_commands=%llu\n",
               static_cast<unsigned long long>(stats_.emitted_commands));
  for (uint32_t index = 0; index < stats_.failure_count; ++index) {
    const auto& failure = stats_.failures[index];
    std::fprintf(stream,
                 "xgraph_audit_failure_%u=opcode:%s(%u),errno:%d,count:%u,"
                 "first_command:%u\n",
                 index, OpcodeName(failure.opcode), failure.opcode,
                 failure.error_code, failure.count, failure.first_command);
    const auto& request = failure.first_request;
    const auto& parameters = failure.first_parameters;
    std::fprintf(stream,
                 "xgraph_audit_failure_detail_%u=rows:%u,features:%u,heads:%u,"
                 "kv_heads:%u,head_dim:%u,kv_length:%u,top_k:%u,flags:0x%08x,"
                 "input:%u,output:%u,intermediate:%u,qbits:%u,qgroup:%u,"
                 "scale_dtype:%u,operands:",
                 index, request.rows, request.features, request.heads,
                 request.kv_heads, request.head_dim, request.kv_length,
                 request.top_k,
                 failure.has_parameters ? parameters.flags : 0,
                 failure.has_parameters ? parameters.input_features : 0,
                 failure.has_parameters ? parameters.output_features : 0,
                 failure.has_parameters ? parameters.intermediate_features
                                        : 0,
                 failure.has_parameters ? parameters.quantization_bits : 0,
                 failure.has_parameters ? parameters.quantization_group_size
                                        : 0,
                 failure.has_parameters ? parameters.scale_data_type : 0);
    for (uint32_t operand = 0; operand < request.operand_count; ++operand) {
      std::fprintf(stream, "%s%u@%u", operand == 0 ? "" : "/",
                   request.operands[operand].role,
                   request.operands[operand].byte_size);
    }
    std::fputc('\n', stream);
  }
  std::fprintf(stream, "xgraph_audit_complete=%s\n",
               stats_.observed_requests != 0 &&
                       stats_.lowerable_requests == stats_.observed_requests
                   ? "PASS"
                   : "INCOMPLETE");
}
