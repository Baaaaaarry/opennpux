#include "hw_sim/gem5_bridge/gem5_xgraph_lowering_audit.h"

#include <cassert>
#include <cerrno>
#include <cstring>
#include <vector>

#include "opennpux/npu_submission.h"

int main() {
  constexpr uint32_t kBase = UINT32_C(0x1000);
  std::vector<uint8_t> memory(0x5000, 0);
  auto* parameters = reinterpret_cast<opennpux_npu_operator_parameters*>(
      memory.data());
  parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters->struct_size = sizeof(*parameters);
  parameters->opcode = OPENNPUX_NPU_OP_ADD;
  parameters->input_features = 4;
  parameters->output_features = 4;

  opennpux_npu_functional_request request = {};
  request.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request.struct_size = sizeof(request);
  request.opcode = OPENNPUX_NPU_OP_ADD;
  request.command_id = 7;
  request.parameter_address = kBase;
  request.parameter_size = sizeof(*parameters);
  request.rows = 1;
  request.features = 4;
  request.operand_count = 3;
  request.operands[0] = {
      OPENNPUX_NPU_OPERAND_INPUT, kBase + 0x1000, 4 * sizeof(float), 0};
  request.operands[1] = {
      OPENNPUX_NPU_OPERAND_SECONDARY, kBase + 0x1100,
      4 * sizeof(float), 0};
  request.operands[2] = {
      OPENNPUX_NPU_OPERAND_OUTPUT, kBase + 0x1200, 4 * sizeof(float), 0};
  const Gem5FunctionalMemoryRegion region = {
      kBase, memory.data(), memory.size()};

  Gem5XGraphLoweringAudit audit;
  audit.Observe(Gem5HostFunctionalExecutionPath::kGenericRequest, request,
                &region, 1);
  request.opcode = OPENNPUX_NPU_OP_EXPERT;
  request.command_id = 8;
  audit.Observe(Gem5HostFunctionalExecutionPath::kHostFusedRoutedExpert,
                request, &region, 1);

  const auto& stats = audit.stats();
  assert(stats.observed_requests == 2);
  assert(stats.lowerable_requests == 1);
  assert(stats.host_fused_requests == 1);
  assert(stats.emitted_commands == 1);
  assert(stats.failure_count == 1);
  assert(stats.failures[0].opcode == OPENNPUX_NPU_OP_EXPERT);
  assert(stats.failures[0].error_code == ENOTSUP);
  assert(stats.failures[0].first_command == 8);
  return 0;
}
