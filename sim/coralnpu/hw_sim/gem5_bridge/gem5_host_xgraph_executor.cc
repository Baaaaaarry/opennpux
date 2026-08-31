#include "hw_sim/gem5_bridge/gem5_host_xgraph_executor.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "hw_sim/gem5_bridge/gem5_tmma_coprocessor.h"
#include "hw_sim/gem5_bridge/xopennpux_isa.h"
#include "opennpux/npu_xgraph_lowering.h"
#include "opennpux/xopennpux_graph.h"

namespace {

constexpr uint32_t kRd = 10;
constexpr uint32_t kRs1 = 11;
constexpr uint32_t kRs2 = 12;

bool Multiply(uint64_t lhs, uint64_t rhs, uint64_t* result) {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool DeviceAddress(const Gem5HostTensorArena& arena, uint32_t offset,
                   uint32_t* address) {
  const uint64_t result = static_cast<uint64_t>(arena.base()) + offset;
  if (address == nullptr || result > UINT32_MAX) return false;
  *address = static_cast<uint32_t>(result);
  return true;
}

bool RangeInArena(const Gem5HostTensorArena& arena, uint32_t offset,
                  uint64_t elements, uint32_t element_bytes = sizeof(float)) {
  uint64_t bytes = 0;
  uint32_t address = 0;
  return Multiply(elements, element_bytes, &bytes) &&
         DeviceAddress(arena, offset, &address) &&
         arena.Translate(address, bytes) != nullptr;
}

bool IsSupportedPrimitive(uint32_t opcode) {
  switch (opcode) {
    case OPENNPUX_XGRAPH_OP_TADD:
    case OPENNPUX_XGRAPH_OP_TMUL:
    case OPENNPUX_XGRAPH_OP_TRMSNORM:
    case OPENNPUX_XGRAPH_OP_TSOFTMAX:
    case OPENNPUX_XGRAPH_OP_TROPE:
    case OPENNPUX_XGRAPH_OP_TSILU:
    case OPENNPUX_XGRAPH_OP_TSIGMOID:
    case OPENNPUX_XGRAPH_OP_TGATHER:
    case OPENNPUX_XGRAPH_OP_TTOPK:
    case OPENNPUX_XGRAPH_OP_TDMA:
    case OPENNPUX_XGRAPH_OP_TROW_SCALE:
      return true;
    default:
      return false;
  }
}

bool ValidateRanges(const opennpux_xgraph_command& command,
                    const Gem5HostTensorArena& arena) {
  uint64_t elements = 0;
  if (!Multiply(command.dim0, command.dim1, &elements)) return false;
  uint64_t source0_elements = elements;
  uint64_t source1_elements = elements;
  uint64_t destination_elements = elements;
  switch (command.opcode) {
    case OPENNPUX_XGRAPH_OP_TRMSNORM:
      source1_elements = command.dim1;
      break;
    case OPENNPUX_XGRAPH_OP_TROPE:
      if (!Multiply(elements, 2, &source1_elements)) return false;
      break;
    case OPENNPUX_XGRAPH_OP_TGATHER:
      if (!Multiply(command.scalar0, command.dim1, &source0_elements)) {
        return false;
      }
      source1_elements = command.dim0;
      break;
    case OPENNPUX_XGRAPH_OP_TTOPK:
      source1_elements = 0;
      if (!Multiply(command.dim0, command.scalar0, &destination_elements)) {
        return false;
      }
      if ((command.flags & OPENNPUX_XGRAPH_TTOPK_SPLIT_OUTPUT) != 0) {
        if (!RangeInArena(arena, command.reserved[0], destination_elements,
                          sizeof(uint32_t))) {
          return false;
        }
      } else if (!Multiply(destination_elements, 2,
                           &destination_elements)) {
        return false;
      }
      break;
    case OPENNPUX_XGRAPH_OP_TSILU:
    case OPENNPUX_XGRAPH_OP_TSIGMOID:
    case OPENNPUX_XGRAPH_OP_TSOFTMAX:
    case OPENNPUX_XGRAPH_OP_TDMA:
      source1_elements = 0;
      break;
    case OPENNPUX_XGRAPH_OP_TROW_SCALE:
      source1_elements = command.dim0;
      break;
    case OPENNPUX_XGRAPH_OP_TADD:
    case OPENNPUX_XGRAPH_OP_TMUL:
      break;
    default:
      return false;
  }
  return RangeInArena(arena, command.source0_offset, source0_elements) &&
         (source1_elements == 0 ||
          RangeInArena(arena, command.source1_offset, source1_elements)) &&
         RangeInArena(arena, command.destination_offset,
                      destination_elements);
}

bool CalculateTraffic(const opennpux_xgraph_command& command,
                      uint64_t* bytes_read, uint64_t* bytes_written) {
  uint64_t elements = 0;
  if (bytes_read == nullptr || bytes_written == nullptr ||
      !Multiply(command.dim0, command.dim1, &elements)) {
    return false;
  }
  uint64_t source0_elements = elements;
  uint64_t source1_elements = elements;
  uint64_t destination_elements = elements;
  switch (command.opcode) {
    case OPENNPUX_XGRAPH_OP_TRMSNORM:
      source1_elements = command.dim1;
      break;
    case OPENNPUX_XGRAPH_OP_TROPE:
      if (!Multiply(elements, 2, &source1_elements)) return false;
      break;
    case OPENNPUX_XGRAPH_OP_TGATHER:
      if (!Multiply(command.scalar0, command.dim1, &source0_elements)) {
        return false;
      }
      source1_elements = command.dim0;
      break;
    case OPENNPUX_XGRAPH_OP_TTOPK:
      source1_elements = 0;
      if (!Multiply(command.dim0, command.scalar0, &destination_elements) ||
          !Multiply(destination_elements, 2, &destination_elements)) {
        return false;
      }
      break;
    case OPENNPUX_XGRAPH_OP_TSILU:
    case OPENNPUX_XGRAPH_OP_TSIGMOID:
    case OPENNPUX_XGRAPH_OP_TSOFTMAX:
    case OPENNPUX_XGRAPH_OP_TDMA:
      source1_elements = 0;
      break;
    case OPENNPUX_XGRAPH_OP_TROW_SCALE:
      source1_elements = command.dim0;
      break;
    case OPENNPUX_XGRAPH_OP_TADD:
    case OPENNPUX_XGRAPH_OP_TMUL:
      break;
    default:
      return false;
  }
  if (source0_elements >
      std::numeric_limits<uint64_t>::max() - source1_elements) {
    return false;
  }
  const uint64_t read_elements = source0_elements + source1_elements;
  return Multiply(read_elements, sizeof(float), bytes_read) &&
         Multiply(destination_elements, sizeof(float), bytes_written);
}

uint32_t EncodeInstruction(uint32_t opcode) {
  switch (opcode) {
    case OPENNPUX_XGRAPH_OP_TADD:
      return xopennpux::EncodeTadd(kRd, kRs1, kRs2);
    case OPENNPUX_XGRAPH_OP_TMUL:
      return xopennpux::EncodeTmul(kRd, kRs1, kRs2);
    case OPENNPUX_XGRAPH_OP_TRMSNORM:
      return xopennpux::EncodeTrmsnorm(kRd, kRs1, kRs2);
    case OPENNPUX_XGRAPH_OP_TSOFTMAX:
      return xopennpux::EncodeTsoftmax(kRd, kRs1);
    case OPENNPUX_XGRAPH_OP_TROPE:
      return xopennpux::EncodeTrope(kRd, kRs1, kRs2);
    case OPENNPUX_XGRAPH_OP_TSILU:
      return xopennpux::EncodeTsilu(kRd, kRs1);
    case OPENNPUX_XGRAPH_OP_TSIGMOID:
      return xopennpux::EncodeTsigmoid(kRd, kRs1);
    case OPENNPUX_XGRAPH_OP_TGATHER:
      return xopennpux::EncodeTgather(kRd, kRs1, kRs2);
    case OPENNPUX_XGRAPH_OP_TTOPK:
      return xopennpux::EncodeTtopk(kRd, kRs1);
    case OPENNPUX_XGRAPH_OP_TDMA:
      return xopennpux::EncodeTdma(kRd, kRs1);
    case OPENNPUX_XGRAPH_OP_TROW_SCALE:
      return xopennpux::EncodeTrowScale(kRd, kRs1, kRs2);
    default:
      return 0;
  }
}

bool BuildPacket(const opennpux_xgraph_command& command,
                 const Gem5HostTensorArena& arena,
                 Gem5TmmaDispatchPacket* packet) {
  if (packet == nullptr ||
      !DeviceAddress(arena, command.source0_offset, &packet->rs1_value) ||
      !DeviceAddress(arena, command.source1_offset, &packet->rs2_value) ||
      !DeviceAddress(arena, command.destination_offset, &packet->rd_value)) {
    return false;
  }
  packet->instruction = EncodeInstruction(command.opcode);
  packet->sequence_id = command.command_id;
  packet->tensor_shape =
      xopennpux::EncodeTensorShape(command.dim0, command.dim1);
  packet->tensor_data_type = xopennpux::EncodeMmaDataTypes(
      xopennpux::DataType::kFp32, xopennpux::DataType::kFp32,
      xopennpux::DataType::kFp32);
  packet->scalar_param0 = command.scalar0;
  if (command.reserved[0] != 0 &&
      !DeviceAddress(arena, command.reserved[0],
                     &packet->tensor_aux_destination_address)) {
    return false;
  }
  packet->tensor_flags = command.flags;
  packet->csr_epoch = 1;
  return true;
}

}  // namespace

Gem5HostXGraphExecutionOutcome ExecuteGem5HostXGraphPrimitive(
    const opennpux_npu_functional_request& request,
    const opennpux_npu_operator_parameters& parameters,
    Gem5HostTensorArena* arena, Gem5HostXGraphExecutionStats* stats) {
  if (arena == nullptr || stats == nullptr || arena->size() > UINT32_MAX) {
    return Gem5HostXGraphExecutionOutcome::kError;
  }
  opennpux_npu_xgraph_lowering_options options = {};
  options.rope_layout = OPENNPUX_NPU_XGRAPH_ROPE_HALF_SPLIT;
  options.activation = OPENNPUX_NPU_XGRAPH_ACTIVATION_SILU;
  opennpux_xgraph_command command = {};
  errno = 0;
  if (opennpux_npu_xgraph_lower_primitive(
          &request, &parameters, &options, arena->base(),
          static_cast<uint32_t>(arena->size()), &command) != 0 ||
      !IsSupportedPrimitive(command.opcode) ||
      !ValidateRanges(command, *arena)) {
    return Gem5HostXGraphExecutionOutcome::kNotEligible;
  }

  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  Gem5TmmaDispatchPacket packet = {};
  if (!BuildPacket(command, *arena, &packet) || packet.instruction == 0 ||
      coprocessor.Submit(packet) != Gem5TmmaSubmitResult::kAccepted) {
    return Gem5HostXGraphExecutionOutcome::kError;
  }
  Gem5TmmaCompletion completion = {};
  if (!coprocessor.ExecuteNext(arena->mutable_storage_for_coprocessor(),
                               arena->base(), &completion) ||
      completion.error != Gem5TmmaExecutionError::kNone) {
    return Gem5HostXGraphExecutionOutcome::kError;
  }
  stats->commands = 1;
  stats->operations = completion.mac_operations + completion.element_operations;
  stats->modeled_cycles = completion.modeled_cycles;
  if (!CalculateTraffic(command, &stats->bytes_read, &stats->bytes_written)) {
    return Gem5HostXGraphExecutionOutcome::kError;
  }
  return Gem5HostXGraphExecutionOutcome::kExecuted;
}
