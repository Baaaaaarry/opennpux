#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_generic_test.h"
#include "hw_sim/gem5_bridge/opennpux/xopennpux_graph.h"
#include "hw_sim/gem5_bridge/xopennpux_ops.h"

namespace {

constexpr uint32_t kExtmemBase = UINT32_C(0x20000000);
constexpr uint32_t kExtmemSize = UINT32_C(0x00800000);
constexpr uint32_t kMailboxAddress =
    kExtmemBase + OPENNPUX_CORAL_GENERIC_TEST_MAILBOX_OFFSET;
constexpr uint32_t kGraphAddress = kExtmemBase + OPENNPUX_XGRAPH_OFFSET;

volatile opennpux_coral_generic_test_mailbox* Mailbox() {
  return reinterpret_cast<volatile opennpux_coral_generic_test_mailbox*>(
      static_cast<uintptr_t>(kMailboxAddress));
}

volatile opennpux_xgraph_header* Graph() {
  return reinterpret_cast<volatile opennpux_xgraph_header*>(
      static_cast<uintptr_t>(kGraphAddress));
}

const volatile opennpux_xgraph_command* Commands() {
  return reinterpret_cast<const volatile opennpux_xgraph_command*>(
      static_cast<uintptr_t>(kGraphAddress +
                             sizeof(opennpux_xgraph_header)));
}

void* Address(uint32_t offset) {
  return reinterpret_cast<void*>(
      static_cast<uintptr_t>(kExtmemBase + offset));
}

const void* ConstAddress(uint32_t offset) {
  return reinterpret_cast<const void*>(
      static_cast<uintptr_t>(kExtmemBase + offset));
}

bool RangeValid(uint32_t offset, uint64_t bytes) {
  return offset <= kExtmemSize && bytes <= kExtmemSize - offset;
}

uint32_t Fnv1a32(const volatile uint8_t* data, uint32_t bytes) {
  uint32_t hash = UINT32_C(2166136261);
  for (uint32_t index = 0; index < bytes; ++index) {
    hash ^= data[index];
    hash *= UINT32_C(16777619);
  }
  return hash;
}

int Fail(uint32_t error, uint32_t completed) {
  volatile opennpux_xgraph_header* graph = Graph();
  graph->error = error;
  graph->completed_commands = completed;
  graph->state = OPENNPUX_XGRAPH_STATE_ERROR;
  volatile opennpux_coral_generic_test_mailbox* mailbox = Mailbox();
  mailbox->magic = OPENNPUX_CORAL_GENERIC_TEST_MAGIC;
  mailbox->version = OPENNPUX_CORAL_GENERIC_TEST_VERSION;
  mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_ERROR;
  mailbox->error_code = error;
  mailbox->output[0] = static_cast<int32_t>(completed);
  return 1;
}

bool ValidateCommand(const volatile opennpux_xgraph_command& command) {
  if (command.data_type != OPENNPUX_XGRAPH_DTYPE_FP32 ||
      command.dim0 == 0 || command.dim1 == 0) {
    return false;
  }
  const uint64_t elements =
      static_cast<uint64_t>(command.dim0) * command.dim1;
  uint64_t source0_elements = elements;
  uint64_t source1_elements = elements;
  uint64_t destination_elements = elements;
  switch (command.opcode) {
    case OPENNPUX_XGRAPH_OP_TDEQUANT: {
      const uint32_t scale_type = (command.scalar0 >> 20) & 0xf;
      const uint32_t scale_bytes = scale_type == 2 ? 4 : 2;
      const uint32_t group_size = command.scalar0 & 0xffff;
      if (command.dim2 == 0 || group_size == 0 || scale_type > 2) {
        return false;
      }
      const uint32_t groups = command.flags >> 16;
      const uint32_t group_base = command.flags & 0xffff;
      if (groups == 0 || group_base >= groups) {
        return false;
      }
      return RangeValid(command.source0_offset,
                        static_cast<uint64_t>((command.dim2 + 7) / 8 - 1) *
                                command.reserved[2] +
                            command.dim1 * 4) &&
             RangeValid(command.source1_offset,
                        static_cast<uint64_t>(groups - 1) *
                                command.reserved[3] +
                            ((command.dim1 + 7) / 8) * 4) &&
             RangeValid(command.reserved[0],
                        static_cast<uint64_t>(groups - 1) *
                                command.reserved[4] +
                            command.dim1 * scale_bytes) &&
             (((command.scalar0 >> 24) & 1) == 0 ||
              RangeValid(command.reserved[1], command.dim2 * 4)) &&
             RangeValid(command.destination_offset,
                        static_cast<uint64_t>(command.dim2) * command.dim1 *
                            sizeof(float));
    }
    case OPENNPUX_XGRAPH_OP_TMMA:
      source0_elements = static_cast<uint64_t>(command.dim0) * command.dim2;
      source1_elements = static_cast<uint64_t>(command.dim2) * command.dim1;
      break;
    case OPENNPUX_XGRAPH_OP_TRMSNORM:
      source1_elements = command.dim1;
      break;
    case OPENNPUX_XGRAPH_OP_TROPE:
      source1_elements = elements * 2;
      break;
    case OPENNPUX_XGRAPH_OP_TGATHER:
      source0_elements = static_cast<uint64_t>(command.scalar0) * command.dim1;
      source1_elements = command.dim0;
      break;
    case OPENNPUX_XGRAPH_OP_TTOPK:
      source1_elements = 0;
      destination_elements =
          static_cast<uint64_t>(command.dim0) * command.scalar0 * 2;
      break;
    case OPENNPUX_XGRAPH_OP_TCAUSALCONV: {
      const bool stateful =
          (command.flags & OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL) != 0;
      if (command.dim2 == 0 ||
          (command.flags & ~(OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL |
                             OPENNPUX_XGRAPH_TCAUSALCONV_SILU)) != 0) {
        return false;
      }
      source1_elements =
          static_cast<uint64_t>(command.dim1) * command.dim2;
      if (stateful && command.dim2 > 1) {
        const uint64_t state_elements =
            static_cast<uint64_t>(command.dim2 - 1) * command.dim1;
        if (!RangeValid(command.reserved[0],
                        state_elements * sizeof(float)) ||
            !RangeValid(command.reserved[1],
                        state_elements * sizeof(float))) {
          return false;
        }
      } else if (command.reserved[0] != 0 || command.reserved[1] != 0) {
        return false;
      }
      break;
    }
    case OPENNPUX_XGRAPH_OP_TATTENTION: {
      const uint32_t heads = command.dim1;
      const uint32_t head_dim = command.dim2;
      const uint32_t kv_heads = command.scalar0;
      const uint32_t kv_length = command.flags;
      if (head_dim == 0 || kv_heads == 0 || kv_length == 0 ||
          command.dim0 > kv_length || heads % kv_heads != 0) {
        return false;
      }
      source0_elements =
          static_cast<uint64_t>(command.dim0) * heads * head_dim;
      source1_elements =
          static_cast<uint64_t>(2) * kv_length * kv_heads * head_dim;
      destination_elements = source0_elements;
      break;
    }
    case OPENNPUX_XGRAPH_OP_TSILU:
    case OPENNPUX_XGRAPH_OP_TSOFTMAX:
    case OPENNPUX_XGRAPH_OP_TDMA:
      source1_elements = 0;
      break;
    case OPENNPUX_XGRAPH_OP_TADD:
    case OPENNPUX_XGRAPH_OP_TMUL:
      break;
    default:
      return false;
  }
  return RangeValid(command.source0_offset, source0_elements * sizeof(float)) &&
         (source1_elements == 0 ||
          RangeValid(command.source1_offset,
                     source1_elements * sizeof(float))) &&
         RangeValid(command.destination_offset,
                    destination_elements * sizeof(float));
}

bool Execute(const volatile opennpux_xgraph_command& command,
             uint64_t* operations, uint64_t* cycles) {
  void* destination = Address(command.destination_offset);
  const void* source0 = ConstAddress(command.source0_offset);
  const void* source1 = ConstAddress(command.source1_offset);
  const uint64_t elements =
      static_cast<uint64_t>(command.dim0) * command.dim1;
  switch (command.opcode) {
    case OPENNPUX_XGRAPH_OP_TDEQUANT:
      xopennpux_dequant_int4_fp32(
          destination, source0, source1, ConstAddress(command.reserved[0]),
          ((command.scalar0 >> 24) & 1) != 0
              ? reinterpret_cast<const uint32_t*>(
                    ConstAddress(command.reserved[1]))
              : nullptr,
          command.dim1, command.dim2, command.scalar0 & 0xffff,
          (command.scalar0 >> 16) & 0xf, (command.scalar0 >> 20) & 0xf,
          command.reserved[2], command.reserved[3], command.reserved[4],
          command.flags & 0xffff, command.flags >> 16);
      *operations += static_cast<uint64_t>(command.dim1) * command.dim2;
      *cycles += static_cast<uint64_t>(command.dim1) * command.dim2;
      return true;
    case OPENNPUX_XGRAPH_OP_TMMA:
      xopennpux_matmul_fp32(destination, source0, source1, command.dim0,
                           command.dim1, command.dim2);
      *operations += elements * command.dim2;
      *cycles += elements * command.dim2;
      return true;
    case OPENNPUX_XGRAPH_OP_TADD:
      xopennpux_add_fp32(destination, source0, source1, command.dim0,
                         command.dim1, 1);
      *operations += elements;
      *cycles += elements;
      return true;
    case OPENNPUX_XGRAPH_OP_TMUL:
      xopennpux_mul_fp32(destination, source0, source1, command.dim0,
                         command.dim1, 1);
      *operations += elements;
      *cycles += elements;
      return true;
    case OPENNPUX_XGRAPH_OP_TRMSNORM: {
      union {
        uint32_t bits;
        float value;
      } epsilon = {command.scalar0};
      xopennpux_rmsnorm_fp32(destination, source0, source1, command.dim0,
                             command.dim1, epsilon.value);
      *operations += elements * 4;
      *cycles += elements * 4;
      return true;
    }
    case OPENNPUX_XGRAPH_OP_TSOFTMAX:
      xopennpux_softmax_fp32(destination, source0, command.dim0,
                             command.dim1);
      *operations += elements * 4;
      *cycles += elements * 4;
      return true;
    case OPENNPUX_XGRAPH_OP_TROPE:
      xopennpux_rope_fp32(
          destination, source0, source1, command.dim0, command.dim1,
          command.scalar0 == 0 ? XOPENNPUX_ROPE_ADJACENT
                               : XOPENNPUX_ROPE_HALF_SPLIT);
      *operations += elements * 3;
      *cycles += elements * 3;
      return true;
    case OPENNPUX_XGRAPH_OP_TSILU:
      xopennpux_silu_fp32(destination, source0, command.dim0, command.dim1);
      *operations += elements * 3;
      *cycles += elements * 3;
      return true;
    case OPENNPUX_XGRAPH_OP_TGATHER:
      xopennpux_gather_fp32(
          destination, source0,
          reinterpret_cast<const uint32_t*>(source1), command.dim0,
          command.dim1, command.scalar0);
      *operations += elements;
      *cycles += elements;
      return true;
    case OPENNPUX_XGRAPH_OP_TDMA:
      xopennpux_dma_fp32(destination, source0, command.dim0, command.dim1);
      *operations += elements;
      *cycles += elements;
      return true;
    case OPENNPUX_XGRAPH_OP_TTOPK:
      xopennpux_topk_fp32(destination, source0, command.dim0, command.dim1,
                          command.scalar0);
      *operations += elements * command.scalar0;
      *cycles += elements * command.scalar0;
      return true;
    case OPENNPUX_XGRAPH_OP_TCAUSALCONV:
      xopennpux_causal_depthwise_conv_fp32(
          destination, source0, source1, command.dim0, command.dim1,
          command.dim2,
          (command.flags & OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL) != 0
              ? ConstAddress(command.reserved[0])
              : nullptr,
          (command.flags & OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL) != 0
              ? Address(command.reserved[1])
              : nullptr,
          command.flags);
      *operations += elements * command.dim2 * 2;
      *cycles += elements * command.dim2 * 2;
      return true;
    case OPENNPUX_XGRAPH_OP_TATTENTION: {
      xopennpux_attention_fp32(destination, source0, source1, command.dim0,
                               command.dim1, command.scalar0, command.dim2,
                               command.flags);
      const uint64_t rows = command.dim0;
      const uint64_t visible_positions =
          rows * (command.flags - rows + 1) + rows * (rows - 1) / 2;
      const uint64_t attention_operations =
          visible_positions * command.dim1 * command.dim2 * 4;
      *operations += attention_operations;
      *cycles += attention_operations;
      return true;
    }
    default:
      return false;
  }
}

}  // namespace

int main() {
  volatile opennpux_coral_generic_test_mailbox* mailbox = Mailbox();
  mailbox->magic = OPENNPUX_CORAL_GENERIC_TEST_MAGIC;
  mailbox->version = OPENNPUX_CORAL_GENERIC_TEST_VERSION;
  mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_STARTED;
  mailbox->error_code = OPENNPUX_CORAL_GENERIC_TEST_ERROR_NONE;

  volatile opennpux_xgraph_header* graph = Graph();
  if (graph->magic != OPENNPUX_XGRAPH_MAGIC ||
      graph->version != OPENNPUX_XGRAPH_VERSION ||
      graph->header_size != sizeof(opennpux_xgraph_header) ||
      graph->command_size != sizeof(opennpux_xgraph_command) ||
      graph->command_count == 0 ||
      graph->command_count > OPENNPUX_XGRAPH_MAX_COMMANDS ||
      graph->total_size != sizeof(opennpux_xgraph_header) +
                               graph->command_count *
                                   sizeof(opennpux_xgraph_command) ||
      graph->state != OPENNPUX_XGRAPH_STATE_READY) {
    return Fail(OPENNPUX_XGRAPH_ERROR_ABI, 0);
  }

  graph->state = OPENNPUX_XGRAPH_STATE_RUNNING;
  graph->error = OPENNPUX_XGRAPH_ERROR_NONE;
  graph->completed_commands = 0;
  uint64_t operations = 0;
  uint64_t cycles = 0;
  const volatile opennpux_xgraph_command* commands = Commands();
  for (uint32_t index = 0; index < graph->command_count; ++index) {
    if (commands[index].command_id != index ||
        !ValidateCommand(commands[index])) {
      return Fail(OPENNPUX_XGRAPH_ERROR_BOUNDS, index);
    }
    if (!Execute(commands[index], &operations, &cycles)) {
      return Fail(OPENNPUX_XGRAPH_ERROR_OPCODE, index);
    }
    graph->completed_commands = index + 1;
  }

  if (!RangeValid(graph->output_offset, graph->output_bytes) ||
      graph->output_bytes < 2 * sizeof(uint32_t)) {
    return Fail(OPENNPUX_XGRAPH_ERROR_RESULT, graph->completed_commands);
  }
  const volatile uint32_t* output =
      reinterpret_cast<const volatile uint32_t*>(Address(graph->output_offset));
  const uint32_t output_checksum = Fnv1a32(
      reinterpret_cast<const volatile uint8_t*>(output), graph->output_bytes);
  graph->operation_count = operations;
  graph->modeled_cycles = cycles;
  graph->output_checksum = output_checksum;
  graph->state = OPENNPUX_XGRAPH_STATE_COMPLETE;

  mailbox->output[0] = static_cast<int32_t>(graph->completed_commands);
  mailbox->output[1] = static_cast<int32_t>(output[1]);
  mailbox->output[2] = static_cast<int32_t>(output[0]);
  mailbox->output[3] = 0;
  mailbox->output_count = OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT;
  mailbox->output_bytes = sizeof(mailbox->output);
  mailbox->output_checksum = Fnv1a32(
      reinterpret_cast<const volatile uint8_t*>(mailbox->output),
      mailbox->output_bytes);
  mailbox->operation_count = operations;
  mailbox->bytes_read = 0;
  mailbox->bytes_written = graph->output_bytes;
  mailbox->cycle_low = static_cast<uint32_t>(cycles);
  mailbox->cycle_high = static_cast<uint32_t>(cycles >> 32);
  mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_COMPLETE;
  return 0;
}
