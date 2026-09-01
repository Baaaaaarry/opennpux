#include "hw_sim/gem5_bridge/gem5_host_xgraph_executor.h"

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "hw_sim/gem5_bridge/gem5_tmma_coprocessor.h"
#include "hw_sim/gem5_bridge/xopennpux_isa.h"
#include "opennpux/xopennpux_graph.h"

namespace {

constexpr uint32_t kRd = 10;
constexpr uint32_t kRs1 = 11;
constexpr uint32_t kRs2 = 12;
constexpr uint32_t kComplexScratchBytes = UINT32_C(64) * 1024 * 1024;
constexpr uint32_t kFunctionalOpcodeCount = 32;

bool ClaimFallbackDiagnostic(uint32_t opcode) {
  const char* value = std::getenv("OPENNPUX_HOST_XGRAPH_DEBUG");
  if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
    return false;
  }
  static bool reported[kFunctionalOpcodeCount] = {};
  if (opcode >= kFunctionalOpcodeCount || reported[opcode]) return false;
  reported[opcode] = true;
  return true;
}

bool Multiply(uint64_t lhs, uint64_t rhs, uint64_t* result) {
  if (result == nullptr ||
      (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)) {
    return false;
  }
  *result = lhs * rhs;
  return true;
}

bool DeviceAddress(uint32_t memory_base, uint32_t offset,
                   uint32_t* address) {
  const uint64_t result = static_cast<uint64_t>(memory_base) + offset;
  if (address == nullptr || result > UINT32_MAX) return false;
  *address = static_cast<uint32_t>(result);
  return true;
}

bool RangeInMemory(size_t memory_size, uint32_t offset, uint64_t elements,
                   uint32_t element_bytes = sizeof(float)) {
  uint64_t bytes = 0;
  return Multiply(elements, element_bytes, &bytes) && offset <= memory_size &&
         bytes <= memory_size - offset;
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
    case OPENNPUX_XGRAPH_OP_TCAUSALCONV:
    case OPENNPUX_XGRAPH_OP_TATTENTION:
      return true;
    default:
      return false;
  }
}

bool IsSupportedCommand(uint32_t opcode) {
  return IsSupportedPrimitive(opcode) || opcode == OPENNPUX_XGRAPH_OP_TMMA ||
         opcode == OPENNPUX_XGRAPH_OP_TDEQUANT ||
         opcode == OPENNPUX_XGRAPH_OP_TRECURRENT;
}

bool IsOutputRole(uint32_t role) {
  return role == OPENNPUX_NPU_OPERAND_OUTPUT ||
         role == OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY ||
         role == OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY ||
         role == OPENNPUX_NPU_OPERAND_OUTPUT_QUATERNARY ||
         role == OPENNPUX_NPU_OPERAND_OUTPUT_INDICES;
}

bool HasOperandRole(const opennpux_npu_functional_request& request,
                    uint32_t role) {
  for (uint32_t index = 0; index < request.operand_count; ++index) {
    if (request.operands[index].role == role) return true;
  }
  return false;
}

bool IsComplexCandidate(const opennpux_npu_functional_request& request) {
  if (request.opcode == OPENNPUX_NPU_OP_RECURRENT_UPDATE ||
      request.opcode == OPENNPUX_NPU_OP_DMA ||
      request.opcode == OPENNPUX_NPU_OP_TOPK) {
    return true;
  }
  // The dense FP32 router path is reference-checked below. GPTQ routers remain
  // on the Host C++ kernel until their dequantized projection is validated
  // against the generic GPTQ backend with production-scale weights.
  if (request.opcode == OPENNPUX_NPU_OP_ROUTER) {
    return !HasOperandRole(request, OPENNPUX_NPU_OPERAND_QWEIGHT) &&
           !HasOperandRole(request, OPENNPUX_NPU_OPERAND_QZEROS) &&
           !HasOperandRole(request, OPENNPUX_NPU_OPERAND_SCALES);
  }
  if (request.opcode != OPENNPUX_NPU_OP_MATMUL) return false;
  return HasOperandRole(request, OPENNPUX_NPU_OPERAND_WEIGHT) ||
         HasOperandRole(request, OPENNPUX_NPU_OPERAND_QWEIGHT) ||
         HasOperandRole(request, OPENNPUX_NPU_OPERAND_QZEROS) ||
         HasOperandRole(request, OPENNPUX_NPU_OPERAND_SCALES);
}

bool IsStagedPrimitiveCandidate(
    const opennpux_npu_functional_request& request) {
  // These operators commonly consume read-only weights supplied by the host
  // weight provider rather than resident tensor-arena storage.
  return request.opcode == OPENNPUX_NPU_OP_NORMALIZE ||
         request.opcode == OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION ||
         request.opcode == OPENNPUX_NPU_OP_EMBED;
}

const uint8_t* TranslateRegions(const Gem5FunctionalMemoryRegion* regions,
                                size_t region_count, uint32_t address,
                                uint32_t size) {
  for (size_t index = 0; index < region_count; ++index) {
    if (address < regions[index].base) continue;
    const uint64_t offset =
        static_cast<uint64_t>(address) - regions[index].base;
    if (regions[index].data != nullptr && offset <= regions[index].size &&
        size <= regions[index].size - offset) {
      return regions[index].data + offset;
    }
  }
  return nullptr;
}

bool AlignMemory(std::vector<uint8_t>* memory, size_t alignment) {
  if (memory == nullptr || alignment == 0) return false;
  const size_t remainder = memory->size() % alignment;
  if (remainder == 0) return true;
  try {
    memory->resize(memory->size() + alignment - remainder);
  } catch (...) {
    return false;
  }
  return true;
}

bool StageExternalOperands(
    const opennpux_npu_functional_request& request,
    const Gem5FunctionalMemoryRegion* regions, size_t region_count,
    uint32_t memory_base, size_t resident_bytes, std::vector<uint8_t>* memory,
    opennpux_npu_functional_request* staged) {
  if (regions == nullptr || memory == nullptr || staged == nullptr) {
    return false;
  }
  *staged = request;
  for (uint32_t index = 0; index < staged->operand_count; ++index) {
    auto& operand = staged->operands[index];
    const uint64_t arena_offset = operand.address >= memory_base
                                      ? operand.address - memory_base
                                      : UINT64_MAX;
    if (arena_offset <= resident_bytes &&
        operand.byte_size <= resident_bytes - arena_offset) {
      continue;
    }
    if (IsOutputRole(operand.role)) return false;
    const uint8_t* source = TranslateRegions(
        regions, region_count, operand.address, operand.byte_size);
    if (source == nullptr || !AlignMemory(memory, 64)) return false;
    const size_t offset = memory->size();
    if (offset > UINT32_MAX || operand.byte_size > UINT32_MAX - offset ||
        static_cast<uint64_t>(memory_base) + offset + operand.byte_size >
            (UINT64_C(1) << 32)) {
      return false;
    }
    try {
      memory->insert(memory->end(), source, source + operand.byte_size);
    } catch (...) {
      return false;
    }
    operand.address = memory_base + static_cast<uint32_t>(offset);
  }
  return true;
}

bool ValidateRanges(const opennpux_xgraph_command& command,
                    size_t memory_size) {
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
        if (!RangeInMemory(memory_size, command.reserved[0],
                           destination_elements, sizeof(uint32_t))) {
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
    case OPENNPUX_XGRAPH_OP_TCAUSALCONV: {
      if (command.dim2 == 0) return false;
      source1_elements = static_cast<uint64_t>(command.dim1) * command.dim2;
      const bool stateful =
          (command.flags & OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL) != 0;
      if (stateful && command.dim2 > 1) {
        const uint64_t state_elements =
            static_cast<uint64_t>(command.dim2 - 1) * command.dim1;
        if (!RangeInMemory(memory_size, command.reserved[0], state_elements) ||
            !RangeInMemory(memory_size, command.reserved[1], state_elements)) {
          return false;
        }
      }
      break;
    }
    case OPENNPUX_XGRAPH_OP_TRECURRENT: {
      const uint32_t key_heads = command.dim1;
      const uint32_t key_dim = command.dim2;
      const uint32_t value_heads = command.scalar0 & 0xffffu;
      const uint32_t value_dim = command.scalar0 >> 16;
      if (key_heads == 0 || key_dim == 0 || value_heads == 0 ||
          value_dim == 0 || value_heads % key_heads != 0) {
        return false;
      }
      source0_elements =
          static_cast<uint64_t>(command.dim0) *
          (static_cast<uint64_t>(2) * key_heads * key_dim +
           static_cast<uint64_t>(value_heads) * value_dim);
      source1_elements =
          static_cast<uint64_t>(command.dim0) * value_heads;
      destination_elements =
          static_cast<uint64_t>(command.dim0) * value_heads * value_dim;
      const uint64_t state_elements =
          static_cast<uint64_t>(value_heads) * key_dim * value_dim;
      if (!RangeInMemory(memory_size, command.reserved[0],
                         source1_elements) ||
          !RangeInMemory(memory_size, command.reserved[1], state_elements) ||
          !RangeInMemory(memory_size, command.reserved[2], value_heads) ||
          !RangeInMemory(memory_size, command.reserved[3], value_heads)) {
        return false;
      }
      break;
    }
    case OPENNPUX_XGRAPH_OP_TATTENTION: {
      const uint32_t heads = command.dim1;
      const uint32_t head_dim = command.dim2;
      const uint32_t kv_heads = command.scalar0;
      const uint32_t kv_length = command.flags;
      const uint32_t attention_flags = command.reserved[1];
      if (heads == 0 || head_dim == 0 || kv_heads == 0 || kv_length == 0 ||
          command.dim0 > kv_length || heads % kv_heads != 0) {
        return false;
      }
      source0_elements =
          static_cast<uint64_t>(command.dim0) * heads * head_dim;
      source1_elements =
          static_cast<uint64_t>(2) * kv_length * kv_heads * head_dim;
      destination_elements = source0_elements;
      if ((attention_flags & OPENNPUX_XGRAPH_TATTENTION_GATED) != 0 &&
          !RangeInMemory(memory_size, command.reserved[0],
                         source0_elements)) {
        return false;
      }
      break;
    }
    case OPENNPUX_XGRAPH_OP_TADD:
    case OPENNPUX_XGRAPH_OP_TMUL:
      break;
    default:
      return false;
  }
  return RangeInMemory(memory_size, command.source0_offset,
                       source0_elements) &&
         (source1_elements == 0 ||
          RangeInMemory(memory_size, command.source1_offset,
                        source1_elements)) &&
         RangeInMemory(memory_size, command.destination_offset,
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
    case OPENNPUX_XGRAPH_OP_TMMA: {
      uint64_t lhs_elements = 0;
      uint64_t rhs_elements = 0;
      if (!Multiply(command.dim0, command.dim2, &lhs_elements) ||
          !Multiply(command.dim2, command.dim1, &rhs_elements) ||
          lhs_elements > UINT64_MAX - rhs_elements) {
        return false;
      }
      return Multiply(lhs_elements + rhs_elements, sizeof(float),
                      bytes_read) &&
             Multiply(elements, sizeof(float), bytes_written);
    }
    case OPENNPUX_XGRAPH_OP_TDEQUANT: {
      uint64_t output_elements = 0;
      if (!Multiply(command.dim1, command.dim2, &output_elements)) {
        return false;
      }
      const uint64_t packed_values = (output_elements + 1) / 2;
      const uint64_t metadata =
          static_cast<uint64_t>(command.flags >> 16) * command.dim1 * 4;
      if (packed_values > UINT64_MAX - metadata) return false;
      *bytes_read = packed_values + metadata;
      return Multiply(output_elements, sizeof(float), bytes_written);
    }
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
    case OPENNPUX_XGRAPH_OP_TCAUSALCONV: {
      if (command.dim2 == 0) return false;
      source1_elements = static_cast<uint64_t>(command.dim1) * command.dim2;
      const bool stateful =
          (command.flags & OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL) != 0;
      if (stateful && command.dim2 > 1) {
        const uint64_t state_elements =
            static_cast<uint64_t>(command.dim2 - 1) * command.dim1;
        if (source0_elements > UINT64_MAX - state_elements ||
            destination_elements > UINT64_MAX - state_elements) {
          return false;
        }
        source0_elements += state_elements;
        destination_elements += state_elements;
      }
      break;
    }
    case OPENNPUX_XGRAPH_OP_TRECURRENT: {
      const uint32_t key_heads = command.dim1;
      const uint32_t key_dim = command.dim2;
      const uint32_t value_heads = command.scalar0 & 0xffffu;
      const uint32_t value_dim = command.scalar0 >> 16;
      if (key_heads == 0 || key_dim == 0 || value_heads == 0 ||
          value_dim == 0 || value_heads % key_heads != 0) {
        return false;
      }
      source0_elements =
          static_cast<uint64_t>(command.dim0) *
          (static_cast<uint64_t>(2) * key_heads * key_dim +
           static_cast<uint64_t>(value_heads) * value_dim);
      source1_elements =
          static_cast<uint64_t>(command.dim0) * value_heads * 2 +
          static_cast<uint64_t>(value_heads) * 2;
      destination_elements =
          static_cast<uint64_t>(command.dim0) * value_heads * value_dim;
      const uint64_t state_elements =
          static_cast<uint64_t>(value_heads) * key_dim * value_dim;
      if (source1_elements > UINT64_MAX - state_elements ||
          destination_elements > UINT64_MAX - state_elements) {
        return false;
      }
      source1_elements += state_elements;
      destination_elements += state_elements;
      break;
    }
    case OPENNPUX_XGRAPH_OP_TATTENTION: {
      const uint32_t heads = command.dim1;
      const uint32_t head_dim = command.dim2;
      const uint32_t kv_heads = command.scalar0;
      const uint32_t kv_length = command.flags;
      const uint32_t attention_flags = command.reserved[1];
      if (heads == 0 || head_dim == 0 || kv_heads == 0 || kv_length == 0 ||
          command.dim0 > kv_length || heads % kv_heads != 0) {
        return false;
      }
      source0_elements =
          static_cast<uint64_t>(command.dim0) * heads * head_dim;
      source1_elements =
          static_cast<uint64_t>(2) * kv_length * kv_heads * head_dim;
      destination_elements = source0_elements;
      if ((attention_flags & OPENNPUX_XGRAPH_TATTENTION_GATED) != 0) {
        if (source1_elements > UINT64_MAX - source0_elements) return false;
        source1_elements += source0_elements;
      }
      break;
    }
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
    case OPENNPUX_XGRAPH_OP_TMMA:
      return xopennpux::EncodeTmma(kRd, kRs1, kRs2);
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
    case OPENNPUX_XGRAPH_OP_TDEQUANT:
      return xopennpux::EncodeTdequant(kRd, kRs1);
    case OPENNPUX_XGRAPH_OP_TDMA:
      return xopennpux::EncodeTdma(kRd, kRs1);
    case OPENNPUX_XGRAPH_OP_TROW_SCALE:
      return xopennpux::EncodeTrowScale(kRd, kRs1, kRs2);
    case OPENNPUX_XGRAPH_OP_TCAUSALCONV:
      return xopennpux::EncodeTcausalconv(kRd, kRs1, kRs2);
    case OPENNPUX_XGRAPH_OP_TRECURRENT:
      return xopennpux::EncodeTrecurrent(kRd, kRs1, kRs2);
    case OPENNPUX_XGRAPH_OP_TATTENTION:
      return xopennpux::EncodeTattention(kRd, kRs1, kRs2);
    default:
      return 0;
  }
}

bool BuildPacket(const opennpux_xgraph_command& command,
                 uint32_t memory_base,
                 Gem5TmmaDispatchPacket* packet) {
  if (packet == nullptr ||
      !DeviceAddress(memory_base, command.source0_offset,
                     &packet->rs1_value) ||
      !DeviceAddress(memory_base, command.source1_offset,
                     &packet->rs2_value) ||
      !DeviceAddress(memory_base, command.destination_offset,
                     &packet->rd_value)) {
    return false;
  }
  packet->instruction = EncodeInstruction(command.opcode);
  packet->sequence_id = command.command_id;
  packet->csr_epoch = 1;
  if (command.opcode == OPENNPUX_XGRAPH_OP_TMMA) {
    packet->mma_shape =
        xopennpux::EncodeMmaShape(command.dim0, command.dim1, command.dim2);
    packet->mma_data_type = xopennpux::EncodeMmaDataTypes(
        xopennpux::DataType::kFp32, xopennpux::DataType::kFp32,
        xopennpux::DataType::kFp32);
    packet->mma_lhs_stride = command.reserved[0];
    packet->mma_rhs_stride = command.reserved[1];
    packet->mma_dst_stride = command.reserved[2];
    packet->mma_flags = command.flags;
    return true;
  }
  if (command.opcode == OPENNPUX_XGRAPH_OP_TDEQUANT) {
    packet->mma_shape =
        xopennpux::EncodeMmaShape(1, command.dim1, command.dim2);
    packet->mma_data_type = xopennpux::EncodeMmaDataTypes(
        xopennpux::DataType::kInt4, xopennpux::DataType::kFp32,
        xopennpux::DataType::kFp32);
    if (!DeviceAddress(memory_base, command.source1_offset,
                       &packet->quant_qzeros_address) ||
        !DeviceAddress(memory_base, command.reserved[0],
                       &packet->quant_scales_address) ||
        (((command.scalar0 >> 24) & 1) != 0 &&
         !DeviceAddress(memory_base, command.reserved[1],
                        &packet->quant_g_idx_address))) {
      return false;
    }
    packet->quant_config = command.scalar0;
    packet->quant_qweight_stride = command.reserved[2];
    packet->quant_qzeros_stride = command.reserved[3];
    packet->quant_scales_stride = command.reserved[4];
    packet->quant_group_range = command.flags;
    return true;
  }
  if (command.opcode == OPENNPUX_XGRAPH_OP_TCAUSALCONV) {
    packet->tensor_shape =
        xopennpux::EncodeTensorShape(command.dim0, command.dim1);
    packet->tensor_data_type = xopennpux::EncodeMmaDataTypes(
        xopennpux::DataType::kFp32, xopennpux::DataType::kFp32,
        xopennpux::DataType::kFp32);
    packet->scalar_param0 = command.dim2 | (command.flags << 16);
    if ((command.flags & OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL) != 0 &&
        (!DeviceAddress(memory_base, command.reserved[0],
                        &packet->tensor_aux_source_address) ||
         !DeviceAddress(memory_base, command.reserved[1],
                        &packet->tensor_aux_destination_address))) {
      return false;
    }
    return true;
  }
  if (command.opcode == OPENNPUX_XGRAPH_OP_TRECURRENT) {
    packet->tensor_shape = xopennpux::EncodeTensorShape(
        command.dim0, command.dim1 * command.dim2);
    packet->tensor_data_type = xopennpux::EncodeMmaDataTypes(
        xopennpux::DataType::kFp32, xopennpux::DataType::kFp32,
        xopennpux::DataType::kFp32);
    packet->recurrent_heads =
        command.dim1 | ((command.scalar0 & 0xffffu) << 16);
    packet->recurrent_dims =
        command.dim2 | ((command.scalar0 >> 16) << 16);
    if (!DeviceAddress(memory_base, command.reserved[0],
                       &packet->recurrent_beta_address) ||
        !DeviceAddress(memory_base, command.reserved[1],
                       &packet->tensor_aux_destination_address) ||
        !DeviceAddress(memory_base, command.reserved[2],
                       &packet->recurrent_a_log_address) ||
        !DeviceAddress(memory_base, command.reserved[3],
                       &packet->recurrent_dt_bias_address)) {
      return false;
    }
    return true;
  }
  if (command.opcode == OPENNPUX_XGRAPH_OP_TATTENTION) {
    packet->tensor_shape = xopennpux::EncodeTensorShape(
        command.dim0, command.dim1 * command.dim2);
    packet->tensor_data_type = xopennpux::EncodeMmaDataTypes(
        xopennpux::DataType::kFp32, xopennpux::DataType::kFp32,
        xopennpux::DataType::kFp32);
    packet->attention_heads = command.dim1 | (command.scalar0 << 16);
    packet->attention_head_dim_flags =
        command.dim2 | (command.reserved[1] << 16);
    packet->attention_kv_length = command.flags;
    if (command.reserved[0] != 0 &&
        !DeviceAddress(memory_base, command.reserved[0],
                       &packet->tensor_aux_source_address)) {
      return false;
    }
    return true;
  }
  packet->tensor_shape =
      xopennpux::EncodeTensorShape(command.dim0, command.dim1);
  packet->tensor_data_type = xopennpux::EncodeMmaDataTypes(
      xopennpux::DataType::kFp32, xopennpux::DataType::kFp32,
      xopennpux::DataType::kFp32);
  packet->scalar_param0 = command.scalar0;
  if (command.reserved[0] != 0 &&
      !DeviceAddress(memory_base, command.reserved[0],
                     &packet->tensor_aux_destination_address)) {
    return false;
  }
  packet->tensor_flags = command.flags;
  return true;
}

bool ExecuteCommands(const std::vector<opennpux_xgraph_command>& commands,
                     uint32_t command_count, uint32_t memory_base,
                     std::vector<uint8_t>* memory,
                     Gem5HostXGraphExecutionStats* stats) {
  if (memory == nullptr || stats == nullptr || command_count == 0 ||
      command_count > commands.size()) {
    return false;
  }
  for (uint32_t index = 0; index < command_count; ++index) {
    if (!IsSupportedCommand(commands[index].opcode) ||
        EncodeInstruction(commands[index].opcode) == 0) {
      return false;
    }
  }
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  for (uint32_t index = 0; index < command_count; ++index) {
    Gem5TmmaDispatchPacket packet = {};
    if (!BuildPacket(commands[index], memory_base, &packet) ||
        coprocessor.Submit(packet) != Gem5TmmaSubmitResult::kAccepted) {
      return false;
    }
    Gem5TmmaCompletion completion = {};
    if (!coprocessor.ExecuteNext(memory, memory_base, &completion) ||
        completion.error != Gem5TmmaExecutionError::kNone) {
      return false;
    }
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    if (!CalculateTraffic(commands[index], &bytes_read, &bytes_written)) {
      return false;
    }
    ++stats->commands;
    stats->operations +=
        completion.mac_operations + completion.element_operations;
    stats->modeled_cycles += completion.modeled_cycles;
    stats->bytes_read += bytes_read;
    stats->bytes_written += bytes_written;
  }
  return true;
}

}  // namespace

Gem5HostXGraphExecutionOutcome ExecuteGem5HostXGraphRequest(
    const opennpux_npu_functional_request& request,
    const opennpux_npu_operator_parameters& parameters,
    const Gem5FunctionalMemoryRegion* regions, size_t region_count,
    Gem5HostTensorArena* arena, Gem5HostXGraphExecutionStats* stats) {
  if (arena == nullptr || stats == nullptr || regions == nullptr ||
      arena->size() > UINT32_MAX) {
    return Gem5HostXGraphExecutionOutcome::kError;
  }
  *stats = {};
  opennpux_npu_xgraph_lowering_options options = {};
  options.rope_layout = OPENNPUX_NPU_XGRAPH_ROPE_HALF_SPLIT;
  options.activation = OPENNPUX_NPU_XGRAPH_ACTIVATION_SILU;
  opennpux_xgraph_command command = {};
  errno = 0;
  if (opennpux_npu_xgraph_lower_primitive(
          &request, &parameters, &options, arena->base(),
          static_cast<uint32_t>(arena->size()), &command) == 0 &&
      IsSupportedPrimitive(command.opcode) &&
      ValidateRanges(command, arena->size())) {
    const std::vector<opennpux_xgraph_command> commands = {command};
    if (!ExecuteCommands(commands, 1, arena->base(),
                         arena->mutable_storage_for_coprocessor(), stats)) {
      return Gem5HostXGraphExecutionOutcome::kError;
    }
    return Gem5HostXGraphExecutionOutcome::kExecuted;
  }
  if (!IsComplexCandidate(request) && !IsStagedPrimitiveCandidate(request)) {
    return Gem5HostXGraphExecutionOutcome::kNotEligible;
  }

  std::vector<uint8_t> memory;
  try {
    memory = *arena->mutable_storage_for_coprocessor();
  } catch (...) {
    return Gem5HostXGraphExecutionOutcome::kError;
  }
  const size_t resident_bytes = memory.size();
  opennpux_npu_functional_request staged = {};
  if (!StageExternalOperands(request, regions, region_count, arena->base(),
                             resident_bytes, &memory, &staged) ||
      !AlignMemory(&memory, 64)) {
    if (ClaimFallbackDiagnostic(request.opcode)) {
      std::fprintf(stderr,
                   "host_xgraph_fallback stage=external-staging opcode=%u "
                   "rows=%u features=%u operands=%u resident=%zu\n",
                   request.opcode, request.rows, request.features,
                   request.operand_count, resident_bytes);
    }
    return Gem5HostXGraphExecutionOutcome::kNotEligible;
  }

  command = {};
  errno = 0;
  const int staged_primitive_result = opennpux_npu_xgraph_lower_primitive(
      &staged, &parameters, &options, arena->base(),
      static_cast<uint32_t>(memory.size()), &command);
  const bool staged_primitive_supported =
      staged_primitive_result == 0 && IsSupportedPrimitive(command.opcode);
  const bool staged_primitive_ranges =
      staged_primitive_supported && ValidateRanges(command, memory.size());
  if (staged_primitive_ranges) {
    const std::vector<opennpux_xgraph_command> commands = {command};
    if (!ExecuteCommands(commands, 1, arena->base(), &memory, stats)) {
      return Gem5HostXGraphExecutionOutcome::kError;
    }
    std::memcpy(arena->mutable_storage_for_coprocessor()->data(),
                memory.data(), resident_bytes);
    return Gem5HostXGraphExecutionOutcome::kExecuted;
  }
  if (!IsComplexCandidate(request)) {
    if (ClaimFallbackDiagnostic(request.opcode)) {
      std::fprintf(stderr,
                   "host_xgraph_fallback stage=primitive-lowering opcode=%u "
                   "result=%d errno=%d lowered_opcode=%u supported=%u "
                   "ranges=%u resident=%zu staged=%zu\n",
                   staged.opcode, staged_primitive_result, errno,
                   command.opcode, staged_primitive_supported ? 1 : 0,
                   staged_primitive_ranges ? 1 : 0, resident_bytes,
                   memory.size());
    }
    return Gem5HostXGraphExecutionOutcome::kNotEligible;
  }

  const size_t scratch_offset = memory.size();
  if (scratch_offset > UINT32_MAX ||
      kComplexScratchBytes > UINT32_MAX - scratch_offset ||
      static_cast<uint64_t>(arena->base()) + scratch_offset +
              kComplexScratchBytes >
          (UINT64_C(1) << 32)) {
    return Gem5HostXGraphExecutionOutcome::kNotEligible;
  }
  try {
    memory.resize(scratch_offset + kComplexScratchBytes);
  } catch (...) {
    return Gem5HostXGraphExecutionOutcome::kError;
  }

  std::vector<opennpux_xgraph_command> commands(
      OPENNPUX_XGRAPH_MAX_COMMANDS);
  std::vector<uint32_t> origins(OPENNPUX_XGRAPH_MAX_COMMANDS);
  uint32_t requests_consumed = 0;
  uint32_t commands_emitted = 0;
  opennpux_npu_xgraph_lowering_failure failure = {};
  errno = 0;
  const int result = opennpux_npu_xgraph_lower_batch(
      &staged, &parameters, &options, 1, arena->base(),
      static_cast<uint32_t>(memory.size()),
      arena->base() + static_cast<uint32_t>(scratch_offset),
      kComplexScratchBytes, commands.data(),
      static_cast<uint32_t>(commands.size()), origins.data(),
      &requests_consumed, &commands_emitted, &failure);
  if (result != 0 || requests_consumed != 1 || commands_emitted == 0) {
    if (ClaimFallbackDiagnostic(staged.opcode)) {
      std::fprintf(stderr,
                   "host_xgraph_fallback stage=complex-lowering opcode=%u "
                   "result=%d errno=%d "
                   "consumed=%u emitted=%u failure_opcode=%u "
                   "failure_error=%d memory=%zu scratch=%zu\n",
                   staged.opcode, result, errno, requests_consumed,
                   commands_emitted, failure.opcode, failure.error_code,
                   memory.size(), scratch_offset);
      std::fprintf(stderr,
                   "host_xgraph_request rows=%u features=%u input=%u "
                   "output=%u qbits=%u qgroup=%u operands=%u flags=%#x\n",
                   staged.rows, staged.features,
                   parameters.input_features, parameters.output_features,
                   parameters.quantization_bits,
                   parameters.quantization_group_size, staged.operand_count,
                   parameters.flags);
      for (uint32_t index = 0; index < staged.operand_count; ++index) {
        std::fprintf(stderr,
                     "host_xgraph_operand index=%u role=%u address=%#x "
                     "bytes=%u flags=%#x\n",
                     index, staged.operands[index].role,
                     staged.operands[index].address,
                     staged.operands[index].byte_size,
                     staged.operands[index].reserved);
      }
    }
    return Gem5HostXGraphExecutionOutcome::kNotEligible;
  }
  for (uint32_t index = 0; index < commands_emitted; ++index) {
    if (!IsSupportedCommand(commands[index].opcode)) {
      if (ClaimFallbackDiagnostic(staged.opcode)) {
        std::fprintf(stderr,
                     "host_xgraph_fallback stage=unsupported-command "
                     "request_opcode=%u "
                     "command=%u command_opcode=%u\n",
                     staged.opcode, index, commands[index].opcode);
      }
      return Gem5HostXGraphExecutionOutcome::kNotEligible;
    }
  }
  if (!ExecuteCommands(commands, commands_emitted, arena->base(), &memory,
                       stats)) {
    return Gem5HostXGraphExecutionOutcome::kError;
  }
  std::memcpy(arena->data(), memory.data(), resident_bytes);
  return Gem5HostXGraphExecutionOutcome::kExecuted;
}
