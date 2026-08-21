#include "hw_sim/gem5_bridge/gem5_generic_command_dispatch.h"

#include <cstring>

#include "hw_sim/gem5_bridge/gem5_host_functional_backend.h"
#include "hw_sim/gem5_bridge/npu_functional_request.h"

namespace {

bool AddressSpaceValid(const Gem5FunctionalMemoryRegion* regions,
                       size_t region_count) {
  if (regions == nullptr || region_count == 0) {
    return false;
  }
  for (size_t index = 0; index < region_count; ++index) {
    const auto& region = regions[index];
    const uint64_t end = static_cast<uint64_t>(region.base) + region.size;
    if (region.data == nullptr || region.size == 0 ||
        end > (UINT64_C(1) << 32)) {
      return false;
    }
    for (size_t other = 0; other < index; ++other) {
      const uint64_t other_end =
          static_cast<uint64_t>(regions[other].base) + regions[other].size;
      if (region.base < other_end && regions[other].base < end) {
        return false;
      }
    }
  }
  return true;
}

void* Address(const Gem5FunctionalMemoryRegion* regions, size_t region_count,
              uint32_t address, size_t size) {
  if (regions == nullptr || size == 0) {
    return nullptr;
  }
  for (size_t index = 0; index < region_count; ++index) {
    const auto& region = regions[index];
    if (region.data != nullptr && address >= region.base &&
        size <= region.size &&
        static_cast<size_t>(address - region.base) <= region.size - size) {
      return region.data + (address - region.base);
    }
  }
  return nullptr;
}

const opennpux_npu_functional_operand* FindOperand(
    const opennpux_npu_functional_request& request, uint32_t role) {
  for (uint32_t index = 0; index < request.operand_count; ++index) {
    if (request.operands[index].role == role) {
      return &request.operands[index];
    }
  }
  return nullptr;
}

Gem5GenericConstBuffer ConstBuffer(
    const opennpux_npu_functional_request& request, uint32_t role,
    const Gem5FunctionalMemoryRegion* regions, size_t region_count) {
  const auto* operand = FindOperand(request, role);
  if (operand == nullptr) {
    return {};
  }
  return {Address(regions, region_count, operand->address, operand->byte_size),
          operand->byte_size};
}

Gem5GenericMutableBuffer MutableBuffer(
    const opennpux_npu_functional_request& request, uint32_t role,
    const Gem5FunctionalMemoryRegion* regions, size_t region_count) {
  const auto buffer = ConstBuffer(request, role, regions, region_count);
  return {const_cast<void*>(buffer.data), buffer.size};
}

bool Required(const Gem5GenericConstBuffer& buffer) {
  return buffer.data != nullptr && buffer.size != 0;
}

bool Required(const Gem5GenericMutableBuffer& buffer) {
  return buffer.data != nullptr && buffer.size != 0;
}

}  // namespace

bool ExecuteGem5FunctionalRequest(
    opennpux_npu_functional_request* request, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size) {
  const Gem5FunctionalMemoryRegion region = {
      extmem_base, extmem, extmem_size,
  };
  return ExecuteGem5FunctionalRequestInRegions(request, &region, 1);
}

bool ExecuteGem5FunctionalRequestInRegions(
    opennpux_npu_functional_request* request,
    const Gem5FunctionalMemoryRegion* regions, size_t region_count) {
  if (request == nullptr || !AddressSpaceValid(regions, region_count) ||
      request->magic != OPENNPUX_NPU_FUNCTIONAL_MAGIC ||
      request->version != OPENNPUX_NPU_FUNCTIONAL_VERSION ||
      request->struct_size != sizeof(*request) ||
      request->operand_count > OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS) {
    return false;
  }

  const auto input = ConstBuffer(*request, OPENNPUX_NPU_OPERAND_INPUT,
                                 regions, region_count);
  const auto input_indices = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_INPUT_INDICES, regions, region_count);
  const auto secondary = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_SECONDARY, regions, region_count);
  const auto tertiary = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_INPUT_TERTIARY, regions, region_count);
  const auto weight = ConstBuffer(*request, OPENNPUX_NPU_OPERAND_WEIGHT,
                                  regions, region_count);
  const auto positions = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_POSITIONS, regions, region_count);
  const auto output = MutableBuffer(*request, OPENNPUX_NPU_OPERAND_OUTPUT,
                                    regions, region_count);
  const auto output_indices = MutableBuffer(
      *request, OPENNPUX_NPU_OPERAND_OUTPUT_INDICES, regions, region_count);
  const auto output_secondary = MutableBuffer(
      *request, OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY, regions, region_count);
  const auto output_tertiary = MutableBuffer(
      *request, OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY, regions, region_count);
  const bool indices_only_topk =
      request->opcode == OPENNPUX_NPU_OP_TOPK && Required(output_indices);
  if ((request->opcode == OPENNPUX_NPU_OP_EMBED ?
           !Required(input_indices) : !Required(input)) ||
      (!indices_only_topk && !Required(output))) {
    request->state = CORAL_OPERATOR_STATE_ERROR;
    request->error = CORAL_OPERATOR_ERROR_ADDRESS;
    return false;
  }

  const auto* parameters = static_cast<const opennpux_npu_operator_parameters*>(
      Address(regions, region_count, request->parameter_address,
              request->parameter_size));
  Gem5GenericGptqOperands gptq = {
      input,
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_QWEIGHT, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_QZEROS, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_SCALES, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_G_IDX, regions, region_count),
      output,
  };
  Gem5GenericGptqOperands q_gptq = {
      input,
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_Q_QWEIGHT, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_Q_QZEROS, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_Q_SCALES, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_Q_G_IDX, regions, region_count),
      output,
  };
  Gem5GenericGptqOperands k_gptq = {
      input,
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_K_QWEIGHT, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_K_QZEROS, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_K_SCALES, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_K_G_IDX, regions, region_count),
      output_secondary,
  };
  Gem5GenericGptqOperands v_gptq = {
      input,
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_V_QWEIGHT, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_V_QZEROS, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_V_SCALES, regions, region_count),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_V_G_IDX, regions, region_count),
      output_tertiary,
  };
  const Gem5GenericGptqExpertOperands gptq_expert = {
      input,
      {ConstBuffer(*request, OPENNPUX_NPU_OPERAND_GATE_QWEIGHT, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_GATE_QZEROS, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_GATE_SCALES, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_GATE_G_IDX, regions,
                   region_count)},
      {ConstBuffer(*request, OPENNPUX_NPU_OPERAND_UP_QWEIGHT, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_UP_QZEROS, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_UP_SCALES, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_UP_G_IDX, regions,
                   region_count)},
      {ConstBuffer(*request, OPENNPUX_NPU_OPERAND_DOWN_QWEIGHT, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_DOWN_QZEROS, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_DOWN_SCALES, regions,
                   region_count),
       ConstBuffer(*request, OPENNPUX_NPU_OPERAND_DOWN_G_IDX, regions,
                   region_count)},
      MutableBuffer(*request, OPENNPUX_NPU_OPERAND_GATE_OUTPUT, regions,
                    region_count),
      MutableBuffer(*request, OPENNPUX_NPU_OPERAND_UP_OUTPUT, regions,
                    region_count),
      MutableBuffer(*request, OPENNPUX_NPU_OPERAND_ACTIVATED, regions,
                    region_count),
      output,
  };

  Gem5HostFunctionalRequest host = {};
  host.opcode = request->opcode;
  host.input = static_cast<const float*>(input.data);
  host.input_indices = static_cast<const uint32_t*>(input_indices.data);
  host.secondary = static_cast<const float*>(secondary.data);
  host.tertiary = static_cast<const float*>(tertiary.data);
  host.weight = static_cast<const float*>(weight.data);
  host.linear_qkv_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_LINEAR_QKV_WEIGHT, regions, region_count);
  host.linear_alpha_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_LINEAR_ALPHA_WEIGHT, regions,
      region_count);
  host.linear_beta_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_LINEAR_BETA_WEIGHT, regions,
      region_count);
  host.linear_gate_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_LINEAR_GATE_WEIGHT, regions,
      region_count);
  host.linear_norm_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_LINEAR_NORM_WEIGHT, regions,
      region_count);
  host.shared_gate_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_SHARED_GATE_WEIGHT, regions,
      region_count);
  host.shared_up_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_SHARED_UP_WEIGHT, regions,
      region_count);
  host.shared_down_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_SHARED_DOWN_WEIGHT, regions,
      region_count);
  host.shared_router_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_SHARED_ROUTER_WEIGHT, regions,
      region_count);
  host.attention_q_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_ATTENTION_Q_WEIGHT, regions,
      region_count);
  host.attention_k_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_ATTENTION_K_WEIGHT, regions,
      region_count);
  host.attention_v_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_ATTENTION_V_WEIGHT, regions,
      region_count);
  host.attention_q_norm_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_ATTENTION_Q_NORM_WEIGHT, regions,
      region_count);
  host.attention_k_norm_weight = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_ATTENTION_K_NORM_WEIGHT, regions,
      region_count);
  host.positions = static_cast<const uint32_t*>(positions.data);
  host.rows = request->rows;
  host.features = request->features;
  host.heads = request->heads;
  host.head_dim = request->head_dim;
  host.kv_heads = request->kv_heads;
  host.kv_length = request->kv_length;
  host.top_k = request->top_k;
  host.vocabulary_size = request->vocabulary_size;
  host.epsilon = request->epsilon;
  host.rope_theta = request->rope_theta;
  host.output = static_cast<float*>(output.data);
  host.output_secondary = static_cast<float*>(output_secondary.data);
  host.output_tertiary = static_cast<float*>(output_tertiary.data);
  host.output_indices = static_cast<uint32_t*>(output_indices.data);
  host.output_indices_count = output_indices.size / sizeof(uint32_t);
  host.operator_parameters = parameters;
  host.gptq_operands = &gptq;
  host.q_gptq_operands = &q_gptq;
  host.k_gptq_operands = &k_gptq;
  host.v_gptq_operands = &v_gptq;
  host.gptq_expert_operands = &gptq_expert;

  request->state = CORAL_OPERATOR_STATE_RUNNING;
  request->error = CORAL_OPERATOR_ERROR_NONE;
  const Gem5HostFunctionalResult result =
      Gem5HostFunctionalBackend().Execute(host);
  if (result.status != Gem5HostFunctionalStatus::kComplete) {
    request->state = CORAL_OPERATOR_STATE_ERROR;
    request->error = result.status == Gem5HostFunctionalStatus::kUnsupported
                         ? CORAL_OPERATOR_ERROR_UNSUPPORTED
                         : CORAL_OPERATOR_ERROR_EXECUTION;
    return false;
  }

  request->operation_count = result.stats.operations;
  request->modeled_cycles = result.stats.modeled_cycles;
  request->bytes_read = result.stats.bytes_read;
  request->bytes_written = result.stats.bytes_written;
  request->state = CORAL_OPERATOR_STATE_COMPLETE;
  return true;
}

bool DispatchGem5GenericCommand(coral_operator_descriptor* descriptor,
                                uint8_t* extmem, uint32_t extmem_base,
                                size_t extmem_size) {
  if (descriptor == nullptr || extmem == nullptr ||
      descriptor->tensor_count != 1) {
    return false;
  }
  const Gem5FunctionalMemoryRegion region = {
      extmem_base, extmem, extmem_size,
  };
  const coral_operator_tensor& envelope = descriptor->tensors[0];
  auto* request = static_cast<opennpux_npu_functional_request*>(Address(
      &region, 1, envelope.address, sizeof(opennpux_npu_functional_request)));
  if (request == nullptr ||
      envelope.size < sizeof(opennpux_npu_functional_request) ||
      request->opcode != descriptor->reserved[0]) {
    descriptor->error = CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR;
    return false;
  }
  if (!ExecuteGem5FunctionalRequest(request, extmem, extmem_base,
                                    extmem_size)) {
    descriptor->error = request->error == CORAL_OPERATOR_ERROR_NONE
                            ? static_cast<uint32_t>(
                                  CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR)
                            : request->error;
    return false;
  }
  descriptor->operation_count = request->operation_count;
  descriptor->modeled_cycles = request->modeled_cycles;
  descriptor->bytes_read = request->bytes_read;
  descriptor->bytes_written = request->bytes_written;
  return true;
}
