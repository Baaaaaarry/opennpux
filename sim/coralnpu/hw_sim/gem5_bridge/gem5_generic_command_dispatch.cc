#include "hw_sim/gem5_bridge/gem5_generic_command_dispatch.h"

#include <cstring>

#include "hw_sim/gem5_bridge/gem5_host_functional_backend.h"
#include "hw_sim/gem5_bridge/npu_functional_request.h"

namespace {

bool RangeValid(uint32_t address, size_t size, uint32_t base,
                size_t capacity) {
  return size != 0 && size <= capacity && address >= base &&
         static_cast<size_t>(address - base) <= capacity - size;
}

void* Address(uint8_t* extmem, uint32_t extmem_base, size_t extmem_size,
              uint32_t address, size_t size) {
  if (!RangeValid(address, size, extmem_base, extmem_size)) {
    return nullptr;
  }
  return extmem + (address - extmem_base);
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
    uint8_t* extmem, uint32_t extmem_base, size_t extmem_size) {
  const auto* operand = FindOperand(request, role);
  if (operand == nullptr) {
    return {};
  }
  return {Address(extmem, extmem_base, extmem_size, operand->address,
                  operand->byte_size),
          operand->byte_size};
}

Gem5GenericMutableBuffer MutableBuffer(
    const opennpux_npu_functional_request& request, uint32_t role,
    uint8_t* extmem, uint32_t extmem_base, size_t extmem_size) {
  const auto buffer = ConstBuffer(request, role, extmem, extmem_base,
                                  extmem_size);
  return {const_cast<void*>(buffer.data), buffer.size};
}

bool Required(const Gem5GenericConstBuffer& buffer) {
  return buffer.data != nullptr && buffer.size != 0;
}

}  // namespace

bool DispatchGem5GenericCommand(coral_operator_descriptor* descriptor,
                                uint8_t* extmem, uint32_t extmem_base,
                                size_t extmem_size) {
  if (descriptor == nullptr || extmem == nullptr ||
      descriptor->tensor_count != 1) {
    return false;
  }
  const coral_operator_tensor& envelope = descriptor->tensors[0];
  auto* request = static_cast<opennpux_npu_functional_request*>(Address(
      extmem, extmem_base, extmem_size, envelope.address,
      sizeof(opennpux_npu_functional_request)));
  if (request == nullptr ||
      envelope.size < sizeof(opennpux_npu_functional_request) ||
      request->magic != OPENNPUX_NPU_FUNCTIONAL_MAGIC ||
      request->version != OPENNPUX_NPU_FUNCTIONAL_VERSION ||
      request->struct_size != sizeof(*request) ||
      request->operand_count > OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS ||
      request->opcode != descriptor->reserved[0]) {
    descriptor->error = CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR;
    return false;
  }

  const auto input = ConstBuffer(*request, OPENNPUX_NPU_OPERAND_INPUT,
                                 extmem, extmem_base, extmem_size);
  const auto secondary = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_SECONDARY, extmem, extmem_base,
      extmem_size);
  const auto weight = ConstBuffer(*request, OPENNPUX_NPU_OPERAND_WEIGHT,
                                  extmem, extmem_base, extmem_size);
  const auto positions = ConstBuffer(
      *request, OPENNPUX_NPU_OPERAND_POSITIONS, extmem, extmem_base,
      extmem_size);
  const auto output = MutableBuffer(*request, OPENNPUX_NPU_OPERAND_OUTPUT,
                                    extmem, extmem_base, extmem_size);
  const auto output_indices = MutableBuffer(
      *request, OPENNPUX_NPU_OPERAND_OUTPUT_INDICES, extmem, extmem_base,
      extmem_size);
  if (!Required(input) || output.data == nullptr || output.size == 0) {
    descriptor->error = CORAL_OPERATOR_ERROR_ADDRESS;
    return false;
  }

  const auto* parameters = static_cast<const opennpux_npu_operator_parameters*>(
      Address(extmem, extmem_base, extmem_size, request->parameter_address,
              request->parameter_size));
  Gem5GenericGptqOperands gptq = {
      input,
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_QWEIGHT, extmem,
                  extmem_base, extmem_size),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_QZEROS, extmem,
                  extmem_base, extmem_size),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_SCALES, extmem,
                  extmem_base, extmem_size),
      ConstBuffer(*request, OPENNPUX_NPU_OPERAND_G_IDX, extmem,
                  extmem_base, extmem_size),
      output,
  };

  Gem5HostFunctionalRequest host = {};
  host.opcode = request->opcode;
  host.input = static_cast<const float*>(input.data);
  host.secondary = static_cast<const float*>(secondary.data);
  host.weight = static_cast<const float*>(weight.data);
  host.positions = static_cast<const uint32_t*>(positions.data);
  host.rows = request->rows;
  host.features = request->features;
  host.heads = request->heads;
  host.head_dim = request->head_dim;
  host.top_k = request->top_k;
  host.epsilon = request->epsilon;
  host.rope_theta = request->rope_theta;
  host.output = static_cast<float*>(output.data);
  host.output_indices = static_cast<uint32_t*>(output_indices.data);
  host.operator_parameters = parameters;
  host.gptq_operands = &gptq;

  request->state = CORAL_OPERATOR_STATE_RUNNING;
  request->error = CORAL_OPERATOR_ERROR_NONE;
  const Gem5HostFunctionalResult result =
      Gem5HostFunctionalBackend().Execute(host);
  if (result.status != Gem5HostFunctionalStatus::kComplete) {
    request->state = CORAL_OPERATOR_STATE_ERROR;
    request->error = result.status == Gem5HostFunctionalStatus::kUnsupported
                         ? CORAL_OPERATOR_ERROR_UNSUPPORTED
                         : CORAL_OPERATOR_ERROR_EXECUTION;
    descriptor->error = request->error;
    return false;
  }

  request->operation_count = result.stats.operations;
  request->modeled_cycles = result.stats.modeled_cycles;
  request->bytes_read = result.stats.bytes_read;
  request->bytes_written = result.stats.bytes_written;
  request->state = CORAL_OPERATOR_STATE_COMPLETE;
  descriptor->operation_count = result.stats.operations;
  descriptor->modeled_cycles = result.stats.modeled_cycles;
  descriptor->bytes_read = result.stats.bytes_read;
  descriptor->bytes_written = result.stats.bytes_written;
  return true;
}
