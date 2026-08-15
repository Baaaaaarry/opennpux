#include "hw_sim/gem5_bridge/gem5_hybrid_operator.h"

#include <cstring>

#include "hw_sim/gem5_bridge/gem5_hybrid_kernels.h"

namespace {

bool RangeValid(uint32_t address, uint32_t size, uint32_t base,
                size_t capacity) {
  return size != 0 && size <= capacity && address >= base &&
         address - base <= capacity - size;
}

}  // namespace

bool ValidateGem5HybridDescriptor(
    const coral_operator_descriptor& descriptor, uint32_t extmem_base,
    size_t extmem_size, uint32_t* error) {
  if (error == nullptr) {
    return false;
  }
  *error = CORAL_OPERATOR_ERROR_NONE;
  if (descriptor.magic != CORAL_OPERATOR_ABI_MAGIC ||
      descriptor.version != CORAL_OPERATOR_ABI_VERSION ||
      descriptor.descriptor_size != sizeof(descriptor) ||
      descriptor.execution_mode != CORAL_OPERATOR_MODE_HYBRID ||
      (descriptor.state != CORAL_OPERATOR_STATE_SUBMITTED &&
       descriptor.state != CORAL_OPERATOR_STATE_RUNNING) ||
      descriptor.tensor_count > CORAL_OPERATOR_MAX_TENSORS) {
    *error = CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR;
    return false;
  }
  for (uint32_t i = 0; i < descriptor.tensor_count; ++i) {
    const coral_operator_tensor& tensor = descriptor.tensors[i];
    const bool type_valid =
        tensor.element_type == CORAL_OPERATOR_ELEMENT_INT8 ||
        tensor.element_type == CORAL_OPERATOR_ELEMENT_INT32 ||
        tensor.element_type == CORAL_OPERATOR_ELEMENT_FLOAT32;
    if (!type_valid || tensor.rank == 0 ||
        tensor.rank > CORAL_OPERATOR_MAX_DIMS ||
        !RangeValid(tensor.address, tensor.size, extmem_base, extmem_size)) {
      *error = CORAL_OPERATOR_ERROR_ADDRESS;
      return false;
    }
  }
  if ((descriptor.multiplier_address == 0) !=
      (descriptor.shift_address == 0)) {
    *error = CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR;
    return false;
  }
  if (descriptor.quantization_count != 0) {
    const uint64_t quantization_bytes =
        static_cast<uint64_t>(descriptor.quantization_count) * sizeof(int32_t);
    if (quantization_bytes > UINT32_MAX ||
        !RangeValid(descriptor.multiplier_address,
                    static_cast<uint32_t>(quantization_bytes), extmem_base,
                    extmem_size) ||
        !RangeValid(descriptor.shift_address,
                    static_cast<uint32_t>(quantization_bytes), extmem_base,
                    extmem_size)) {
      *error = CORAL_OPERATOR_ERROR_ADDRESS;
      return false;
    }
  }
  return true;
}

bool DispatchGem5HybridOperator(
    coral_operator_descriptor* descriptor, uint8_t* extmem,
    uint32_t extmem_base, size_t extmem_size,
    Gem5HybridOperatorResult* result) {
  if (descriptor == nullptr || extmem == nullptr || result == nullptr) {
    return false;
  }
  std::memset(result, 0, sizeof(*result));
  result->opcode = descriptor->opcode;
  uint32_t error = CORAL_OPERATOR_ERROR_NONE;
  if (!ValidateGem5HybridDescriptor(
          *descriptor, extmem_base, extmem_size, &error)) {
    descriptor->state = CORAL_OPERATOR_STATE_ERROR;
    descriptor->error = error;
    return false;
  }

  descriptor->state = CORAL_OPERATOR_STATE_RUNNING;
  descriptor->error = CORAL_OPERATOR_ERROR_NONE;
  descriptor->host_elapsed_ns = 0;
  descriptor->modeled_cycles = 0;
  descriptor->bytes_read = 0;
  descriptor->bytes_written = 0;

  bool success = false;
  switch (descriptor->opcode) {
    case CORAL_OPERATOR_OP_PARTIAL_MOBILENET:
      descriptor->error = CORAL_OPERATOR_ERROR_UNSUPPORTED;
      break;
    case CORAL_OPERATOR_OP_CONV_2D_INT8:
      success = RunGem5HybridConv2D(
          descriptor, extmem, extmem_base, extmem_size);
      break;
    case CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8:
      success = RunGem5HybridDepthwiseConv2D(
          descriptor, extmem, extmem_base, extmem_size);
      break;
    case CORAL_OPERATOR_OP_MATMUL_INT8:
      success = RunGem5HybridMatMulInt8(
          descriptor, extmem, extmem_base, extmem_size);
      break;
    case CORAL_OPERATOR_OP_FULLY_CONNECTED_INT8:
      success = RunGem5HybridFullyConnectedInt8(
          descriptor, extmem, extmem_base, extmem_size);
      break;
    case CORAL_OPERATOR_OP_ADD_INT8:
      success = RunGem5HybridAddInt8(
          descriptor, extmem, extmem_base, extmem_size);
      break;
    case CORAL_OPERATOR_OP_SOFTMAX:
      success = RunGem5HybridSoftmax(
          descriptor, extmem, extmem_base, extmem_size);
      break;
    case CORAL_OPERATOR_OP_LAYER_NORM:
      success = RunGem5HybridLayerNorm(
          descriptor, extmem, extmem_base, extmem_size);
      break;
    case CORAL_OPERATOR_OP_QWEN_TINY_INFER:
      success = RunGem5HybridQwenTinyInfer(
          descriptor, extmem, extmem_base, extmem_size);
      break;
    default:
      descriptor->error = CORAL_OPERATOR_ERROR_UNSUPPORTED;
      break;
  }

  if (!success && descriptor->error == CORAL_OPERATOR_ERROR_NONE) {
    descriptor->error = CORAL_OPERATOR_ERROR_EXECUTION;
  }
  if (success && descriptor->modeled_cycles == 0) {
    descriptor->modeled_cycles = descriptor->operation_count;
  }
  descriptor->state = success ? CORAL_OPERATOR_STATE_COMPLETE :
                                CORAL_OPERATOR_STATE_ERROR;
  return success;
}
