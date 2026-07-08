#include "hw_sim/gem5_bridge/gem5_hybrid_operator.h"

#include <cassert>
#include <cstring>
#include <cstdint>
#include <vector>

namespace {

constexpr uint32_t kBase = 0x20000000;

void SetTensor(coral_operator_tensor* tensor, uint32_t address, uint32_t size,
               uint32_t rank, uint32_t d0, uint32_t d1, uint32_t d2,
               uint32_t d3, uint32_t type) {
  tensor->address = address;
  tensor->size = size;
  tensor->rank = rank;
  tensor->dimensions[0] = d0;
  tensor->dimensions[1] = d1;
  tensor->dimensions[2] = d2;
  tensor->dimensions[3] = d3;
  tensor->element_type = type;
}

coral_operator_descriptor Descriptor(uint32_t opcode) {
  coral_operator_descriptor descriptor = {};
  descriptor.magic = CORAL_OPERATOR_ABI_MAGIC;
  descriptor.version = CORAL_OPERATOR_ABI_VERSION;
  descriptor.descriptor_size = sizeof(descriptor);
  descriptor.opcode = opcode;
  descriptor.execution_mode = CORAL_OPERATOR_MODE_HYBRID;
  descriptor.state = CORAL_OPERATOR_STATE_SUBMITTED;
  descriptor.tensor_count = 4;
  descriptor.stride_height = 1;
  descriptor.stride_width = 1;
  descriptor.activation_min = -128;
  descriptor.activation_max = 127;
  return descriptor;
}

void Write32(std::vector<uint8_t>* memory, size_t offset, int32_t value) {
  std::memcpy(memory->data() + offset, &value, sizeof(value));
}

void TestConv2D() {
  std::vector<uint8_t> memory(256, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_CONV_2D_INT8);
  SetTensor(&descriptor.tensors[0], kBase, 1, 4, 1, 1, 1, 1,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[1], kBase + 16, 1, 4, 1, 1, 1, 1,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[2], kBase + 32, 4, 1, 1, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT32);
  SetTensor(&descriptor.tensors[3], kBase + 48, 1, 4, 1, 1, 1, 1,
            CORAL_OPERATOR_ELEMENT_INT8);
  memory[0] = 3;
  memory[16] = 2;
  Write32(&memory, 32, 1);
  Write32(&memory, 64, 1073741824);
  Write32(&memory, 68, 1);
  descriptor.multiplier_address = kBase + 64;
  descriptor.shift_address = kBase + 68;
  descriptor.quantization_count = 1;

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(static_cast<int8_t>(memory[48]) == 7);
  assert(descriptor.operation_count == 1);
}

void TestDepthwiseConv2D() {
  std::vector<uint8_t> memory(256, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8);
  SetTensor(&descriptor.tensors[0], kBase, 2, 4, 1, 1, 1, 2,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[1], kBase + 16, 2, 4, 1, 1, 1, 2,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[2], kBase + 32, 8, 1, 2, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT32);
  SetTensor(&descriptor.tensors[3], kBase + 48, 2, 4, 1, 1, 1, 2,
            CORAL_OPERATOR_ELEMENT_INT8);
  memory[0] = 3;
  memory[1] = 4;
  memory[16] = 2;
  memory[17] = 3;
  Write32(&memory, 32, 1);
  Write32(&memory, 36, 2);
  Write32(&memory, 64, 1073741824);
  Write32(&memory, 68, 1073741824);
  Write32(&memory, 72, 1);
  Write32(&memory, 76, 1);
  descriptor.multiplier_address = kBase + 64;
  descriptor.shift_address = kBase + 72;
  descriptor.quantization_count = 2;

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(static_cast<int8_t>(memory[48]) == 7);
  assert(static_cast<int8_t>(memory[49]) == 14);
}

}  // namespace

int main() {
  TestConv2D();
  TestDepthwiseConv2D();
  return 0;
}
