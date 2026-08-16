#include "hw_sim/gem5_bridge/gem5_hybrid_operator.h"
#include "hw_sim/gem5_bridge/coral_gptq_matmul.h"
#include "hw_sim/gem5_bridge/qwen_device_inference.h"

#include <cassert>
#include <cmath>
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

void WriteFloat(std::vector<uint8_t>* memory, size_t offset, float value) {
  std::memcpy(memory->data() + offset, &value, sizeof(value));
}

float ReadFloat(const std::vector<uint8_t>& memory, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, memory.data() + offset, sizeof(value));
  return value;
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
  assert(descriptor.modeled_cycles == 1);
}

void TestAsyncRunningDescriptor() {
  std::vector<uint8_t> memory(256, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_CONV_2D_INT8);
  descriptor.state = CORAL_OPERATOR_STATE_RUNNING;
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
  assert(descriptor.state == CORAL_OPERATOR_STATE_COMPLETE);
  assert(descriptor.error == CORAL_OPERATOR_ERROR_NONE);
  assert(static_cast<int8_t>(memory[48]) == 7);
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
  assert(descriptor.operation_count == 2);
  assert(descriptor.modeled_cycles == 2);
}

void TestMatMulInt8() {
  std::vector<uint8_t> memory(256, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_MATMUL_INT8);
  descriptor.tensor_count = 3;
  SetTensor(&descriptor.tensors[0], kBase, 4, 2, 2, 2, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[1], kBase + 16, 4, 2, 2, 2, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[2], kBase + 32, 4, 2, 2, 2, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  memory[0] = 1;
  memory[1] = 2;
  memory[2] = 3;
  memory[3] = 4;
  memory[16] = 1;
  memory[17] = 0;
  memory[18] = 0;
  memory[19] = 1;

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(static_cast<int8_t>(memory[32]) == 1);
  assert(static_cast<int8_t>(memory[33]) == 2);
  assert(static_cast<int8_t>(memory[34]) == 3);
  assert(static_cast<int8_t>(memory[35]) == 4);
  assert(descriptor.operation_count == 8);
}

void TestFullyConnectedInt8() {
  std::vector<uint8_t> memory(256, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_FULLY_CONNECTED_INT8);
  descriptor.tensor_count = 4;
  SetTensor(&descriptor.tensors[0], kBase, 2, 2, 1, 2, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[1], kBase + 16, 4, 2, 2, 2, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[2], kBase + 32, 8, 1, 2, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT32);
  SetTensor(&descriptor.tensors[3], kBase + 48, 2, 2, 1, 2, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  memory[0] = 1;
  memory[1] = 2;
  memory[16] = 3;
  memory[17] = 4;
  memory[18] = 5;
  memory[19] = 6;
  Write32(&memory, 32, 1);
  Write32(&memory, 36, -1);

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(static_cast<int8_t>(memory[48]) == 12);
  assert(static_cast<int8_t>(memory[49]) == 16);
  assert(descriptor.operation_count == 4);
}

void TestAddInt8() {
  std::vector<uint8_t> memory(256, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_ADD_INT8);
  descriptor.tensor_count = 3;
  SetTensor(&descriptor.tensors[0], kBase, 3, 1, 3, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[1], kBase + 16, 3, 1, 3, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  SetTensor(&descriptor.tensors[2], kBase + 32, 3, 1, 3, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  memory[0] = 1;
  memory[1] = 2;
  memory[2] = 3;
  memory[16] = 4;
  memory[17] = 5;
  memory[18] = 6;

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(static_cast<int8_t>(memory[32]) == 5);
  assert(static_cast<int8_t>(memory[33]) == 7);
  assert(static_cast<int8_t>(memory[34]) == 9);
}

void TestSoftmaxFloat() {
  std::vector<uint8_t> memory(256, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_SOFTMAX);
  descriptor.tensor_count = 2;
  SetTensor(&descriptor.tensors[0], kBase, 8, 1, 2, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_FLOAT32);
  SetTensor(&descriptor.tensors[1], kBase + 16, 8, 1, 2, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_FLOAT32);
  WriteFloat(&memory, 0, 1.0f);
  WriteFloat(&memory, 4, 2.0f);

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(std::fabs(ReadFloat(memory, 16) - 0.268941f) < 0.0001f);
  assert(std::fabs(ReadFloat(memory, 20) - 0.731059f) < 0.0001f);
}

void TestLayerNormFloat() {
  std::vector<uint8_t> memory(256, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_LAYER_NORM);
  descriptor.tensor_count = 4;
  SetTensor(&descriptor.tensors[0], kBase, 16, 2, 2, 2, 0, 0,
            CORAL_OPERATOR_ELEMENT_FLOAT32);
  SetTensor(&descriptor.tensors[1], kBase + 32, 8, 1, 2, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_FLOAT32);
  SetTensor(&descriptor.tensors[2], kBase + 48, 8, 1, 2, 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_FLOAT32);
  SetTensor(&descriptor.tensors[3], kBase + 64, 16, 2, 2, 2, 0, 0,
            CORAL_OPERATOR_ELEMENT_FLOAT32);
  WriteFloat(&memory, 0, 1.0f);
  WriteFloat(&memory, 4, 2.0f);
  WriteFloat(&memory, 8, 3.0f);
  WriteFloat(&memory, 12, 4.0f);
  WriteFloat(&memory, 32, 1.0f);
  WriteFloat(&memory, 36, 1.0f);
  WriteFloat(&memory, 48, 0.0f);
  WriteFloat(&memory, 52, 0.0f);

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(std::fabs(ReadFloat(memory, 64) + 0.99998f) < 0.0001f);
  assert(std::fabs(ReadFloat(memory, 68) - 0.99998f) < 0.0001f);
  assert(std::fabs(ReadFloat(memory, 72) + 0.99998f) < 0.0001f);
  assert(std::fabs(ReadFloat(memory, 76) - 0.99998f) < 0.0001f);
}

void TestQwenTinyInfer() {
  std::vector<uint8_t> memory(sizeof(opennpux_qwen_device_request), 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_QWEN_TINY_INFER);
  descriptor.tensor_count = 1;
  SetTensor(&descriptor.tensors[0], kBase, memory.size(), 1, memory.size(), 0,
            0, 0, CORAL_OPERATOR_ELEMENT_INT8);
  auto* request = reinterpret_cast<opennpux_qwen_device_request*>(
      memory.data());
  request->magic = OPENNPUX_QWEN_DEVICE_MAGIC;
  request->version = OPENNPUX_QWEN_DEVICE_VERSION;
  request->struct_size = sizeof(*request);
  request->state = OPENNPUX_QWEN_DEVICE_PENDING;
  request->epsilon = 1.0e-5;
  for (double& weight : request->rms_attn_weight) weight = 1.0;
  for (double& weight : request->rms_ffn_weight) weight = 1.0;

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(request->state == OPENNPUX_QWEN_DEVICE_COMPLETE);
  assert(request->error == 0);
  assert(request->next_token == 0);
  assert(request->completed_operators == 19);
  assert(descriptor.modeled_cycles == 173);
}

void TestGptqMatMulDispatch() {
  std::vector<uint8_t> memory(1024, 0);
  auto descriptor = Descriptor(CORAL_OPERATOR_OP_GPTQ_MATMUL_INT4);
  descriptor.tensor_count = 1;
  constexpr uint32_t request_offset = 0;
  constexpr uint32_t input_offset = 128;
  constexpr uint32_t qweight_offset = 160;
  constexpr uint32_t qzeros_offset = 176;
  constexpr uint32_t scales_offset = 192;
  constexpr uint32_t output_offset = 224;
  SetTensor(&descriptor.tensors[0], kBase + request_offset,
            sizeof(coral_gptq_matmul_request), 1,
            sizeof(coral_gptq_matmul_request), 0, 0, 0,
            CORAL_OPERATOR_ELEMENT_INT8);
  auto* request = reinterpret_cast<coral_gptq_matmul_request*>(
      memory.data() + request_offset);
  request->magic = CORAL_GPTQ_MATMUL_MAGIC;
  request->version = CORAL_GPTQ_MATMUL_VERSION;
  request->struct_size = sizeof(*request);
  request->state = CORAL_GPTQ_MATMUL_PENDING;
  request->rows = 1;
  request->input_columns = 2;
  request->output_columns = 1;
  request->group_size = 2;
  request->zero_bias = 0;
  request->input_address = kBase + input_offset;
  request->qweight_address = kBase + qweight_offset;
  request->qzeros_address = kBase + qzeros_offset;
  request->scales_address = kBase + scales_offset;
  request->output_address = kBase + output_offset;
  WriteFloat(&memory, input_offset, 2.0f);
  WriteFloat(&memory, input_offset + 4, 3.0f);
  Write32(&memory, qweight_offset, 0x21);
  Write32(&memory, qzeros_offset, 0);
  WriteFloat(&memory, scales_offset, 0.5f);

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  assert(request->state == CORAL_GPTQ_MATMUL_COMPLETE);
  assert(request->error == CORAL_OPERATOR_ERROR_NONE);
  assert(std::fabs(ReadFloat(memory, output_offset) - 4.0f) < 1.0e-6f);
  assert(request->operations == 4);
  assert(request->output_checksum != 0);
}

}  // namespace

int main() {
  TestConv2D();
  TestAsyncRunningDescriptor();
  TestDepthwiseConv2D();
  TestMatMulInt8();
  TestFullyConnectedInt8();
  TestAddInt8();
  TestSoftmaxFloat();
  TestLayerNormFloat();
  TestQwenTinyInfer();
  TestGptqMatMulDispatch();
  return 0;
}
