#include "hw_sim/gem5_bridge/gem5_generic_command_dispatch.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "hw_sim/gem5_bridge/npu_functional_request.h"
#include "hw_sim/gem5_bridge/npu_submission.h"

namespace {

constexpr uint32_t kBase = UINT32_C(0x20000000);

template <typename T>
T* At(std::vector<uint8_t>* memory, uint32_t offset) {
  return reinterpret_cast<T*>(memory->data() + offset);
}

coral_operator_descriptor Descriptor(uint32_t request_offset,
                                     uint32_t generic_opcode) {
  coral_operator_descriptor descriptor = {};
  descriptor.magic = CORAL_OPERATOR_ABI_MAGIC;
  descriptor.version = CORAL_OPERATOR_ABI_VERSION;
  descriptor.descriptor_size = sizeof(descriptor);
  descriptor.opcode = CORAL_OPERATOR_OP_GENERIC_COMMAND;
  descriptor.state = CORAL_OPERATOR_STATE_SUBMITTED;
  descriptor.execution_mode = CORAL_OPERATOR_MODE_HYBRID;
  descriptor.tensor_count = 1;
  descriptor.tensors[0].address = kBase + request_offset;
  descriptor.tensors[0].size = sizeof(opennpux_npu_functional_request);
  descriptor.tensors[0].rank = 1;
  descriptor.tensors[0].dimensions[0] = descriptor.tensors[0].size;
  descriptor.tensors[0].element_type = CORAL_OPERATOR_ELEMENT_INT8;
  descriptor.reserved[0] = generic_opcode;
  return descriptor;
}

void AddOperand(opennpux_npu_functional_request* request, uint32_t role,
                uint32_t offset, uint32_t size) {
  auto& operand = request->operands[request->operand_count++];
  operand.role = role;
  operand.address = kBase + offset;
  operand.byte_size = size;
}

void TestAdd() {
  std::vector<uint8_t> memory(4096);
  constexpr uint32_t kRequest = 64;
  constexpr uint32_t kInput = 1024;
  constexpr uint32_t kSecondary = 1056;
  constexpr uint32_t kOutput = 1088;
  auto* request = At<opennpux_npu_functional_request>(&memory, kRequest);
  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = OPENNPUX_NPU_OP_ADD;
  request->rows = 1;
  request->features = 4;
  AddOperand(request, OPENNPUX_NPU_OPERAND_INPUT, kInput, 4 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_SECONDARY, kSecondary,
             4 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT, kOutput,
             4 * sizeof(float));
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float secondary[] = {10.0f, 20.0f, 30.0f, 40.0f};
  std::memcpy(memory.data() + kInput, input, sizeof(input));
  std::memcpy(memory.data() + kSecondary, secondary, sizeof(secondary));

  assert(ExecuteGem5FunctionalRequest(request, memory.data(), kBase,
                                     memory.size()));
  const float* output = At<float>(&memory, kOutput);
  assert(output[0] == 11.0f && output[3] == 44.0f);
  assert(request->state == CORAL_OPERATOR_STATE_COMPLETE);
  assert(request->operation_count == 4);

  std::memset(memory.data() + kOutput, 0, sizeof(input));
  request->state = 0;
  request->operation_count = 0;
  auto descriptor = Descriptor(kRequest, OPENNPUX_NPU_OP_ADD);
  assert(DispatchGem5GenericCommand(&descriptor, memory.data(), kBase,
                                    memory.size()));
  assert(output[0] == 11.0f && output[3] == 44.0f);
  assert(request->state == CORAL_OPERATOR_STATE_COMPLETE);
  assert(request->operation_count == 4);
  assert(descriptor.operation_count == request->operation_count);
  assert(descriptor.bytes_written == sizeof(input));
}

void TestGptqMatMul() {
  std::vector<uint8_t> memory(4096);
  constexpr uint32_t kRequest = 64;
  constexpr uint32_t kParameters = 768;
  constexpr uint32_t kInput = 1024;
  constexpr uint32_t kQweight = 1056;
  constexpr uint32_t kQzeros = 1088;
  constexpr uint32_t kScales = 1120;
  constexpr uint32_t kOutput = 1152;
  auto* request = At<opennpux_npu_functional_request>(&memory, kRequest);
  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = OPENNPUX_NPU_OP_MATMUL;
  request->rows = 1;
  request->parameter_address = kBase + kParameters;
  request->parameter_size = sizeof(opennpux_npu_operator_parameters);
  AddOperand(request, OPENNPUX_NPU_OPERAND_INPUT, kInput, 2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_QWEIGHT, kQweight,
             sizeof(uint32_t));
  AddOperand(request, OPENNPUX_NPU_OPERAND_QZEROS, kQzeros,
             sizeof(uint32_t));
  AddOperand(request, OPENNPUX_NPU_OPERAND_SCALES, kScales, sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT, kOutput, sizeof(float));

  auto* parameters = At<opennpux_npu_operator_parameters>(&memory, kParameters);
  *parameters = {};
  parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters->struct_size = sizeof(*parameters);
  parameters->opcode = OPENNPUX_NPU_OP_MATMUL;
  parameters->flags = OPENNPUX_NPU_PARAMETER_GPTQ;
  parameters->input_features = 2;
  parameters->output_features = 1;
  parameters->quantization_bits = 4;
  parameters->quantization_group_size = 2;
  parameters->scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
  const float input[] = {2.0f, 3.0f};
  const uint32_t qweight = UINT32_C(0x21);
  const uint32_t qzeros = 0;
  const float scale = 0.5f;
  std::memcpy(memory.data() + kInput, input, sizeof(input));
  std::memcpy(memory.data() + kQweight, &qweight, sizeof(qweight));
  std::memcpy(memory.data() + kQzeros, &qzeros, sizeof(qzeros));
  std::memcpy(memory.data() + kScales, &scale, sizeof(scale));

  auto descriptor = Descriptor(kRequest, OPENNPUX_NPU_OP_MATMUL);
  assert(DispatchGem5GenericCommand(&descriptor, memory.data(), kBase,
                                    memory.size()));
  assert(*At<float>(&memory, kOutput) == 4.0f);
  assert(request->operation_count == 4);
}

void TestEmbedding() {
  std::vector<uint8_t> memory(4096);
  constexpr uint32_t kRequest = 64;
  constexpr uint32_t kTokenIds = 1024;
  constexpr uint32_t kTable = 1056;
  constexpr uint32_t kOutput = 1120;
  auto* request = At<opennpux_npu_functional_request>(&memory, kRequest);
  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = OPENNPUX_NPU_OP_EMBED;
  request->rows = 2;
  request->features = 2;
  request->vocabulary_size = 3;
  AddOperand(request, OPENNPUX_NPU_OPERAND_INPUT_INDICES, kTokenIds,
             2 * sizeof(uint32_t));
  AddOperand(request, OPENNPUX_NPU_OPERAND_WEIGHT, kTable,
             6 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT, kOutput,
             4 * sizeof(float));
  const uint32_t token_ids[] = {2, 0};
  const float table[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::memcpy(memory.data() + kTokenIds, token_ids, sizeof(token_ids));
  std::memcpy(memory.data() + kTable, table, sizeof(table));

  auto descriptor = Descriptor(kRequest, OPENNPUX_NPU_OP_EMBED);
  assert(DispatchGem5GenericCommand(&descriptor, memory.data(), kBase,
                                    memory.size()));
  const float* output = At<float>(&memory, kOutput);
  assert(output[0] == 5.0f && output[1] == 6.0f &&
         output[2] == 1.0f && output[3] == 2.0f);
  assert(request->operation_count == 4);
}

void TestDenseMatMul() {
  std::vector<uint8_t> memory(4096);
  constexpr uint32_t kRequest = 64;
  constexpr uint32_t kParameters = 768;
  constexpr uint32_t kInput = 1024;
  constexpr uint32_t kWeight = 1056;
  constexpr uint32_t kOutput = 1120;
  auto* request = At<opennpux_npu_functional_request>(&memory, kRequest);
  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = OPENNPUX_NPU_OP_MATMUL;
  request->rows = 1;
  request->parameter_address = kBase + kParameters;
  request->parameter_size = sizeof(opennpux_npu_operator_parameters);
  AddOperand(request, OPENNPUX_NPU_OPERAND_INPUT, kInput, 2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_WEIGHT, kWeight,
             4 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT, kOutput,
             2 * sizeof(float));

  auto* parameters = At<opennpux_npu_operator_parameters>(&memory, kParameters);
  *parameters = {};
  parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters->struct_size = sizeof(*parameters);
  parameters->opcode = OPENNPUX_NPU_OP_MATMUL;
  parameters->input_features = 2;
  parameters->output_features = 2;
  const float input[] = {2.0f, 3.0f};
  const float weight[] = {1.0f, 2.0f, 3.0f, 4.0f};
  std::memcpy(memory.data() + kInput, input, sizeof(input));
  std::memcpy(memory.data() + kWeight, weight, sizeof(weight));

  auto descriptor = Descriptor(kRequest, OPENNPUX_NPU_OP_MATMUL);
  assert(DispatchGem5GenericCommand(&descriptor, memory.data(), kBase,
                                    memory.size()));
  const float* output = At<float>(&memory, kOutput);
  assert(output[0] == 11.0f && output[1] == 16.0f);
  assert(request->operation_count == 8);
}

void TestDiscontiguousAddressSpace() {
  constexpr uint32_t kSubmissionBase = UINT32_C(0x24000000);
  constexpr uint32_t kArenaBase = UINT32_C(0x30000000);
  std::vector<uint8_t> submission(256);
  std::vector<uint8_t> arena(256);
  opennpux_npu_functional_request request = {};
  request.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request.struct_size = sizeof(request);
  request.opcode = OPENNPUX_NPU_OP_ADD;
  request.rows = 1;
  request.features = 4;
  request.parameter_address = kSubmissionBase;
  request.parameter_size = sizeof(opennpux_npu_operator_parameters);
  auto add_absolute = [&request](uint32_t role, uint32_t address,
                                 uint32_t size) {
    auto& operand = request.operands[request.operand_count++];
    operand.role = role;
    operand.address = address;
    operand.byte_size = size;
  };
  add_absolute(OPENNPUX_NPU_OPERAND_INPUT, kArenaBase,
               4 * sizeof(float));
  add_absolute(OPENNPUX_NPU_OPERAND_SECONDARY, kArenaBase + 32,
               4 * sizeof(float));
  add_absolute(OPENNPUX_NPU_OPERAND_OUTPUT, kArenaBase + 64,
               4 * sizeof(float));
  auto* parameters = reinterpret_cast<opennpux_npu_operator_parameters*>(
      submission.data());
  parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters->struct_size = sizeof(*parameters);
  parameters->opcode = OPENNPUX_NPU_OP_ADD;
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float secondary[] = {4.0f, 3.0f, 2.0f, 1.0f};
  std::memcpy(arena.data(), input, sizeof(input));
  std::memcpy(arena.data() + 32, secondary, sizeof(secondary));
  const Gem5FunctionalMemoryRegion regions[] = {
      {kSubmissionBase, submission.data(), submission.size()},
      {kArenaBase, arena.data(), arena.size()},
  };
  assert(ExecuteGem5FunctionalRequestInRegions(
      &request, regions, sizeof(regions) / sizeof(regions[0])));
  const float* output = reinterpret_cast<const float*>(arena.data() + 64);
  assert(output[0] == 5.0f && output[3] == 5.0f);
  assert(request.state == CORAL_OPERATOR_STATE_COMPLETE);

  request.state = 0;
  request.operands[0].address = UINT32_C(0x28000000);
  assert(!ExecuteGem5FunctionalRequestInRegions(
      &request, regions, sizeof(regions) / sizeof(regions[0])));
  assert(request.error == CORAL_OPERATOR_ERROR_ADDRESS);

  const Gem5FunctionalMemoryRegion overlapping[] = {
      {kArenaBase, arena.data(), arena.size()},
      {kArenaBase + 64, submission.data(), submission.size()},
  };
  assert(!ExecuteGem5FunctionalRequestInRegions(
      &request, overlapping, sizeof(overlapping) / sizeof(overlapping[0])));
}

void TestRejectsOpcodeMismatch() {
  std::vector<uint8_t> memory(1024);
  constexpr uint32_t kRequest = 64;
  auto* request = At<opennpux_npu_functional_request>(&memory, kRequest);
  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = OPENNPUX_NPU_OP_ADD;
  auto descriptor = Descriptor(kRequest, OPENNPUX_NPU_OP_MUL);
  assert(!DispatchGem5GenericCommand(&descriptor, memory.data(), kBase,
                                     memory.size()));
  assert(descriptor.error == CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR);
}

}  // namespace

int main() {
  TestAdd();
  TestEmbedding();
  TestDenseMatMul();
  TestDiscontiguousAddressSpace();
  TestGptqMatMul();
  TestRejectsOpcodeMismatch();
  std::puts("gem5_generic_command_dispatch=PASS");
  return 0;
}
