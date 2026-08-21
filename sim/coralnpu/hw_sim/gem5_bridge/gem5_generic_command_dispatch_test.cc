#include "hw_sim/gem5_bridge/gem5_generic_command_dispatch.h"

#include <cassert>
#include <cmath>
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

void TestGptqExpert() {
  std::vector<uint8_t> memory(8192);
  constexpr uint32_t kRequest = 64;
  constexpr uint32_t kParameters = 768;
  constexpr uint32_t kInput = 1024;
  constexpr uint32_t kOutput = 1088;
  constexpr uint32_t kGateOutput = 1120;
  constexpr uint32_t kUpOutput = 1152;
  constexpr uint32_t kActivated = 1184;
  constexpr uint32_t kGate = 2048;
  constexpr uint32_t kUp = 2176;
  constexpr uint32_t kDown = 2304;
  auto* request = At<opennpux_npu_functional_request>(&memory, kRequest);
  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = OPENNPUX_NPU_OP_EXPERT;
  request->rows = 1;
  request->features = 2;
  request->parameter_address = kBase + kParameters;
  request->parameter_size = sizeof(opennpux_npu_operator_parameters);
  AddOperand(request, OPENNPUX_NPU_OPERAND_INPUT, kInput, 2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT, kOutput, 2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_GATE_OUTPUT, kGateOutput,
             2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_UP_OUTPUT, kUpOutput,
             2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_ACTIVATED, kActivated,
             2 * sizeof(float));
  const auto add_projection = [request](uint32_t base, uint32_t qweight_role,
                                         uint32_t qzeros_role,
                                         uint32_t scales_role) {
    AddOperand(request, qweight_role, base, 2 * sizeof(uint32_t));
    AddOperand(request, qzeros_role, base + 32, sizeof(uint32_t));
    AddOperand(request, scales_role, base + 64, 2 * sizeof(float));
  };
  add_projection(kGate, OPENNPUX_NPU_OPERAND_GATE_QWEIGHT,
                 OPENNPUX_NPU_OPERAND_GATE_QZEROS,
                 OPENNPUX_NPU_OPERAND_GATE_SCALES);
  add_projection(kUp, OPENNPUX_NPU_OPERAND_UP_QWEIGHT,
                 OPENNPUX_NPU_OPERAND_UP_QZEROS,
                 OPENNPUX_NPU_OPERAND_UP_SCALES);
  add_projection(kDown, OPENNPUX_NPU_OPERAND_DOWN_QWEIGHT,
                 OPENNPUX_NPU_OPERAND_DOWN_QZEROS,
                 OPENNPUX_NPU_OPERAND_DOWN_SCALES);

  auto* parameters = At<opennpux_npu_operator_parameters>(&memory, kParameters);
  *parameters = {};
  parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters->struct_size = sizeof(*parameters);
  parameters->opcode = OPENNPUX_NPU_OP_EXPERT;
  parameters->flags = OPENNPUX_NPU_PARAMETER_GPTQ;
  parameters->input_features = 2;
  parameters->output_features = 2;
  parameters->intermediate_features = 2;
  parameters->quantization_bits = 4;
  parameters->quantization_group_size = 2;
  parameters->scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
  parameters->quantized_zero_bias = 1;
  const float input[] = {1.0f, 2.0f};
  std::memcpy(memory.data() + kInput, input, sizeof(input));
  for (uint32_t base : {kGate, kUp, kDown}) {
    const uint32_t qweight[] = {UINT32_C(0x22), UINT32_C(0x22)};
    const uint32_t qzeros = 0;
    const float scales[] = {1.0f, 1.0f};
    std::memcpy(memory.data() + base, qweight, sizeof(qweight));
    std::memcpy(memory.data() + base + 32, &qzeros, sizeof(qzeros));
    std::memcpy(memory.data() + base + 64, scales, sizeof(scales));
  }
  assert(ExecuteGem5FunctionalRequest(request, memory.data(), kBase,
                                      memory.size()));
  const float* output = At<float>(&memory, kOutput);
  assert(output[0] > 0.0f && output[0] == output[1]);
  assert(request->operation_count != 0);
}

void TestFloatQkv() {
  std::vector<uint8_t> memory(8192);
  constexpr uint32_t kRequest = 64;
  constexpr uint32_t kParameters = 768;
  constexpr uint32_t kInput = 1024;
  constexpr uint32_t kQuery = 1056;
  constexpr uint32_t kKey = 1088;
  constexpr uint32_t kValue = 1120;
  constexpr uint32_t kGate = 1152;
  constexpr uint32_t kQWeight = 2048;
  constexpr uint32_t kKWeight = 2112;
  constexpr uint32_t kVWeight = 2144;
  constexpr uint32_t kQNorm = 2176;
  constexpr uint32_t kKNorm = 2208;
  auto* request = At<opennpux_npu_functional_request>(&memory, kRequest);
  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = OPENNPUX_NPU_OP_MATMUL;
  request->rows = 1;
  request->features = 2;
  request->heads = 1;
  request->kv_heads = 1;
  request->head_dim = 2;
  request->epsilon = 0.0f;
  request->parameter_address = kBase + kParameters;
  request->parameter_size = sizeof(opennpux_npu_operator_parameters);
  AddOperand(request, OPENNPUX_NPU_OPERAND_INPUT, kInput, 2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT, kQuery, 2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY, kKey,
             2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY, kValue,
             2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT_QUATERNARY, kGate,
             2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_ATTENTION_Q_WEIGHT, kQWeight,
             8 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_ATTENTION_K_WEIGHT, kKWeight,
             4 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_ATTENTION_V_WEIGHT, kVWeight,
             4 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_ATTENTION_Q_NORM_WEIGHT, kQNorm,
             2 * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_ATTENTION_K_NORM_WEIGHT, kKNorm,
             2 * sizeof(float));
  auto* parameters = At<opennpux_npu_operator_parameters>(&memory, kParameters);
  *parameters = {};
  parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters->struct_size = sizeof(*parameters);
  parameters->opcode = OPENNPUX_NPU_OP_MATMUL;
  parameters->input_features = 2;
  const float input[] = {1.0f, 2.0f};
  const float q_weight[] = {1.0f, 0.0f, 0.0f, 1.0f,
                            10.0f, 10.0f, 10.0f, 10.0f};
  const float identity[] = {1.0f, 0.0f, 0.0f, 1.0f};
  const float norm[] = {1.0f, 1.0f};
  std::memcpy(memory.data() + kInput, input, sizeof(input));
  std::memcpy(memory.data() + kQWeight, q_weight, sizeof(q_weight));
  std::memcpy(memory.data() + kKWeight, identity, sizeof(identity));
  std::memcpy(memory.data() + kVWeight, identity, sizeof(identity));
  std::memcpy(memory.data() + kQNorm, norm, sizeof(norm));
  std::memcpy(memory.data() + kKNorm, norm, sizeof(norm));
  assert(ExecuteGem5FunctionalRequest(request, memory.data(), kBase,
                                      memory.size()));
  const float* query = At<float>(&memory, kQuery);
  const float* key = At<float>(&memory, kKey);
  const float* value = At<float>(&memory, kValue);
  const float* gate = At<float>(&memory, kGate);
  assert(std::fabs(query[0] - 0.6324555f) < 1.0e-5f);
  assert(std::fabs(key[1] - 1.2649110f) < 1.0e-5f);
  assert(value[0] == 1.0f && value[1] == 2.0f);
  assert(gate[0] == 30.0f && gate[1] == 30.0f);
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
  const float weight[] = {1.0f, 3.0f, 2.0f, 4.0f};
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
  TestGptqExpert();
  TestFloatQkv();
  TestRejectsOpcodeMismatch();
  std::puts("gem5_generic_command_dispatch=PASS");
  return 0;
}
