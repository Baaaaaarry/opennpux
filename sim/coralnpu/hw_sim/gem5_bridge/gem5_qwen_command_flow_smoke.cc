#include <cstdint>

#include "hw_sim/gem5_bridge/coral_generic_test.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/coral_operator_client.h"
#include "hw_sim/gem5_bridge/npu_functional_request.h"
#include "hw_sim/gem5_bridge/npu_submission.h"

namespace {

constexpr uint32_t kExtmemBase = UINT32_C(0x20000000);
constexpr uint32_t kMailboxAddress =
    kExtmemBase + OPENNPUX_CORAL_GENERIC_TEST_MAILBOX_OFFSET;
constexpr uint32_t kDescriptorAddress =
    kExtmemBase + CORAL_OPERATOR_DESCRIPTOR_OFFSET;
constexpr uint32_t kRequestAddress = kDescriptorAddress + UINT32_C(0x400);
constexpr uint32_t kParametersAddress = kRequestAddress + UINT32_C(0x200);
constexpr uint32_t kDataBase = kExtmemBase + CORAL_OPERATOR_STAGING_OFFSET;
constexpr uint32_t kTokenIds = kDataBase;
constexpr uint32_t kPositions = kTokenIds + UINT32_C(0x40);
constexpr uint32_t kEmbeddingTable = kPositions + UINT32_C(0x40);
constexpr uint32_t kMatrix = kEmbeddingTable + UINT32_C(0x100);
constexpr uint32_t kBias = kMatrix + UINT32_C(0x100);
constexpr uint32_t kScale = kBias + UINT32_C(0x40);
constexpr uint32_t kNormWeight = kScale + UINT32_C(0x40);
constexpr uint32_t kTensor0 = kNormWeight + UINT32_C(0x40);
constexpr uint32_t kTensor1 = kTensor0 + UINT32_C(0x80);
constexpr uint32_t kTensor2 = kTensor1 + UINT32_C(0x80);
constexpr uint32_t kTensor3 = kTensor2 + UINT32_C(0x80);
constexpr uint32_t kTensor4 = kTensor3 + UINT32_C(0x80);
constexpr uint32_t kTensor5 = kTensor4 + UINT32_C(0x80);
constexpr uint32_t kTensor6 = kTensor5 + UINT32_C(0x80);
constexpr uint32_t kTopValue = kTensor6 + UINT32_C(0x80);
constexpr uint32_t kTopIndex = kTopValue + UINT32_C(0x10);
constexpr uint32_t kRows = 2;
constexpr uint32_t kFeatures = 4;
constexpr uint32_t kElementCount = kRows * kFeatures;

volatile opennpux_coral_generic_test_mailbox* Mailbox() {
  return reinterpret_cast<volatile opennpux_coral_generic_test_mailbox*>(
      static_cast<uintptr_t>(kMailboxAddress));
}

opennpux_npu_functional_request* Request() {
  return reinterpret_cast<opennpux_npu_functional_request*>(
      static_cast<uintptr_t>(kRequestAddress));
}

void AddOperand(opennpux_npu_functional_request* request, uint32_t role,
                uint32_t address, uint32_t byte_size) {
  opennpux_npu_functional_operand& operand =
      request->operands[request->operand_count++];
  operand.role = role;
  operand.address = address;
  operand.byte_size = byte_size;
  operand.reserved = 0;
}

void InitializeRequest(uint32_t opcode, uint32_t command_id) {
  opennpux_npu_functional_request* request = Request();
  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = opcode;
  request->command_id = command_id;
  request->state = CORAL_OPERATOR_STATE_SUBMITTED;
}

bool Submit(uint32_t opcode, uint64_t* operations, uint64_t* bytes_read,
            uint64_t* bytes_written, uint64_t* cycles) {
  auto* descriptor = reinterpret_cast<coral_operator_descriptor*>(
      static_cast<uintptr_t>(kDescriptorAddress));
  opennpux::InitializeOperatorDescriptor(
      descriptor, CORAL_OPERATOR_OP_GENERIC_COMMAND,
      CORAL_OPERATOR_MODE_HYBRID);
  const uint32_t dimensions[CORAL_OPERATOR_MAX_DIMS] = {
      sizeof(opennpux_npu_functional_request), 0, 0, 0};
  descriptor->reserved[0] = opcode;
  if (!opennpux::SetOperatorTensor(
          descriptor, 0, kRequestAddress,
          sizeof(opennpux_npu_functional_request), 1, dimensions,
          CORAL_OPERATOR_ELEMENT_INT8, 0) ||
      !opennpux::SubmitHybridOperator(descriptor, kDescriptorAddress)) {
    return false;
  }
  const opennpux_npu_functional_request* request = Request();
  if (request->state != CORAL_OPERATOR_STATE_COMPLETE ||
      request->error != CORAL_OPERATOR_ERROR_NONE) {
    return false;
  }
  *operations += request->operation_count;
  *bytes_read += request->bytes_read;
  *bytes_written += request->bytes_written;
  *cycles += request->modeled_cycles;
  return true;
}

uint32_t Fnv1a32(const volatile uint8_t* data, uint32_t bytes) {
  uint32_t hash = UINT32_C(2166136261);
  for (uint32_t index = 0; index < bytes; ++index) {
    hash ^= data[index];
    hash *= UINT32_C(16777619);
  }
  return hash;
}

int Fail(uint32_t command, uint32_t error) {
  volatile opennpux_coral_generic_test_mailbox* mailbox = Mailbox();
  mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_ERROR;
  mailbox->error_code = error;
  mailbox->output[0] = static_cast<int32_t>(command);
  return 1;
}

}  // namespace

int main() {
  volatile opennpux_coral_generic_test_mailbox* mailbox = Mailbox();
  mailbox->magic = OPENNPUX_CORAL_GENERIC_TEST_MAGIC;
  mailbox->version = OPENNPUX_CORAL_GENERIC_TEST_VERSION;
  mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_STARTED;
  mailbox->error_code = OPENNPUX_CORAL_GENERIC_TEST_ERROR_NONE;

  auto* token_ids = reinterpret_cast<volatile uint32_t*>(kTokenIds);
  auto* positions = reinterpret_cast<volatile uint32_t*>(kPositions);
  auto* embedding = reinterpret_cast<volatile float*>(kEmbeddingTable);
  auto* matrix = reinterpret_cast<volatile float*>(kMatrix);
  auto* bias = reinterpret_cast<volatile float*>(kBias);
  auto* scale = reinterpret_cast<volatile float*>(kScale);
  auto* norm_weight = reinterpret_cast<volatile float*>(kNormWeight);
  token_ids[0] = 0;
  token_ids[1] = 1;
  positions[0] = 0;
  positions[1] = 1;
  const float embedding_values[kElementCount] = {
      1.0f, 2.0f, 3.0f, 4.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  for (uint32_t index = 0; index < kElementCount; ++index) {
    embedding[index] = embedding_values[index];
    bias[index] = 1.0f;
    scale[index] = 0.5f;
  }
  for (uint32_t index = 0; index < kFeatures; ++index) {
    norm_weight[index] = 1.0f;
    for (uint32_t column = 0; column < kFeatures; ++column) {
      matrix[index * kFeatures + column] = index == column ? 1.0f : 0.0f;
    }
  }

  uint64_t operations = 0;
  uint64_t bytes_read = 0;
  uint64_t bytes_written = 0;
  uint64_t cycles = 0;
  uint32_t command = 0;

  InitializeRequest(OPENNPUX_NPU_OP_EMBED, ++command);
  Request()->rows = kRows;
  Request()->features = kFeatures;
  Request()->vocabulary_size = kRows;
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_INPUT_INDICES, kTokenIds,
             kRows * sizeof(uint32_t));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_WEIGHT, kEmbeddingTable,
             kElementCount * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT, kTensor0,
             kElementCount * sizeof(float));
  if (!Submit(OPENNPUX_NPU_OP_EMBED, &operations, &bytes_read,
              &bytes_written, &cycles)) {
    return Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT);
  }

  InitializeRequest(OPENNPUX_NPU_OP_MATMUL, ++command);
  Request()->rows = kRows;
  Request()->parameter_address = kParametersAddress;
  Request()->parameter_size = sizeof(opennpux_npu_operator_parameters);
  auto* parameters = reinterpret_cast<opennpux_npu_operator_parameters*>(
      static_cast<uintptr_t>(kParametersAddress));
  *parameters = {};
  parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  parameters->struct_size = sizeof(*parameters);
  parameters->opcode = OPENNPUX_NPU_OP_MATMUL;
  parameters->input_features = kFeatures;
  parameters->output_features = kFeatures;
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_INPUT, kTensor0,
             kElementCount * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_WEIGHT, kMatrix,
             kFeatures * kFeatures * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT, kTensor1,
             kElementCount * sizeof(float));
  if (!Submit(OPENNPUX_NPU_OP_MATMUL, &operations, &bytes_read,
              &bytes_written, &cycles)) {
    return Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT);
  }

  struct ElementwiseStep {
    uint32_t opcode;
    uint32_t input;
    uint32_t secondary;
    uint32_t output;
  };
  const ElementwiseStep elementwise[] = {
      {OPENNPUX_NPU_OP_ADD, kTensor1, kBias, kTensor2},
      {OPENNPUX_NPU_OP_MUL, kTensor2, kScale, kTensor3},
  };
  for (const ElementwiseStep& step : elementwise) {
    InitializeRequest(step.opcode, ++command);
    Request()->rows = kRows;
    Request()->features = kFeatures;
    AddOperand(Request(), OPENNPUX_NPU_OPERAND_INPUT, step.input,
               kElementCount * sizeof(float));
    AddOperand(Request(), OPENNPUX_NPU_OPERAND_SECONDARY, step.secondary,
               kElementCount * sizeof(float));
    AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT, step.output,
               kElementCount * sizeof(float));
    if (!Submit(step.opcode, &operations, &bytes_read, &bytes_written,
                &cycles)) {
      return Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT);
    }
  }

  InitializeRequest(OPENNPUX_NPU_OP_NORMALIZE, ++command);
  Request()->rows = kRows;
  Request()->features = kFeatures;
  Request()->epsilon = 1.0e-5f;
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_INPUT, kTensor3,
             kElementCount * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_WEIGHT, kNormWeight,
             kFeatures * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT, kTensor4,
             kElementCount * sizeof(float));
  if (!Submit(OPENNPUX_NPU_OP_NORMALIZE, &operations, &bytes_read,
              &bytes_written, &cycles)) {
    return Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT);
  }

  InitializeRequest(OPENNPUX_NPU_OP_ROPE, ++command);
  Request()->rows = kRows;
  Request()->features = kFeatures;
  Request()->heads = 2;
  Request()->head_dim = 2;
  Request()->rope_theta = 10000.0f;
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_INPUT, kTensor4,
             kElementCount * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_POSITIONS, kPositions,
             kRows * sizeof(uint32_t));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT, kTensor5,
             kElementCount * sizeof(float));
  if (!Submit(OPENNPUX_NPU_OP_ROPE, &operations, &bytes_read,
              &bytes_written, &cycles)) {
    return Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT);
  }

  InitializeRequest(OPENNPUX_NPU_OP_ACTIVATION, ++command);
  Request()->rows = kRows;
  Request()->features = kFeatures;
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_INPUT, kTensor5,
             kElementCount * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT, kTensor6,
             kElementCount * sizeof(float));
  if (!Submit(OPENNPUX_NPU_OP_ACTIVATION, &operations, &bytes_read,
              &bytes_written, &cycles)) {
    return Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT);
  }

  InitializeRequest(OPENNPUX_NPU_OP_SOFTMAX, ++command);
  Request()->rows = kRows;
  Request()->features = kFeatures;
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_INPUT, kTensor6,
             kElementCount * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT, kTensor0,
             kElementCount * sizeof(float));
  if (!Submit(OPENNPUX_NPU_OP_SOFTMAX, &operations, &bytes_read,
              &bytes_written, &cycles)) {
    return Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT);
  }

  InitializeRequest(OPENNPUX_NPU_OP_TOPK, ++command);
  Request()->rows = 1;
  Request()->features = kElementCount;
  Request()->top_k = 1;
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_INPUT, kTensor0,
             kElementCount * sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT, kTopValue,
             sizeof(float));
  AddOperand(Request(), OPENNPUX_NPU_OPERAND_OUTPUT_INDICES, kTopIndex,
             sizeof(uint32_t));
  if (!Submit(OPENNPUX_NPU_OP_TOPK, &operations, &bytes_read,
              &bytes_written, &cycles)) {
    return Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT);
  }

  const uint32_t top_index =
      *reinterpret_cast<volatile uint32_t*>(kTopIndex);
  const float top_value = *reinterpret_cast<volatile float*>(kTopValue);
  union {
    float value;
    int32_t bits;
  } encoded = {top_value};
  mailbox->output[0] = static_cast<int32_t>(command);
  mailbox->output[1] = static_cast<int32_t>(top_index);
  mailbox->output[2] = encoded.bits;
  mailbox->output[3] = 0;
  mailbox->output_count = OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT;
  mailbox->output_bytes = sizeof(mailbox->output);
  mailbox->output_checksum = Fnv1a32(
      reinterpret_cast<const volatile uint8_t*>(mailbox->output),
      mailbox->output_bytes);
  mailbox->operation_count = operations;
  mailbox->bytes_read = bytes_read;
  mailbox->bytes_written = bytes_written;
  mailbox->cycle_low = static_cast<uint32_t>(cycles);
  mailbox->cycle_high = static_cast<uint32_t>(cycles >> 32);
  mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_COMPLETE;
  // The synthetic graph is deterministic; checking the winning index catches
  // broken tensor chaining that a command-count-only test would miss.
  return command == 9 && top_index == 5 ? 0 :
      Fail(command, OPENNPUX_CORAL_GENERIC_TEST_ERROR_OUTPUT);
}
