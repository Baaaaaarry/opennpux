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
constexpr uint32_t kInput0Address =
    kExtmemBase + CORAL_OPERATOR_STAGING_OFFSET;
constexpr uint32_t kInput1Address = kInput0Address + UINT32_C(0x40);
constexpr uint32_t kOutputAddress = kInput1Address + UINT32_C(0x40);
constexpr uint32_t kElements = 4;

void AddOperand(opennpux_npu_functional_request* request, uint32_t role,
                uint32_t address, uint32_t byte_size) {
  opennpux_npu_functional_operand& operand =
      request->operands[request->operand_count++];
  operand.role = role;
  operand.address = address;
  operand.byte_size = byte_size;
  operand.reserved = 0;
}

uint32_t Fnv1a32(const volatile uint8_t* data, uint32_t bytes) {
  uint32_t hash = UINT32_C(2166136261);
  for (uint32_t i = 0; i < bytes; ++i) {
    hash ^= data[i];
    hash *= UINT32_C(16777619);
  }
  return hash;
}

}  // namespace

int main() {
  auto* mailbox = reinterpret_cast<
      volatile opennpux_coral_generic_test_mailbox*>(
      static_cast<uintptr_t>(kMailboxAddress));
  auto* descriptor = reinterpret_cast<coral_operator_descriptor*>(
      static_cast<uintptr_t>(kDescriptorAddress));
  auto* request = reinterpret_cast<opennpux_npu_functional_request*>(
      static_cast<uintptr_t>(kRequestAddress));
  auto* input0 = reinterpret_cast<volatile float*>(
      static_cast<uintptr_t>(kInput0Address));
  auto* input1 = reinterpret_cast<volatile float*>(
      static_cast<uintptr_t>(kInput1Address));
  auto* output = reinterpret_cast<volatile float*>(
      static_cast<uintptr_t>(kOutputAddress));

  mailbox->magic = OPENNPUX_CORAL_GENERIC_TEST_MAGIC;
  mailbox->version = OPENNPUX_CORAL_GENERIC_TEST_VERSION;
  mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_STARTED;
  mailbox->error_code = OPENNPUX_CORAL_GENERIC_TEST_ERROR_NONE;
  const float lhs[kElements] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float rhs[kElements] = {10.0f, 20.0f, 30.0f, 40.0f};
  for (uint32_t index = 0; index < kElements; ++index) {
    input0[index] = lhs[index];
    input1[index] = rhs[index];
    output[index] = 0.0f;
  }

  *request = {};
  request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  request->struct_size = sizeof(*request);
  request->opcode = OPENNPUX_NPU_OP_ADD;
  request->command_id = 1;
  request->state = CORAL_OPERATOR_STATE_SUBMITTED;
  request->rows = 1;
  request->features = kElements;
  AddOperand(request, OPENNPUX_NPU_OPERAND_INPUT, kInput0Address,
             kElements * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_SECONDARY, kInput1Address,
             kElements * sizeof(float));
  AddOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT, kOutputAddress,
             kElements * sizeof(float));

  *descriptor = {};
  descriptor->magic = CORAL_OPERATOR_ABI_MAGIC;
  descriptor->version = CORAL_OPERATOR_ABI_VERSION;
  descriptor->descriptor_size = sizeof(*descriptor);
  descriptor->opcode = CORAL_OPERATOR_OP_GENERIC_COMMAND;
  descriptor->state = CORAL_OPERATOR_STATE_SUBMITTED;
  descriptor->flags = CORAL_OPERATOR_FLAG_CUSTOM_INSTRUCTION;
  descriptor->execution_mode = CORAL_OPERATOR_MODE_HYBRID;
  descriptor->tensor_count = 1;
  descriptor->tensors[0].address = kRequestAddress;
  descriptor->tensors[0].size = sizeof(*request);
  descriptor->tensors[0].rank = 1;
  descriptor->tensors[0].dimensions[0] = sizeof(*request);
  descriptor->tensors[0].element_type = CORAL_OPERATOR_ELEMENT_INT8;
  descriptor->reserved[0] = OPENNPUX_NPU_OP_ADD;

  if (!opennpux::SubmitHybridOperator(descriptor, kDescriptorAddress)) {
    mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_ERROR;
    mailbox->error_code = descriptor->error != 0 ?
        descriptor->error : OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT;
    return 1;
  }

  const int32_t expected[kElements] = {11, 22, 33, 44};
  for (uint32_t index = 0; index < kElements; ++index) {
    if (static_cast<int32_t>(output[index]) != expected[index]) {
      mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_ERROR;
      mailbox->error_code = OPENNPUX_CORAL_GENERIC_TEST_ERROR_OUTPUT;
      return 1;
    }
    mailbox->output[index] = static_cast<int32_t>(output[index]);
  }
  mailbox->output_count = OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT;
  mailbox->output_bytes =
      OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT * sizeof(mailbox->output[0]);
  mailbox->output_checksum = Fnv1a32(
      reinterpret_cast<const volatile uint8_t*>(mailbox->output),
      mailbox->output_bytes);
  mailbox->operation_count = request->operation_count;
  mailbox->bytes_read = request->bytes_read;
  mailbox->bytes_written = request->bytes_written;
  mailbox->cycle_low = static_cast<uint32_t>(request->modeled_cycles);
  mailbox->cycle_high = static_cast<uint32_t>(request->modeled_cycles >> 32);
  mailbox->state = OPENNPUX_CORAL_GENERIC_TEST_COMPLETE;
  return 0;
}
