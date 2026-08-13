#include <cstdint>

#include "hw_sim/gem5_bridge/coral_mobilenet.h"
#include "hw_sim/gem5_bridge/gem5_npu_launch_descriptor.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/coral_operator_client.h"

namespace {

constexpr uint32_t kExtmemBase = UINT32_C(0x20000000);
constexpr uint32_t kMailboxAddress =
    kExtmemBase + OPENNPUX_CORAL_MOBILENET_MAILBOX_OFFSET;
constexpr uint32_t kDescriptorAddress =
    kExtmemBase + CORAL_OPERATOR_DESCRIPTOR_OFFSET;
constexpr uint32_t kInput0Address =
    kExtmemBase + CORAL_OPERATOR_STAGING_OFFSET;
constexpr uint32_t kInput1Address = kInput0Address + UINT32_C(0x10);
constexpr uint32_t kOutputAddress = kInput1Address + UINT32_C(0x10);
constexpr uint32_t kElements = 4;

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
  auto* mailbox = reinterpret_cast<volatile opennpux_coral_mobilenet_mailbox*>(
      static_cast<uintptr_t>(kMailboxAddress));
  auto* descriptor = reinterpret_cast<coral_operator_descriptor*>(
      static_cast<uintptr_t>(kDescriptorAddress));
  auto* input0 = reinterpret_cast<volatile int8_t*>(
      static_cast<uintptr_t>(kInput0Address));
  auto* input1 = reinterpret_cast<volatile int8_t*>(
      static_cast<uintptr_t>(kInput1Address));
  auto* output = reinterpret_cast<volatile int8_t*>(
      static_cast<uintptr_t>(kOutputAddress));

  mailbox->magic = OPENNPUX_CORAL_MOBILENET_MAGIC;
  mailbox->version = OPENNPUX_CORAL_MOBILENET_VERSION;
  mailbox->state = OPENNPUX_CORAL_MOBILENET_STARTED;
  mailbox->error_code = OPENNPUX_CORAL_MOBILENET_ERROR_NONE;

  const int8_t lhs[kElements] = {1, 2, 3, 4};
  const int8_t rhs[kElements] = {10, 20, 30, 40};
  for (uint32_t i = 0; i < kElements; ++i) {
    input0[i] = lhs[i];
    input1[i] = rhs[i];
    output[i] = 0;
  }

  opennpux::InitializeNpuLaunchAddDescriptor(
      descriptor, kInput0Address, kInput1Address, kOutputAddress, kElements);

  if (!opennpux::SubmitHybridOperator(descriptor, kDescriptorAddress)) {
    mailbox->state = OPENNPUX_CORAL_MOBILENET_ERROR;
    mailbox->error_code = descriptor->error;
    return 1;
  }

  const int8_t expected[kElements] = {11, 22, 33, 44};
  for (uint32_t i = 0; i < kElements; ++i) {
    if (output[i] != expected[i]) {
      mailbox->state = OPENNPUX_CORAL_MOBILENET_ERROR;
      mailbox->error_code = OPENNPUX_CORAL_MOBILENET_ERROR_OUTPUT;
      return 1;
    }
    mailbox->output[i] = output[i];
  }
  // The existing mobilenet-test result ABI has five output slots. Use the
  // final slot to expose the operator error code for this focused smoke test.
  mailbox->output[kElements] = static_cast<int32_t>(descriptor->error);
  mailbox->output_count = OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT;
  mailbox->output_bytes =
      OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT * sizeof(mailbox->output[0]);
  mailbox->output_checksum = Fnv1a32(
      reinterpret_cast<const volatile uint8_t*>(mailbox->output),
      mailbox->output_bytes);
  mailbox->operation_count = descriptor->operation_count;
  mailbox->bytes_read = descriptor->bytes_read;
  mailbox->bytes_written = descriptor->bytes_written;
  mailbox->cycle_low = static_cast<uint32_t>(descriptor->modeled_cycles);
  mailbox->cycle_high =
      static_cast<uint32_t>(descriptor->modeled_cycles >> 32);
  mailbox->state = OPENNPUX_CORAL_MOBILENET_COMPLETE;
  return 0;
}
