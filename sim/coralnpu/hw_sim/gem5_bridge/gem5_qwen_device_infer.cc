#include <cstdint>

#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/coral_operator_client.h"
#include "hw_sim/gem5_bridge/qwen_device_inference.h"

namespace {

constexpr uint32_t kSharedRequestAddress = UINT32_C(0x20000000);
constexpr uint32_t kExtmemBase = UINT32_C(0x20000000);
constexpr uint32_t kDescriptorAddress =
    kExtmemBase + CORAL_OPERATOR_DESCRIPTOR_OFFSET;

}  // namespace

int main() {
  auto* request = reinterpret_cast<volatile opennpux_qwen_device_request*>(
      static_cast<uintptr_t>(kSharedRequestAddress));
  auto* descriptor = reinterpret_cast<coral_operator_descriptor*>(
      static_cast<uintptr_t>(kDescriptorAddress));
  if (request->magic != OPENNPUX_QWEN_DEVICE_MAGIC ||
      request->version != OPENNPUX_QWEN_DEVICE_VERSION ||
      request->struct_size != sizeof(*request) ||
      request->state != OPENNPUX_QWEN_DEVICE_PENDING) {
    request->error = CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR;
    request->state = OPENNPUX_QWEN_DEVICE_ERROR;
    return 1;
  }

  opennpux::InitializeOperatorDescriptor(
      descriptor, CORAL_OPERATOR_OP_QWEN_TINY_INFER,
      CORAL_OPERATOR_MODE_HYBRID);
  const uint32_t dimensions[CORAL_OPERATOR_MAX_DIMS] = {
      sizeof(*request), 0, 0, 0};
  if (!opennpux::SetOperatorTensor(
          descriptor, 0, kSharedRequestAddress, sizeof(*request), 1,
          dimensions, CORAL_OPERATOR_ELEMENT_INT8, 0) ||
      !opennpux::SubmitHybridOperator(descriptor, kDescriptorAddress)) {
    request->error = descriptor->error == CORAL_OPERATOR_ERROR_NONE
        ? CORAL_OPERATOR_ERROR_EXECUTION
        : descriptor->error;
    request->state = OPENNPUX_QWEN_DEVICE_ERROR;
    return 1;
  }
  return request->state == OPENNPUX_QWEN_DEVICE_COMPLETE ? 0 : 1;
}
