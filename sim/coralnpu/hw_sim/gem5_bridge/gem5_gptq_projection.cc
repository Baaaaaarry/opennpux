#include <cstdint>

#include "hw_sim/gem5_bridge/coral_gptq_matmul.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/coral_operator_client.h"

namespace {

constexpr uint32_t kExtmemBase = UINT32_C(0x20000000);
constexpr uint32_t kRequestAddress = kExtmemBase;
constexpr uint32_t kDescriptorAddress =
    kExtmemBase + CORAL_OPERATOR_DESCRIPTOR_OFFSET;

}  // namespace

int main() {
  auto* request = reinterpret_cast<volatile coral_gptq_matmul_request*>(
      static_cast<uintptr_t>(kRequestAddress));
  auto* descriptor = reinterpret_cast<coral_operator_descriptor*>(
      static_cast<uintptr_t>(kDescriptorAddress));
  if (request->magic != CORAL_GPTQ_MATMUL_MAGIC ||
      request->version != CORAL_GPTQ_MATMUL_VERSION ||
      request->struct_size != sizeof(*request) ||
      request->state != CORAL_GPTQ_MATMUL_PENDING) {
    return 1;
  }

  opennpux::InitializeOperatorDescriptor(
      descriptor, CORAL_OPERATOR_OP_GPTQ_MATMUL_INT4,
      CORAL_OPERATOR_MODE_HYBRID);
  const uint32_t dimensions[CORAL_OPERATOR_MAX_DIMS] = {
      sizeof(*request), 0, 0, 0};
  if (!opennpux::SetOperatorTensor(
          descriptor, 0, kRequestAddress, sizeof(*request), 1, dimensions,
          CORAL_OPERATOR_ELEMENT_INT8, 0) ||
      !opennpux::SubmitHybridOperator(descriptor, kDescriptorAddress)) {
    return 1;
  }
  return request->state == CORAL_GPTQ_MATMUL_COMPLETE &&
                 request->error == CORAL_OPERATOR_ERROR_NONE &&
                 request->operations != 0 && request->output_checksum != 0
             ? 0
             : 1;
}
