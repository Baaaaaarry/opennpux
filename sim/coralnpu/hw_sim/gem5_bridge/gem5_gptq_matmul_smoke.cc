#include <cstdint>

#include "hw_sim/gem5_bridge/coral_gptq_matmul.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/coral_operator_client.h"

namespace {

constexpr uint32_t kExtmemBase = UINT32_C(0x20000000);
constexpr uint32_t kRequestAddress = kExtmemBase;
constexpr uint32_t kInputAddress = kExtmemBase + UINT32_C(0x100);
constexpr uint32_t kQweightAddress = kExtmemBase + UINT32_C(0x120);
constexpr uint32_t kQzerosAddress = kExtmemBase + UINT32_C(0x140);
constexpr uint32_t kScalesAddress = kExtmemBase + UINT32_C(0x160);
constexpr uint32_t kOutputAddress = kExtmemBase + UINT32_C(0x180);
constexpr uint32_t kDescriptorAddress =
    kExtmemBase + CORAL_OPERATOR_DESCRIPTOR_OFFSET;

}  // namespace

int main() {
  auto* request = reinterpret_cast<volatile coral_gptq_matmul_request*>(
      static_cast<uintptr_t>(kRequestAddress));
  auto* descriptor = reinterpret_cast<coral_operator_descriptor*>(
      static_cast<uintptr_t>(kDescriptorAddress));
  auto* input = reinterpret_cast<volatile float*>(
      static_cast<uintptr_t>(kInputAddress));
  auto* qweight = reinterpret_cast<volatile uint32_t*>(
      static_cast<uintptr_t>(kQweightAddress));
  auto* qzeros = reinterpret_cast<volatile uint32_t*>(
      static_cast<uintptr_t>(kQzerosAddress));
  auto* scales = reinterpret_cast<volatile float*>(
      static_cast<uintptr_t>(kScalesAddress));
  auto* output = reinterpret_cast<volatile float*>(
      static_cast<uintptr_t>(kOutputAddress));

  input[0] = 2.0f;
  input[1] = 3.0f;
  qweight[0] = UINT32_C(0x21);
  qzeros[0] = 0;
  scales[0] = 0.5f;
  output[0] = 0.0f;

  request->magic = CORAL_GPTQ_MATMUL_MAGIC;
  request->version = CORAL_GPTQ_MATMUL_VERSION;
  request->struct_size = sizeof(*request);
  request->state = CORAL_GPTQ_MATMUL_PENDING;
  request->error = CORAL_OPERATOR_ERROR_NONE;
  request->rows = 1;
  request->input_columns = 2;
  request->output_columns = 1;
  request->group_size = 2;
  request->zero_bias = 0;
  request->input_address = kInputAddress;
  request->qweight_address = kQweightAddress;
  request->qzeros_address = kQzerosAddress;
  request->scales_address = kScalesAddress;
  request->g_idx_address = 0;
  request->output_address = kOutputAddress;
  request->scale_data_type = CORAL_GPTQ_SCALE_FLOAT32;

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
                 request->operations == 4 && output[0] == 4.0f
             ? 0
             : 1;
}
