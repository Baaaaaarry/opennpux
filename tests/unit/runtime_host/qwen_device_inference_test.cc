#include "opennpux/qwen_model.h"

#include <cassert>
#include <cstring>
#include <vector>

#include "hw_sim/gem5_bridge/gem5_hybrid_operator.h"

namespace {

constexpr uint32_t kBase = UINT32_C(0x20000000);

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  opennpux_qwen_device_request request = {};
  opennpux_qwen_model_info expected = {};
  assert(opennpux_qwen_build_device_request(
             argv[1], "open npux", &request, &expected) == 0);
  assert(request.state == OPENNPUX_QWEN_DEVICE_PENDING);
  assert(request.next_token == 0);
  assert(request.logits_checksum == 0);

  std::vector<uint8_t> memory(sizeof(request), 0);
  std::memcpy(memory.data(), &request, sizeof(request));
  coral_operator_descriptor descriptor = {};
  descriptor.magic = CORAL_OPERATOR_ABI_MAGIC;
  descriptor.version = CORAL_OPERATOR_ABI_VERSION;
  descriptor.descriptor_size = sizeof(descriptor);
  descriptor.opcode = CORAL_OPERATOR_OP_QWEN_TINY_INFER;
  descriptor.state = CORAL_OPERATOR_STATE_SUBMITTED;
  descriptor.execution_mode = CORAL_OPERATOR_MODE_HYBRID;
  descriptor.tensor_count = 1;
  descriptor.tensors[0].address = kBase;
  descriptor.tensors[0].size = sizeof(request);
  descriptor.tensors[0].rank = 1;
  descriptor.tensors[0].dimensions[0] = sizeof(request);
  descriptor.tensors[0].element_type = CORAL_OPERATOR_ELEMENT_INT8;

  Gem5HybridOperatorResult result = {};
  assert(DispatchGem5HybridOperator(
      &descriptor, memory.data(), kBase, memory.size(), &result));
  const auto* output = reinterpret_cast<const opennpux_qwen_device_request*>(
      memory.data());
  assert(output->state == OPENNPUX_QWEN_DEVICE_COMPLETE);
  assert(output->error == 0);
  assert(output->completed_operators == 19);
  assert(output->next_token == expected.next_token);
  assert(output->logits_checksum == expected.logits_checksum);
  return 0;
}
