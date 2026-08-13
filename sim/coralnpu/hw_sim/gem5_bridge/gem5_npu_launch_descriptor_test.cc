#include "hw_sim/gem5_bridge/gem5_npu_launch_descriptor.h"

#include <cassert>

int main() {
  coral_operator_descriptor descriptor = {};
  opennpux::InitializeNpuLaunchAddDescriptor(
      &descriptor, 0x20500000, 0x20500010, 0x20500020, 4);

  assert(descriptor.magic == CORAL_OPERATOR_ABI_MAGIC);
  assert(descriptor.version == CORAL_OPERATOR_ABI_VERSION);
  assert(descriptor.opcode == CORAL_OPERATOR_OP_ADD_INT8);
  assert(descriptor.state == CORAL_OPERATOR_STATE_IDLE);
  assert(descriptor.execution_mode == CORAL_OPERATOR_MODE_HYBRID);
  assert(descriptor.tensor_count == 3);
  assert(descriptor.activation_min == -128);
  assert(descriptor.activation_max == 127);
  for (uint32_t i = 0; i < descriptor.tensor_count; ++i) {
    assert(descriptor.tensors[i].size == 4);
    assert(descriptor.tensors[i].rank == 1);
    assert(descriptor.tensors[i].dimensions[0] == 4);
    assert(descriptor.tensors[i].element_type ==
           CORAL_OPERATOR_ELEMENT_INT8);
  }
  return 0;
}
