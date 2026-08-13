#ifndef HW_SIM_GEM5_BRIDGE_GEM5_NPU_LAUNCH_DESCRIPTOR_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_NPU_LAUNCH_DESCRIPTOR_H_

#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_operator.h"

namespace opennpux {

inline void InitializeNpuLaunchAddDescriptor(
    coral_operator_descriptor* descriptor, uint32_t input0_address,
    uint32_t input1_address, uint32_t output_address, uint32_t elements) {
  auto* words = reinterpret_cast<volatile uint32_t*>(descriptor);
  for (size_t i = 0; i < sizeof(*descriptor) / sizeof(uint32_t); ++i) {
    words[i] = 0;
  }
  descriptor->magic = CORAL_OPERATOR_ABI_MAGIC;
  descriptor->version = CORAL_OPERATOR_ABI_VERSION;
  descriptor->descriptor_size = sizeof(*descriptor);
  descriptor->opcode = CORAL_OPERATOR_OP_ADD_INT8;
  descriptor->state = CORAL_OPERATOR_STATE_IDLE;
  descriptor->execution_mode = CORAL_OPERATOR_MODE_HYBRID;
  descriptor->tensor_count = 3;
  descriptor->activation_min = -128;
  descriptor->activation_max = 127;

  const uint32_t addresses[3] = {
      input0_address, input1_address, output_address};
  for (uint32_t i = 0; i < 3; ++i) {
    coral_operator_tensor* tensor = &descriptor->tensors[i];
    tensor->address = addresses[i];
    tensor->size = elements;
    tensor->rank = 1;
    tensor->dimensions[0] = elements;
    tensor->element_type = CORAL_OPERATOR_ELEMENT_INT8;
    tensor->zero_point = 0;
  }
}

}  // namespace opennpux

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_NPU_LAUNCH_DESCRIPTOR_H_
