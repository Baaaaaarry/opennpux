#ifndef HW_SIM_GEM5_BRIDGE_CORAL_OPERATOR_CLIENT_H_
#define HW_SIM_GEM5_BRIDGE_CORAL_OPERATOR_CLIENT_H_

#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_operator.h"

namespace opennpux {

struct OperatorStagingAllocator {
  uint32_t base;
  uint32_t size;
  uint32_t cursor;
};

inline void OperatorFence() {
  asm volatile("fence rw, rw" ::: "memory");
}

inline uint32_t OperatorMode() {
  return *reinterpret_cast<volatile uint32_t*>(CORAL_OPERATOR_MODE_REG);
}

inline uint32_t OperatorCapabilities() {
  return *reinterpret_cast<volatile uint32_t*>(
      CORAL_OPERATOR_CAPABILITIES_REG);
}

inline bool HybridOperatorSupported(uint32_t opcode) {
  return opcode < 32 &&
         (OperatorCapabilities() & (UINT32_C(1) << opcode)) != 0;
}

inline void InitializeStagingAllocator(
    OperatorStagingAllocator* allocator, uint32_t extmem_base) {
  allocator->base = extmem_base + CORAL_OPERATOR_STAGING_OFFSET;
  allocator->size = CORAL_OPERATOR_STAGING_SIZE;
  allocator->cursor = 0;
}

inline uint32_t AllocateStaging(
    OperatorStagingAllocator* allocator, uint32_t size,
    uint32_t alignment = 16) {
  if (allocator == nullptr || size == 0 || alignment == 0 ||
      (alignment & (alignment - 1)) != 0) {
    return 0;
  }
  const uint32_t aligned =
      (allocator->cursor + alignment - 1) & ~(alignment - 1);
  if (aligned < allocator->cursor || size > allocator->size ||
      aligned > allocator->size - size) {
    return 0;
  }
  allocator->cursor = aligned + size;
  return allocator->base + aligned;
}

inline bool SetOperatorTensor(
    coral_operator_descriptor* descriptor, uint32_t index, uint32_t address,
    uint32_t size, const uint32_t dimensions[CORAL_OPERATOR_MAX_DIMS],
    uint32_t element_type, int32_t zero_point) {
  if (descriptor == nullptr || dimensions == nullptr ||
      index >= CORAL_OPERATOR_MAX_TENSORS || address == 0 || size == 0) {
    return false;
  }
  coral_operator_tensor* tensor = &descriptor->tensors[index];
  tensor->address = address;
  tensor->size = size;
  for (size_t i = 0; i < CORAL_OPERATOR_MAX_DIMS; ++i) {
    tensor->dimensions[i] = dimensions[i];
  }
  tensor->element_type = element_type;
  tensor->zero_point = zero_point;
  if (descriptor->tensor_count <= index) {
    descriptor->tensor_count = index + 1;
  }
  return true;
}

inline void InitializeOperatorDescriptor(
    coral_operator_descriptor* descriptor, uint32_t opcode,
    uint32_t execution_mode) {
  auto* words = reinterpret_cast<volatile uint32_t*>(descriptor);
  for (size_t i = 0; i < sizeof(*descriptor) / sizeof(uint32_t); ++i) {
    words[i] = 0;
  }
  descriptor->magic = CORAL_OPERATOR_ABI_MAGIC;
  descriptor->version = CORAL_OPERATOR_ABI_VERSION;
  descriptor->descriptor_size = sizeof(*descriptor);
  descriptor->opcode = opcode;
  descriptor->execution_mode = execution_mode;
}

inline bool SubmitHybridOperator(
    coral_operator_descriptor* descriptor, uint32_t descriptor_address) {
  descriptor->state = CORAL_OPERATOR_STATE_SUBMITTED;
  OperatorFence();
  *reinterpret_cast<volatile uint32_t*>(CORAL_OPERATOR_DOORBELL_REG) =
      descriptor_address;
  OperatorFence();
  return *reinterpret_cast<volatile uint32_t*>(CORAL_OPERATOR_STATUS_REG) ==
             CORAL_OPERATOR_STATE_COMPLETE &&
         descriptor->state == CORAL_OPERATOR_STATE_COMPLETE &&
         descriptor->error == CORAL_OPERATOR_ERROR_NONE;
}

}  // namespace opennpux

#endif  // HW_SIM_GEM5_BRIDGE_CORAL_OPERATOR_CLIENT_H_
