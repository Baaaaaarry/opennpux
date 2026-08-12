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

inline uint32_t SampledRtlMask() {
  return *reinterpret_cast<volatile uint32_t*>(
      CORAL_OPERATOR_SAMPLED_RTL_MASK_REG);
}

inline bool HybridOperatorSupported(uint32_t opcode) {
  return opcode < 32 &&
         (OperatorCapabilities() & (UINT32_C(1) << opcode)) != 0;
}

inline bool OperatorUsesHybrid(uint32_t opcode) {
  const uint32_t mode = OperatorMode();
  if (mode == CORAL_OPERATOR_MODE_HYBRID) {
    return HybridOperatorSupported(opcode);
  }
  if (mode == CORAL_OPERATOR_MODE_SAMPLED) {
    return HybridOperatorSupported(opcode) &&
           (SampledRtlMask() & (UINT32_C(1) << opcode)) == 0;
  }
  return false;
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
  // RTL erratum workaround: the Coral external memory path deadlocks on
  // vector loads/stores that span a 16-byte line boundary. Keep every
  // staging allocation 16-byte aligned so vectorized copies to/from the
  // staging area never cross a line. The recorded tensor sizes are
  // unaffected; this only pads the allocator cursor.
  const uint32_t aligned_size = (size + 15) & ~UINT32_C(15);
  const uint32_t aligned =
      (allocator->cursor + alignment - 1) & ~(alignment - 1);
  if (aligned < allocator->cursor || aligned_size > allocator->size ||
      aligned > allocator->size - aligned_size) {
    return 0;
  }
  allocator->cursor = aligned + aligned_size;
  return allocator->base + aligned;
}

inline bool SetOperatorTensor(
    coral_operator_descriptor* descriptor, uint32_t index, uint32_t address,
    uint32_t size, uint32_t rank,
    const uint32_t dimensions[CORAL_OPERATOR_MAX_DIMS],
    uint32_t element_type, int32_t zero_point) {
  if (descriptor == nullptr || dimensions == nullptr ||
      index >= CORAL_OPERATOR_MAX_TENSORS || address == 0 || size == 0 ||
      rank == 0 || rank > CORAL_OPERATOR_MAX_DIMS) {
    return false;
  }
  coral_operator_tensor* tensor = &descriptor->tensors[index];
  tensor->address = address;
  tensor->size = size;
  tensor->rank = rank;
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
  descriptor->flags |= CORAL_OPERATOR_FLAG_CUSTOM_INSTRUCTION;
  OperatorFence();
  const uint32_t operator_base = CORAL_OPERATOR_MMIO_BASE;
  // CUSTOM_0 funct7=0/funct3=0/rd=x0. Coral Decode lowers this instruction to
  // an SW of rs2 at rs1+4, which is the existing operator doorbell register.
  asm volatile(".insn r 0x0b, 0, 0, x0, %0, %1"
               :
               : "r"(operator_base), "r"(descriptor_address)
               : "memory");
  OperatorFence();
  volatile uint32_t* status =
      reinterpret_cast<volatile uint32_t*>(CORAL_OPERATOR_STATUS_REG);
  while (*status == CORAL_OPERATOR_STATE_RUNNING) {
    asm volatile("nop");
  }
  return *status == CORAL_OPERATOR_STATE_COMPLETE &&
         descriptor->state == CORAL_OPERATOR_STATE_COMPLETE &&
         descriptor->error == CORAL_OPERATOR_ERROR_NONE;
}

}  // namespace opennpux

#endif  // HW_SIM_GEM5_BRIDGE_CORAL_OPERATOR_CLIENT_H_
