#include <stddef.h>
#include <stdint.h>

#include "hw_sim/gem5_bridge/npu_submission.h"

#define EXTMEM_BASE UINT32_C(0x20000000)
#define COMMAND_BUFFER_SIZE UINT32_C(0x00010000)
#define ERROR_ABI UINT32_C(1)
#define ERROR_BOUNDS UINT32_C(2)
#define ERROR_CHECKSUM UINT32_C(3)
#define ERROR_COMMAND UINT32_C(4)

static uint32_t
checksum(const volatile uint8_t *bytes, uint32_t size)
{
    const uint32_t begin = (uint32_t)offsetof(
        struct opennpux_npu_invocation_header, checksum);
    uint32_t value = UINT32_C(2166136261);
    for (uint32_t index = 0; index < size; ++index) {
        uint8_t byte = index >= begin && index < begin + 4 ? 0 : bytes[index];
        value ^= byte;
        value *= UINT32_C(16777619);
    }
    return value;
}

static int
range_valid(uint64_t offset, uint32_t count, uint32_t record_size,
            uint32_t total_size)
{
    const uint64_t bytes = (uint64_t)count * record_size;
    return offset <= total_size && bytes <= total_size - offset;
}

static void
finish(volatile struct opennpux_npu_completion *completion, uint32_t state,
       uint32_t error, uint32_t fault)
{
    completion->error_code = error;
    completion->faulting_command = fault;
    __asm__ volatile("" ::: "memory");
    completion->state = state;
}

int
main(void)
{
    volatile uint8_t *base = (volatile uint8_t *)EXTMEM_BASE;
    volatile struct opennpux_npu_invocation_header *header =
        (volatile struct opennpux_npu_invocation_header *)base;
    if (header->completion_address < EXTMEM_BASE ||
        header->completion_address >
            EXTMEM_BASE + COMMAND_BUFFER_SIZE - sizeof(struct opennpux_npu_completion)) {
        return 1;
    }
    volatile struct opennpux_npu_completion *completion =
        (volatile struct opennpux_npu_completion *)(uintptr_t)
            header->completion_address;
    completion->magic = OPENNPUX_NPU_COMPLETION_MAGIC;
    completion->version = OPENNPUX_NPU_COMPLETION_VERSION;
    completion->struct_size = sizeof(*completion);
    completion->sequence = header->sequence;
    completion->state = OPENNPUX_NPU_COMPLETION_RUNNING;

    if (header->magic != OPENNPUX_NPU_INVOCATION_MAGIC ||
        header->version != OPENNPUX_NPU_INVOCATION_VERSION ||
        header->header_size != sizeof(*header) || header->total_size < sizeof(*header) ||
        header->total_size > COMMAND_BUFFER_SIZE || header->command_count == 0) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_ABI, 0);
        return 1;
    }
    if (!range_valid(header->binding_offset, header->binding_count,
                     sizeof(struct opennpux_npu_tensor_binding), header->total_size) ||
        !range_valid(header->command_offset, header->command_count,
                     sizeof(struct opennpux_npu_command), header->total_size)) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_BOUNDS, 0);
        return 1;
    }
    if (checksum(base, header->total_size) != header->checksum) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_CHECKSUM, 0);
        return 1;
    }

    volatile struct opennpux_npu_command *commands =
        (volatile struct opennpux_npu_command *)(base + header->command_offset);
    uint64_t cycles = 0;
    uint64_t bytes = 0;
    for (uint32_t index = 0; index < header->command_count; ++index) {
        if (commands[index].opcode == 0 ||
            commands[index].first_binding > header->binding_count ||
            commands[index].binding_count >
                header->binding_count - commands[index].first_binding) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_COMMAND, index);
            return 1;
        }
        cycles += commands[index].estimated_operations;
        bytes += commands[index].estimated_bytes;
        completion->completed_commands = index + 1;
    }
    completion->npu_cycles = cycles;
    completion->dma_bytes_read = bytes;
    finish(completion, OPENNPUX_NPU_COMPLETION_SUCCESS, 0, UINT32_MAX);
    return 0;
}
