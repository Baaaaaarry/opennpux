#include <stdint.h>

#include "hw_sim/gem5_bridge/coral_command.h"

#define CORAL_EXTMEM_BASE UINT32_C(0x20000000)
#define CORAL_EXTMEM_SIZE UINT32_C(0x1000)

static int
tensor_in_bounds(uint32_t offset, uint32_t count)
{
    if ((offset & UINT32_C(3)) != 0 ||
        count > OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS) {
        return 0;
    }
    return offset <= CORAL_EXTMEM_SIZE &&
           count * sizeof(uint32_t) <= CORAL_EXTMEM_SIZE - offset;
}

static void
finish_error(volatile struct opennpux_coral_command *command,
             uint32_t error)
{
    command->error_code = error;
    command->status = OPENNPUX_CORAL_COMMAND_ERROR;
}

int
main(void)
{
    volatile uint8_t *shared = (volatile uint8_t *)CORAL_EXTMEM_BASE;
    volatile struct opennpux_coral_command *command =
        (volatile struct opennpux_coral_command *)(
            shared + OPENNPUX_CORAL_COMMAND_OFFSET);

    if (command->magic != OPENNPUX_CORAL_COMMAND_MAGIC ||
        command->abi_version != OPENNPUX_CORAL_COMMAND_ABI_VERSION ||
        command->struct_size != sizeof(*command)) {
        finish_error(command, OPENNPUX_CORAL_COMMAND_ERROR_ABI);
        return 1;
    }
    if (command->opcode != OPENNPUX_CORAL_OPCODE_VECTOR_ADD_U32) {
        finish_error(command, OPENNPUX_CORAL_COMMAND_ERROR_OPCODE);
        return 1;
    }
    if (!tensor_in_bounds(command->input0_offset, command->element_count) ||
        !tensor_in_bounds(command->input1_offset, command->element_count) ||
        !tensor_in_bounds(command->output_offset, command->element_count) ||
        command->output_size != command->element_count * sizeof(uint32_t)) {
        finish_error(command, OPENNPUX_CORAL_COMMAND_ERROR_BOUNDS);
        return 1;
    }

    volatile uint32_t *input0 =
        (volatile uint32_t *)(shared + command->input0_offset);
    volatile uint32_t *input1 =
        (volatile uint32_t *)(shared + command->input1_offset);
    volatile uint32_t *output =
        (volatile uint32_t *)(shared + command->output_offset);

    command->status = OPENNPUX_CORAL_COMMAND_RUNNING;
    command->error_code = OPENNPUX_CORAL_COMMAND_ERROR_NONE;
    command->completed_elements = 0;
    for (uint32_t i = 0; i < command->element_count; ++i) {
        output[i] = input0[i] + input1[i];
        command->completed_elements = i + 1;
    }
    __asm__ volatile("" ::: "memory");
    command->status = OPENNPUX_CORAL_COMMAND_COMPLETE;
    return 0;
}
