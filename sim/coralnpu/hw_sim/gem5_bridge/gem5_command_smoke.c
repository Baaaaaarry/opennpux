#include <stdint.h>

#include "hw_sim/gem5_bridge/coral_command.h"

#define CORAL_EXTMEM_BASE UINT32_C(0x20000000)
#define CORAL_EXTMEM_SIZE UINT32_C(0x1000)
#define CUSTOM_MAC_BASE UINT32_C(0x30000000)
#define CUSTOM_MAC_OPERAND_A UINT32_C(0x00)
#define CUSTOM_MAC_OPERAND_B UINT32_C(0x04)
#define CUSTOM_MAC_ACCUMULATOR UINT32_C(0x08)
#define CUSTOM_MAC_COMMAND UINT32_C(0x0c)
#define CUSTOM_MAC_RESULT UINT32_C(0x10)
#define CUSTOM_MAC_STATUS UINT32_C(0x14)
#define CUSTOM_MAC_CYCLES UINT32_C(0x18)
#define CUSTOM_MAC_ID UINT32_C(0x1c)
#define CUSTOM_MAC_EXPECTED_ID UINT32_C(0x4e5058a1)

static volatile uint32_t *
custom_reg(uint32_t offset)
{
    return (volatile uint32_t *)(uintptr_t)(CUSTOM_MAC_BASE + offset);
}

static int
custom_add(uint32_t lhs, uint32_t rhs, uint32_t *result)
{
    *custom_reg(CUSTOM_MAC_OPERAND_A) = lhs;
    *custom_reg(CUSTOM_MAC_OPERAND_B) = 1;
    *custom_reg(CUSTOM_MAC_ACCUMULATOR) = rhs;
    *custom_reg(CUSTOM_MAC_COMMAND) = 1;
    while ((*custom_reg(CUSTOM_MAC_STATUS) & 1) == 0) {
    }
    *result = *custom_reg(CUSTOM_MAC_RESULT);
    return 0;
}

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
    if (command->opcode != OPENNPUX_CORAL_OPCODE_VECTOR_ADD_U32 &&
        command->opcode != OPENNPUX_CORAL_OPCODE_VECTOR_ADD_CUSTOM_U32) {
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
    command->accelerator_cycles = 0;
    const uint32_t custom_cycles_before =
        command->opcode == OPENNPUX_CORAL_OPCODE_VECTOR_ADD_CUSTOM_U32 ?
            *custom_reg(CUSTOM_MAC_CYCLES) : 0;
    if (command->opcode == OPENNPUX_CORAL_OPCODE_VECTOR_ADD_CUSTOM_U32 &&
        *custom_reg(CUSTOM_MAC_ID) != CUSTOM_MAC_EXPECTED_ID) {
        finish_error(command, OPENNPUX_CORAL_COMMAND_ERROR_ACCELERATOR);
        return 1;
    }
    for (uint32_t i = 0; i < command->element_count; ++i) {
        if (command->opcode == OPENNPUX_CORAL_OPCODE_VECTOR_ADD_CUSTOM_U32) {
            uint32_t value = 0;
            custom_add(input0[i], input1[i], &value);
            output[i] = value;
        } else {
            output[i] = input0[i] + input1[i];
        }
        command->completed_elements = i + 1;
    }
    if (command->opcode == OPENNPUX_CORAL_OPCODE_VECTOR_ADD_CUSTOM_U32) {
        command->accelerator_cycles =
            *custom_reg(CUSTOM_MAC_CYCLES) - custom_cycles_before;
    }
    __asm__ volatile("" ::: "memory");
    command->status = OPENNPUX_CORAL_COMMAND_COMPLETE;
    return 0;
}
