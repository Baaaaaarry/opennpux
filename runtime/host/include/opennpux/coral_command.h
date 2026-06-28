#ifndef OPENNPUX_CORAL_COMMAND_H
#define OPENNPUX_CORAL_COMMAND_H

#include <stdint.h>

#define OPENNPUX_CORAL_COMMAND_MAGIC UINT32_C(0x4e505843)
#define OPENNPUX_CORAL_COMMAND_ABI_VERSION UINT32_C(1)
#define OPENNPUX_CORAL_COMMAND_SIZE UINT32_C(64)

#define OPENNPUX_CORAL_OPCODE_VECTOR_ADD_U32 UINT32_C(1)

#define OPENNPUX_CORAL_COMMAND_PENDING UINT32_C(0)
#define OPENNPUX_CORAL_COMMAND_RUNNING UINT32_C(1)
#define OPENNPUX_CORAL_COMMAND_COMPLETE UINT32_C(2)
#define OPENNPUX_CORAL_COMMAND_ERROR UINT32_C(3)

#define OPENNPUX_CORAL_COMMAND_ERROR_NONE UINT32_C(0)
#define OPENNPUX_CORAL_COMMAND_ERROR_ABI UINT32_C(1)
#define OPENNPUX_CORAL_COMMAND_ERROR_OPCODE UINT32_C(2)
#define OPENNPUX_CORAL_COMMAND_ERROR_BOUNDS UINT32_C(3)

#define OPENNPUX_CORAL_COMMAND_OFFSET UINT32_C(0x000)
#define OPENNPUX_CORAL_INPUT0_OFFSET UINT32_C(0x100)
#define OPENNPUX_CORAL_INPUT1_OFFSET UINT32_C(0x500)
#define OPENNPUX_CORAL_OUTPUT_OFFSET UINT32_C(0x900)
#define OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS UINT32_C(256)

struct opennpux_coral_command {
    uint32_t magic;
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t opcode;
    uint32_t sequence;
    uint32_t flags;
    uint32_t element_count;
    uint32_t reserved0;
    uint32_t input0_offset;
    uint32_t input1_offset;
    uint32_t output_offset;
    uint32_t output_size;
    uint32_t status;
    uint32_t error_code;
    uint32_t completed_elements;
    uint32_t reserved1;
};

#endif

