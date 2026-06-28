#ifndef OPENNPUX_CORAL_MODEL_H
#define OPENNPUX_CORAL_MODEL_H

#include <stdint.h>

#define OPENNPUX_CORAL_MODEL_MAGIC UINT32_C(0x4e50584d)
#define OPENNPUX_CORAL_MODEL_VERSION UINT32_C(1)
#define OPENNPUX_CORAL_MODEL_MAX_COMMANDS UINT32_C(64)

struct opennpux_coral_model_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t file_size;
    uint32_t command_count;
    uint32_t command_offset;
    uint32_t reserved[2];
};

struct opennpux_coral_model_command {
    uint32_t opcode;
    uint32_t element_count;
    uint32_t input0_offset;
    uint32_t input1_offset;
    uint32_t expected_checksum;
    uint32_t flags;
    uint32_t reserved[2];
};

#endif
