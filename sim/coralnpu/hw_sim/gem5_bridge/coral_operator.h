#ifndef HW_SIM_GEM5_BRIDGE_CORAL_OPERATOR_H_
#define HW_SIM_GEM5_BRIDGE_CORAL_OPERATOR_H_

#include <stdint.h>

#define CORAL_OPERATOR_ABI_MAGIC UINT32_C(0x4e50584f)
#define CORAL_OPERATOR_ABI_VERSION UINT32_C(1)

#define CORAL_OPERATOR_MMIO_BASE UINT32_C(0x30000100)
#define CORAL_OPERATOR_MODE_REG (CORAL_OPERATOR_MMIO_BASE + UINT32_C(0x00))
#define CORAL_OPERATOR_DOORBELL_REG \
    (CORAL_OPERATOR_MMIO_BASE + UINT32_C(0x04))
#define CORAL_OPERATOR_STATUS_REG \
    (CORAL_OPERATOR_MMIO_BASE + UINT32_C(0x08))

#define CORAL_OPERATOR_DESCRIPTOR_OFFSET UINT32_C(0x00400100)
#define CORAL_OPERATOR_MAX_TENSORS UINT32_C(4)
#define CORAL_OPERATOR_MAX_DIMS UINT32_C(4)

enum coral_operator_mode {
    CORAL_OPERATOR_MODE_RTL = 0,
    CORAL_OPERATOR_MODE_HYBRID = 1,
};

enum coral_operator_opcode {
    CORAL_OPERATOR_OP_INVALID = 0,
    CORAL_OPERATOR_OP_PARTIAL_MOBILENET = 1,
    CORAL_OPERATOR_OP_CONV_2D_INT8 = 2,
    CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8 = 3,
    CORAL_OPERATOR_OP_MATMUL_INT8 = 4,
};

enum coral_operator_state {
    CORAL_OPERATOR_STATE_IDLE = 0,
    CORAL_OPERATOR_STATE_SUBMITTED = 1,
    CORAL_OPERATOR_STATE_RUNNING = 2,
    CORAL_OPERATOR_STATE_COMPLETE = 3,
    CORAL_OPERATOR_STATE_ERROR = 4,
};

enum coral_operator_error {
    CORAL_OPERATOR_ERROR_NONE = 0,
    CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR = 1,
    CORAL_OPERATOR_ERROR_UNSUPPORTED = 2,
    CORAL_OPERATOR_ERROR_EXECUTION = 3,
    CORAL_OPERATOR_ERROR_ADDRESS = 4,
};

struct coral_operator_tensor {
    uint32_t address;
    uint32_t size;
    uint32_t dimensions[CORAL_OPERATOR_MAX_DIMS];
    uint32_t element_type;
    int32_t zero_point;
};

struct coral_operator_descriptor {
    uint32_t magic;
    uint32_t version;
    uint32_t descriptor_size;
    uint32_t opcode;
    uint32_t state;
    uint32_t error;
    uint32_t flags;
    uint32_t tensor_count;
    struct coral_operator_tensor tensors[CORAL_OPERATOR_MAX_TENSORS];
    uint32_t stride_height;
    uint32_t stride_width;
    uint32_t padding_height;
    uint32_t padding_width;
    uint32_t multiplier_address;
    uint32_t shift_address;
    int32_t output_zero_point;
    int32_t activation_min;
    int32_t activation_max;
    uint32_t reserved0;
    uint64_t host_elapsed_ns;
    uint64_t modeled_cycles;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint32_t reserved[8];
};

#endif  // HW_SIM_GEM5_BRIDGE_CORAL_OPERATOR_H_
