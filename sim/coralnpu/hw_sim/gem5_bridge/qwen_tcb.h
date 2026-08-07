#ifndef HW_SIM_GEM5_BRIDGE_QWEN_TCB_H_
#define HW_SIM_GEM5_BRIDGE_QWEN_TCB_H_

#include <stdint.h>

#define OPENNPUX_QWEN_MAX_OPS UINT32_C(64)
#define OPENNPUX_QWEN_OP_KIND_COUNT UINT32_C(9)
#define OPENNPUX_QWEN_OP_MAX_DIMS UINT32_C(4)
#define OPENNPUX_QWEN_TCB_MAGIC UINT32_C(0x4e455751)
#define OPENNPUX_QWEN_TCB_VERSION UINT32_C(1)
#define OPENNPUX_QWEN_TCB_MAX_SIZE UINT32_C(8192)
#define OPENNPUX_QWEN_TCB_TENSOR_BASE UINT32_C(0x00001000)
#define OPENNPUX_QWEN_TCB_TENSOR_ALIGN UINT32_C(64)

#define OPENNPUX_QWEN_TCB_STATE_PENDING UINT32_C(0)
#define OPENNPUX_QWEN_TCB_STATE_RUNNING UINT32_C(1)
#define OPENNPUX_QWEN_TCB_STATE_COMPLETE UINT32_C(2)
#define OPENNPUX_QWEN_TCB_STATE_ERROR UINT32_C(3)

#define OPENNPUX_QWEN_TCB_ERROR_NONE UINT32_C(0)
#define OPENNPUX_QWEN_TCB_ERROR_ABI UINT32_C(1)
#define OPENNPUX_QWEN_TCB_ERROR_CHECKSUM UINT32_C(2)
#define OPENNPUX_QWEN_TCB_ERROR_OPERATOR UINT32_C(3)
#define OPENNPUX_QWEN_TCB_ERROR_BOUNDS UINT32_C(4)

struct opennpux_qwen_tcb_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t total_size;
    uint32_t op_count;
    uint32_t prompt_tokens;
    uint32_t hidden_size;
    uint32_t vocab_size;
    uint32_t prompt_checksum;
    uint32_t logits_checksum;
    uint32_t next_token;
    uint32_t tcb_checksum;
    uint64_t operation_count;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t modeled_cycles;
    uint32_t tcb_state;
    uint32_t tcb_error;
    uint32_t device_checksum;
    uint32_t device_completed_ops;
    uint64_t device_modeled_cycles;
    uint32_t reserved[2];
};

struct opennpux_qwen_tcb_op {
    uint32_t index;
    uint32_t kind;
    uint32_t layer;
    uint32_t flags;
    uint32_t rank;
    uint32_t dims[OPENNPUX_QWEN_OP_MAX_DIMS];
    uint32_t input_offset;
    uint32_t weight_offset;
    uint32_t output_offset;
    uint32_t scratch_offset;
    uint64_t operations;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t modeled_cycles;
    uint32_t reserved[4];
};

#endif  // HW_SIM_GEM5_BRIDGE_QWEN_TCB_H_
