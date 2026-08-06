#ifndef OPENNPUX_QWEN_MODEL_H
#define OPENNPUX_QWEN_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_QWEN_TINY_FORMAT "OPENNPUX_QWEN_TINY_V1"
#define OPENNPUX_QWEN_MAX_OPS UINT32_C(64)
#define OPENNPUX_QWEN_OP_KIND_COUNT UINT32_C(9)
#define OPENNPUX_QWEN_OP_MAX_DIMS UINT32_C(4)
#define OPENNPUX_QWEN_TCB_MAGIC UINT32_C(0x4e455751)
#define OPENNPUX_QWEN_TCB_VERSION UINT32_C(1)
#define OPENNPUX_QWEN_TCB_MAX_SIZE UINT32_C(8192)
#define OPENNPUX_QWEN_TCB_TENSOR_BASE UINT32_C(0x00001000)
#define OPENNPUX_QWEN_TCB_TENSOR_ALIGN UINT32_C(64)

struct opennpux_qwen_op_entry {
    uint32_t index;
    uint32_t kind;
    uint32_t layer;
    uint32_t dim_count;
    uint32_t dims[OPENNPUX_QWEN_OP_MAX_DIMS];
    uint64_t operations;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t modeled_cycles;
};

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
    uint32_t reserved[8];
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

struct opennpux_qwen_model_info {
    char format[32];
    char name[64];
    uint32_t version;
    uint32_t layer_count;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t head_count;
    uint32_t head_dim;
    uint32_t prompt_token_count;
    uint32_t operator_count;
    uint32_t next_token;
    uint32_t logits_checksum;
    uint32_t weight_checksum;
    uint32_t op_mask;
};

struct opennpux_qwen_run_result {
    struct opennpux_qwen_model_info info;
    uint32_t prompt_checksum;
    uint32_t completed_operators;
    uint32_t prefill_pass;
    uint32_t decode_pass;
    uint32_t output_checksum;
    uint32_t next_token;
    uint64_t modeled_cycles;
    uint64_t operation_count;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint32_t tcb_size;
    uint32_t tcb_checksum;
    uint32_t op_counts[OPENNPUX_QWEN_OP_KIND_COUNT];
    struct opennpux_qwen_op_entry ops[OPENNPUX_QWEN_MAX_OPS];
};

enum opennpux_qwen_op_bit {
    OPENNPUX_QWEN_OP_EMBED = 1u << 0,
    OPENNPUX_QWEN_OP_MATMUL = 1u << 1,
    OPENNPUX_QWEN_OP_ADD = 1u << 2,
    OPENNPUX_QWEN_OP_MUL = 1u << 3,
    OPENNPUX_QWEN_OP_RMS_NORM = 1u << 4,
    OPENNPUX_QWEN_OP_ROPE = 1u << 5,
    OPENNPUX_QWEN_OP_SILU = 1u << 6,
    OPENNPUX_QWEN_OP_SOFTMAX = 1u << 7,
    OPENNPUX_QWEN_OP_TOPK = 1u << 8,
};

int opennpux_qwen_load_model_info(const char *path,
                                  struct opennpux_qwen_model_info *info);
int opennpux_qwen_run_golden(const char *path,
                             struct opennpux_qwen_run_result *result);
int opennpux_qwen_run_hybrid_sim(const char *path,
                                 struct opennpux_qwen_run_result *result);
int opennpux_qwen_build_tcb(const struct opennpux_qwen_run_result *result,
                            void *buffer, uint32_t buffer_size,
                            uint32_t *tcb_size, uint32_t *tcb_checksum);
const char *opennpux_qwen_required_ops_string(void);
uint32_t opennpux_qwen_required_op_mask(void);
const char *opennpux_qwen_op_name(uint32_t index);

#ifdef __cplusplus
}
#endif

#endif
