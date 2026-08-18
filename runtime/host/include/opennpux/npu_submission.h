#ifndef OPENNPUX_NPU_SUBMISSION_H
#define OPENNPUX_NPU_SUBMISSION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_NPU_INVOCATION_MAGIC UINT32_C(0x4958504e)
#define OPENNPUX_NPU_INVOCATION_VERSION UINT32_C(2)
#define OPENNPUX_NPU_COMPLETION_MAGIC UINT32_C(0x4358504e)
#define OPENNPUX_NPU_COMPLETION_VERSION UINT32_C(1)
#define OPENNPUX_NPU_TRACE_MAGIC UINT32_C(0x5458504e)
#define OPENNPUX_NPU_TRACE_VERSION UINT32_C(2)
#define OPENNPUX_NPU_RECORD_ALIGNMENT UINT32_C(64)
#define OPENNPUX_NPU_MAX_BINDINGS UINT32_C(4096)
#define OPENNPUX_NPU_MAX_COMMANDS UINT32_C(65536)
#define OPENNPUX_NPU_MAX_RANK UINT32_C(8)
#define OPENNPUX_NPU_INVOCATION_HEADER_SIZE UINT32_C(144)
#define OPENNPUX_NPU_TENSOR_BINDING_SIZE UINT32_C(112)
#define OPENNPUX_NPU_COMMAND_SIZE UINT32_C(112)
#define OPENNPUX_NPU_COMPLETION_SIZE UINT32_C(112)
#define OPENNPUX_NPU_TRACE_HEADER_SIZE UINT32_C(64)
#define OPENNPUX_NPU_TRACE_RECORD_SIZE UINT32_C(32)
#define OPENNPUX_NPU_TRACE_MAX_OPCODE UINT32_C(17)
#define OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC UINT32_C(0x5058504e)
#define OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION UINT32_C(2)
#define OPENNPUX_NPU_OPERATOR_PARAMETERS_SIZE UINT32_C(64)

#define OPENNPUX_NPU_RESOURCE_BINDING_NONE UINT16_C(0xffff)
#define OPENNPUX_NPU_RUNTIME_FIELD_MASK UINT64_C(0xffff)
#define OPENNPUX_NPU_RUNTIME_SEQUENCE_SHIFT UINT32_C(16)
#define OPENNPUX_NPU_RUNTIME_KV_SHIFT UINT32_C(32)
#define OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT UINT32_C(48)
#define OPENNPUX_NPU_RESOURCE_STATE_SHIFT UINT32_C(16)
#define OPENNPUX_NPU_RESOURCE_SCRATCH_SHIFT UINT32_C(32)

enum opennpux_npu_entry_point {
    OPENNPUX_NPU_ENTRY_DEFAULT = 0,
    OPENNPUX_NPU_ENTRY_PREFILL = 1,
    OPENNPUX_NPU_ENTRY_DECODE = 2,
};

enum opennpux_npu_opcode {
    OPENNPUX_NPU_OP_INVALID = 0,
    OPENNPUX_NPU_OP_EMBED = 1,
    OPENNPUX_NPU_OP_MATMUL = 2,
    OPENNPUX_NPU_OP_ADD = 3,
    OPENNPUX_NPU_OP_MUL = 4,
    OPENNPUX_NPU_OP_NORMALIZE = 5,
    OPENNPUX_NPU_OP_ROPE = 6,
    OPENNPUX_NPU_OP_SOFTMAX = 7,
    OPENNPUX_NPU_OP_TOPK = 8,
    OPENNPUX_NPU_OP_CONVOLUTION = 9,
    OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION = 10,
    OPENNPUX_NPU_OP_RECURRENT_UPDATE = 11,
    OPENNPUX_NPU_OP_ROUTER = 12,
    OPENNPUX_NPU_OP_EXPERT = 13,
    OPENNPUX_NPU_OP_DMA = 14,
    OPENNPUX_NPU_OP_ATTENTION = 15,
    OPENNPUX_NPU_OP_ACTIVATION = 16,
    OPENNPUX_NPU_OP_COMBINE = 17,
    OPENNPUX_NPU_OP_CUSTOM = 0x10000,
};

enum opennpux_npu_data_type {
    OPENNPUX_NPU_DTYPE_INVALID = 0,
    OPENNPUX_NPU_DTYPE_INT4_PACKED = 1,
    OPENNPUX_NPU_DTYPE_INT8 = 2,
    OPENNPUX_NPU_DTYPE_INT32 = 3,
    OPENNPUX_NPU_DTYPE_FLOAT16 = 4,
    OPENNPUX_NPU_DTYPE_BFLOAT16 = 5,
    OPENNPUX_NPU_DTYPE_FLOAT32 = 6,
};

enum opennpux_npu_binding_flags {
    OPENNPUX_NPU_BIND_READ = 1u << 0,
    OPENNPUX_NPU_BIND_WRITE = 1u << 1,
    OPENNPUX_NPU_BIND_PERSISTENT = 1u << 2,
    OPENNPUX_NPU_BIND_WEIGHT = 1u << 3,
    OPENNPUX_NPU_BIND_PAGE_QUEUE = 1u << 4,
    OPENNPUX_NPU_BIND_PAGE_CACHE = 1u << 5,
    OPENNPUX_NPU_BIND_ROUTE_TABLE = 1u << 6,
    OPENNPUX_NPU_BIND_PAGE_RESIDENCY = 1u << 7,
};

enum opennpux_npu_invocation_flags {
    OPENNPUX_NPU_INVOKE_PROFILE = 1u << 0,
    OPENNPUX_NPU_INVOKE_ALLOW_FALLBACK = 1u << 1,
    OPENNPUX_NPU_INVOKE_NUMERICAL = 1u << 2,
};

enum opennpux_npu_command_flags {
    OPENNPUX_NPU_COMMAND_USES_WEIGHT = 1u << 0,
};

enum opennpux_npu_operator_phase {
    OPENNPUX_NPU_PHASE_INVALID = 0,
    OPENNPUX_NPU_PHASE_EMBED = 1,
    OPENNPUX_NPU_PHASE_NORMALIZE = 2,
    OPENNPUX_NPU_PHASE_MATMUL = 3,
    OPENNPUX_NPU_PHASE_ATTENTION = 4,
    OPENNPUX_NPU_PHASE_STATE_UPDATE = 5,
    OPENNPUX_NPU_PHASE_ROUTER = 6,
    OPENNPUX_NPU_PHASE_EXPERT = 7,
    OPENNPUX_NPU_PHASE_COMBINE = 8,
    OPENNPUX_NPU_PHASE_TOPK = 9,
};

enum opennpux_npu_operator_parameter_flags {
    OPENNPUX_NPU_PARAMETER_GPTQ = 1u << 0,
    OPENNPUX_NPU_PARAMETER_DYNAMIC_ROWS = 1u << 1,
};

enum opennpux_npu_completion_state {
    OPENNPUX_NPU_COMPLETION_PENDING = 0,
    OPENNPUX_NPU_COMPLETION_RUNNING = 1,
    OPENNPUX_NPU_COMPLETION_SUCCESS = 2,
    OPENNPUX_NPU_COMPLETION_ERROR = 3,
};

struct opennpux_npu_invocation_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t total_size;
    uint64_t sequence;
    uint64_t executable_id;
    uint64_t context_id;
    uint32_t entry_point;
    uint32_t flags;
    uint32_t priority;
    uint32_t binding_count;
    uint32_t command_count;
    uint32_t checksum;
    uint64_t binding_offset;
    uint64_t command_offset;
    uint64_t parameter_offset;
    uint64_t parameter_size;
    uint64_t persistent_state_handle;
    uint64_t dependency_fence;
    uint64_t completion_address;
    uint64_t reserved[3];
};

struct opennpux_npu_tensor_binding {
    uint32_t tensor_id;
    uint32_t flags;
    uint32_t data_type;
    uint32_t rank;
    uint64_t device_address;
    uint64_t byte_size;
    uint32_t dimensions[OPENNPUX_NPU_MAX_RANK];
    uint32_t strides[OPENNPUX_NPU_MAX_RANK];
    uint64_t memory_object;
    uint64_t memory_offset;
};

struct opennpux_npu_command {
    uint32_t command_id;
    uint32_t opcode;
    uint32_t flags;
    uint32_t capability_id;
    uint32_t first_binding;
    uint32_t binding_count;
    uint32_t dependency_token;
    uint32_t completion_token;
    uint64_t parameter_offset;
    uint64_t parameter_size;
    uint64_t scratch_offset;
    uint64_t scratch_size;
    uint64_t estimated_operations;
    uint64_t estimated_bytes;
    uint64_t profiling_tag;
    uint64_t parameter_symbol;
    uint64_t runtime_shape;
    uint64_t resource_bindings;
};

struct opennpux_npu_operator_parameters {
    uint32_t magic;
    uint32_t version;
    uint32_t struct_size;
    uint32_t opcode;
    uint32_t phase;
    uint32_t flags;
    uint32_t input_features;
    uint32_t output_features;
    uint32_t intermediate_features;
    uint32_t head_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t quantization_bits;
    uint32_t quantization_group_size;
    uint32_t scale_data_type;
    uint32_t quantized_zero_bias;
};

struct opennpux_npu_completion {
    uint32_t magic;
    uint32_t version;
    uint32_t struct_size;
    uint32_t state;
    uint64_t sequence;
    uint32_t error_code;
    uint32_t completed_commands;
    uint32_t faulting_command;
    uint32_t reserved0;
    uint64_t npu_cycles;
    uint64_t dma_bytes_read;
    uint64_t dma_bytes_written;
    uint64_t stall_cycles;
    uint64_t completion_fence;
    uint64_t trace_address;
    uint64_t trace_size;
    uint64_t reserved[2];
};

struct opennpux_npu_trace_header {
    uint32_t magic;
    uint32_t version;
    uint32_t struct_size;
    uint32_t record_count;
    uint32_t command_count;
    uint32_t dependency_edges;
    uint32_t weight_page_requests;
    uint32_t weight_checksum;
    uint64_t capability_mask;
    uint64_t estimated_operations;
    uint64_t estimated_bytes;
    uint64_t weight_dma_bytes;
};

struct opennpux_npu_trace_record {
    uint32_t opcode;
    uint32_t command_count;
    uint64_t estimated_operations;
    uint64_t estimated_bytes;
    uint64_t weight_dma_bytes;
};

struct opennpux_npu_submission_builder {
    uint8_t *buffer;
    size_t capacity;
    struct opennpux_npu_invocation_header *header;
    struct opennpux_npu_tensor_binding *bindings;
    struct opennpux_npu_command *commands;
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_invocation_header) ==
              OPENNPUX_NPU_INVOCATION_HEADER_SIZE);
static_assert(sizeof(struct opennpux_npu_tensor_binding) ==
              OPENNPUX_NPU_TENSOR_BINDING_SIZE);
static_assert(sizeof(struct opennpux_npu_command) ==
              OPENNPUX_NPU_COMMAND_SIZE);
static_assert(sizeof(struct opennpux_npu_completion) ==
              OPENNPUX_NPU_COMPLETION_SIZE);
static_assert(sizeof(struct opennpux_npu_trace_header) ==
              OPENNPUX_NPU_TRACE_HEADER_SIZE);
static_assert(sizeof(struct opennpux_npu_trace_record) ==
              OPENNPUX_NPU_TRACE_RECORD_SIZE);
static_assert(sizeof(struct opennpux_npu_operator_parameters) ==
              OPENNPUX_NPU_OPERATOR_PARAMETERS_SIZE);
#else
_Static_assert(sizeof(struct opennpux_npu_invocation_header) ==
               OPENNPUX_NPU_INVOCATION_HEADER_SIZE,
               "NPU invocation header ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_tensor_binding) ==
               OPENNPUX_NPU_TENSOR_BINDING_SIZE,
               "NPU tensor binding ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_command) ==
               OPENNPUX_NPU_COMMAND_SIZE,
               "NPU command ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_completion) ==
               OPENNPUX_NPU_COMPLETION_SIZE,
               "NPU completion ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_trace_header) ==
               OPENNPUX_NPU_TRACE_HEADER_SIZE,
               "NPU trace header ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_trace_record) ==
               OPENNPUX_NPU_TRACE_RECORD_SIZE,
               "NPU trace record ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_operator_parameters) ==
               OPENNPUX_NPU_OPERATOR_PARAMETERS_SIZE,
               "NPU operator parameter ABI size changed");
#endif

int opennpux_npu_submission_begin(
    struct opennpux_npu_submission_builder *builder, void *buffer,
    size_t capacity, uint64_t sequence, uint64_t executable_id,
    uint64_t context_id, uint32_t entry_point, uint32_t binding_count,
    uint32_t command_count);
struct opennpux_npu_tensor_binding *opennpux_npu_submission_binding(
    struct opennpux_npu_submission_builder *builder, uint32_t index);
struct opennpux_npu_command *opennpux_npu_submission_command(
    struct opennpux_npu_submission_builder *builder, uint32_t index);
int opennpux_npu_submission_finalize(
    struct opennpux_npu_submission_builder *builder);
int opennpux_npu_submission_validate(const void *buffer, size_t size);
uint32_t opennpux_npu_submission_checksum(const void *buffer, size_t size);
uint64_t opennpux_npu_pack_runtime_shape(
    uint32_t batch_size, uint32_t sequence_length, uint32_t kv_length,
    uint32_t active_experts);
uint64_t opennpux_npu_pack_resource_bindings(
    uint32_t weight_binding, uint32_t state_binding,
    uint32_t scratch_binding);

#ifdef __cplusplus
}
#endif

#endif
