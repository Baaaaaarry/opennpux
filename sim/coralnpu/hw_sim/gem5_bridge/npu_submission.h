#ifndef HW_SIM_GEM5_BRIDGE_NPU_SUBMISSION_H_
#define HW_SIM_GEM5_BRIDGE_NPU_SUBMISSION_H_

#include <stddef.h>
#include <stdint.h>

#define OPENNPUX_NPU_INVOCATION_MAGIC UINT32_C(0x4958504e)
#define OPENNPUX_NPU_INVOCATION_VERSION UINT32_C(1)
#define OPENNPUX_NPU_COMPLETION_MAGIC UINT32_C(0x4358504e)
#define OPENNPUX_NPU_COMPLETION_VERSION UINT32_C(1)
#define OPENNPUX_NPU_RECORD_ALIGNMENT UINT32_C(64)
#define OPENNPUX_NPU_MAX_BINDINGS UINT32_C(4096)
#define OPENNPUX_NPU_MAX_COMMANDS UINT32_C(65536)
#define OPENNPUX_NPU_MAX_RANK UINT32_C(8)
#define OPENNPUX_NPU_INVOCATION_HEADER_SIZE UINT32_C(144)
#define OPENNPUX_NPU_TENSOR_BINDING_SIZE UINT32_C(112)
#define OPENNPUX_NPU_COMMAND_SIZE UINT32_C(112)
#define OPENNPUX_NPU_COMPLETION_SIZE UINT32_C(112)
#define OPENNPUX_NPU_RUNTIME_FIELD_MASK UINT64_C(0xffff)
#define OPENNPUX_NPU_RUNTIME_SEQUENCE_SHIFT UINT32_C(16)
#define OPENNPUX_NPU_RUNTIME_KV_SHIFT UINT32_C(32)
#define OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT UINT32_C(48)
#define OPENNPUX_NPU_RESOURCE_STATE_SHIFT UINT32_C(16)
#define OPENNPUX_NPU_RESOURCE_SCRATCH_SHIFT UINT32_C(32)

#define OPENNPUX_NPU_COMPLETION_PENDING UINT32_C(0)
#define OPENNPUX_NPU_COMPLETION_RUNNING UINT32_C(1)
#define OPENNPUX_NPU_COMPLETION_SUCCESS UINT32_C(2)
#define OPENNPUX_NPU_COMPLETION_ERROR UINT32_C(3)
#define OPENNPUX_NPU_BIND_READ UINT32_C(1)
#define OPENNPUX_NPU_BIND_WRITE UINT32_C(2)
#define OPENNPUX_NPU_BIND_PERSISTENT UINT32_C(4)
#define OPENNPUX_NPU_BIND_WEIGHT UINT32_C(8)

struct opennpux_npu_invocation_header {
    uint32_t magic, version, header_size, total_size;
    uint64_t sequence, executable_id, context_id;
    uint32_t entry_point, flags, priority, binding_count, command_count, checksum;
    uint64_t binding_offset, command_offset, parameter_offset, parameter_size;
    uint64_t persistent_state_handle, dependency_fence, completion_address;
    uint64_t reserved[3];
};

struct opennpux_npu_tensor_binding {
    uint32_t tensor_id, flags, data_type, rank;
    uint64_t device_address, byte_size;
    uint32_t dimensions[OPENNPUX_NPU_MAX_RANK];
    uint32_t strides[OPENNPUX_NPU_MAX_RANK];
    uint64_t memory_object, memory_offset;
};

struct opennpux_npu_command {
    uint32_t command_id, opcode, flags, capability_id;
    uint32_t first_binding, binding_count, dependency_token, completion_token;
    uint64_t parameter_offset, parameter_size, scratch_offset, scratch_size;
    uint64_t estimated_operations, estimated_bytes, profiling_tag;
    uint64_t parameter_symbol, runtime_shape, resource_bindings;
};

struct opennpux_npu_completion {
    uint32_t magic, version, struct_size, state;
    uint64_t sequence;
    uint32_t error_code, completed_commands, faulting_command, reserved0;
    uint64_t npu_cycles, dma_bytes_read, dma_bytes_written, stall_cycles;
    uint64_t completion_fence, trace_address, trace_size, reserved[2];
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_invocation_header) == 144);
static_assert(sizeof(struct opennpux_npu_tensor_binding) == 112);
static_assert(sizeof(struct opennpux_npu_command) == 112);
static_assert(sizeof(struct opennpux_npu_completion) == 112);
#else
_Static_assert(sizeof(struct opennpux_npu_invocation_header) == 144,
               "invocation ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_tensor_binding) == 112,
               "binding ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_command) == 112,
               "command ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_completion) == 112,
               "completion ABI size changed");
#endif

#endif
