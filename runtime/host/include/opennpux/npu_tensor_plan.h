#ifndef OPENNPUX_NPU_TENSOR_PLAN_H
#define OPENNPUX_NPU_TENSOR_PLAN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_NPU_TENSOR_PLAN_MAGIC UINT32_C(0x5054504e)
#define OPENNPUX_NPU_TENSOR_PLAN_VERSION UINT32_C(1)
#define OPENNPUX_NPU_TENSOR_PLAN_HEADER_SIZE UINT32_C(80)
#define OPENNPUX_NPU_TENSOR_PLAN_TENSOR_SIZE UINT32_C(80)
#define OPENNPUX_NPU_TENSOR_PLAN_COMMAND_SIZE UINT32_C(48)
#define OPENNPUX_NPU_TENSOR_PLAN_SLOT_SIZE UINT32_C(16)
#define OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK UINT32_C(8)
#define OPENNPUX_NPU_TENSOR_PLAN_MAX_INPUTS UINT32_C(4)
#define OPENNPUX_NPU_TENSOR_PLAN_MAX_OUTPUTS UINT32_C(3)
#define OPENNPUX_NPU_TENSOR_NONE UINT32_MAX

enum opennpux_npu_tensor_storage {
    OPENNPUX_NPU_TENSOR_INPUT = 1,
    OPENNPUX_NPU_TENSOR_OUTPUT = 2,
    OPENNPUX_NPU_TENSOR_SCRATCH = 3,
    OPENNPUX_NPU_TENSOR_PERSISTENT = 4,
};

enum opennpux_npu_dimension_symbol {
    OPENNPUX_NPU_DIMENSION_STATIC = 0,
    OPENNPUX_NPU_DIMENSION_BATCH = 1,
    OPENNPUX_NPU_DIMENSION_SEQUENCE = 2,
    OPENNPUX_NPU_DIMENSION_KV = 3,
    OPENNPUX_NPU_DIMENSION_ACTIVE_EXPERTS = 4,
};

struct opennpux_npu_tensor_plan_header {
    uint32_t magic, version, header_size, total_size;
    uint32_t tensor_count, command_count, slot_count;
    uint32_t tensor_record_size, command_record_size, slot_record_size;
    uint32_t checksum, reserved0;
    uint64_t tensor_offset, command_offset, slot_offset;
    uint64_t scratch_bytes_per_runtime_row;
};

struct opennpux_npu_tensor_plan_tensor {
    uint32_t tensor_id, storage, data_type, rank;
    uint32_t producer_command, last_consumer_command, allocation_slot;
    uint32_t dimension_symbols;
    uint32_t dimensions[OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK];
    uint64_t bytes_per_runtime_row, reserved0;
};

struct opennpux_npu_tensor_plan_command {
    uint32_t command_id, input_count, output_count, reserved0;
    uint32_t input_tensor_ids[OPENNPUX_NPU_TENSOR_PLAN_MAX_INPUTS];
    uint32_t output_tensor_ids[OPENNPUX_NPU_TENSOR_PLAN_MAX_OUTPUTS];
    uint32_t reserved1;
};

struct opennpux_npu_tensor_plan_slot {
    uint32_t slot_id, reserved0;
    uint64_t bytes_per_runtime_row;
};

struct opennpux_npu_tensor_plan {
    void *storage;
    size_t storage_size;
    const struct opennpux_npu_tensor_plan_header *header;
    const struct opennpux_npu_tensor_plan_tensor *tensors;
    const struct opennpux_npu_tensor_plan_command *commands;
    const struct opennpux_npu_tensor_plan_slot *slots;
};

struct opennpux_npu_tensor_plan_memory {
    uint64_t input_address, input_size;
    uint64_t output_address, output_size;
    uint64_t persistent_address, persistent_size;
    uint64_t scratch_address, scratch_size;
};

struct opennpux_npu_tensor_plan_runtime {
    uint32_t batch_size, sequence_length, kv_length, active_experts;
};

struct opennpux_npu_tensor_view {
    uint32_t tensor_id, storage, data_type, rank;
    uint32_t dimensions[OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK];
    uint64_t address, size;
};

struct opennpux_npu_command_tensor_views {
    uint32_t command_id, input_count, output_count, reserved;
    struct opennpux_npu_tensor_view
        inputs[OPENNPUX_NPU_TENSOR_PLAN_MAX_INPUTS];
    struct opennpux_npu_tensor_view
        outputs[OPENNPUX_NPU_TENSOR_PLAN_MAX_OUTPUTS];
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_tensor_plan_header) ==
              OPENNPUX_NPU_TENSOR_PLAN_HEADER_SIZE);
static_assert(sizeof(struct opennpux_npu_tensor_plan_tensor) ==
              OPENNPUX_NPU_TENSOR_PLAN_TENSOR_SIZE);
static_assert(sizeof(struct opennpux_npu_tensor_plan_command) ==
              OPENNPUX_NPU_TENSOR_PLAN_COMMAND_SIZE);
static_assert(sizeof(struct opennpux_npu_tensor_plan_slot) ==
              OPENNPUX_NPU_TENSOR_PLAN_SLOT_SIZE);
#else
_Static_assert(sizeof(struct opennpux_npu_tensor_plan_header) ==
               OPENNPUX_NPU_TENSOR_PLAN_HEADER_SIZE, "tensor plan header ABI changed");
_Static_assert(sizeof(struct opennpux_npu_tensor_plan_tensor) ==
               OPENNPUX_NPU_TENSOR_PLAN_TENSOR_SIZE, "tensor plan tensor ABI changed");
_Static_assert(sizeof(struct opennpux_npu_tensor_plan_command) ==
               OPENNPUX_NPU_TENSOR_PLAN_COMMAND_SIZE, "tensor plan command ABI changed");
_Static_assert(sizeof(struct opennpux_npu_tensor_plan_slot) ==
               OPENNPUX_NPU_TENSOR_PLAN_SLOT_SIZE, "tensor plan slot ABI changed");
#endif

int opennpux_npu_tensor_plan_load(
    const char *path, struct opennpux_npu_tensor_plan *plan);
void opennpux_npu_tensor_plan_unload(struct opennpux_npu_tensor_plan *plan);
int opennpux_npu_tensor_plan_scratch_size(
    const struct opennpux_npu_tensor_plan *plan, uint32_t batch_size,
    uint32_t sequence_length, uint64_t *size);
int opennpux_npu_tensor_plan_persistent_size(
    const struct opennpux_npu_tensor_plan *plan,
    const struct opennpux_npu_tensor_plan_runtime *runtime, uint64_t *size);
int opennpux_npu_tensor_plan_storage_size(
    const struct opennpux_npu_tensor_plan *plan, uint32_t storage,
    const struct opennpux_npu_tensor_plan_runtime *runtime, uint64_t *size);
int opennpux_npu_tensor_plan_memory_layout(
    const struct opennpux_npu_tensor_plan *plan,
    const struct opennpux_npu_tensor_plan_runtime *runtime,
    uint64_t arena_address, uint64_t arena_capacity,
    struct opennpux_npu_tensor_plan_memory *memory, uint64_t *required_size);
int opennpux_npu_tensor_plan_resolve(
    const struct opennpux_npu_tensor_plan *plan, uint32_t tensor_id,
    const struct opennpux_npu_tensor_plan_runtime *runtime,
    const struct opennpux_npu_tensor_plan_memory *memory,
    uint64_t *tensor_address, uint64_t *tensor_size);
int opennpux_npu_tensor_plan_resolve_scratch(
    const struct opennpux_npu_tensor_plan *plan, uint32_t tensor_id,
    uint32_t batch_size, uint32_t sequence_length, uint64_t scratch_address,
    uint64_t scratch_size, uint64_t *tensor_address, uint64_t *tensor_size);
int opennpux_npu_tensor_plan_resolve_command(
    const struct opennpux_npu_tensor_plan *plan, uint32_t command_id,
    const struct opennpux_npu_tensor_plan_runtime *runtime,
    const struct opennpux_npu_tensor_plan_memory *memory,
    struct opennpux_npu_command_tensor_views *views);

#ifdef __cplusplus
}
#endif

#endif
