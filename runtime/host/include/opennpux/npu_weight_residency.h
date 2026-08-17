#ifndef OPENNPUX_NPU_WEIGHT_RESIDENCY_H
#define OPENNPUX_NPU_WEIGHT_RESIDENCY_H

#include "opennpux/npu_weight_pager.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC UINT32_C(0x5358504e)
#define OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION UINT32_C(1)
#define OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID UINT32_C(1)

struct opennpux_npu_weight_residency_header {
    uint32_t magic, version, header_size, record_size;
    uint32_t capacity, valid_records;
    uint64_t generation;
    uint64_t reserved[4];
};

struct opennpux_npu_weight_residency_record {
    uint32_t command_id, role_id, component_id, shard_index;
    uint64_t expert_id, range_file_offset, range_size, page_file_offset;
    uint32_t cache_slot, page_size, flags, reserved;
};

static inline size_t
opennpux_npu_weight_residency_size(uint32_t capacity)
{
    return sizeof(struct opennpux_npu_weight_residency_header) +
        (size_t)capacity * sizeof(struct opennpux_npu_weight_residency_record);
}

static inline int
opennpux_npu_weight_residency_init(void *storage, size_t storage_size,
                                  uint32_t capacity)
{
    const size_t required = opennpux_npu_weight_residency_size(capacity);
    if (storage == NULL || capacity == 0 || required < sizeof(
            struct opennpux_npu_weight_residency_header) ||
        required > storage_size) {
        return -1;
    }
    memset(storage, 0, required);
    struct opennpux_npu_weight_residency_header *header = storage;
    header->magic = OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC;
    header->version = OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION;
    header->header_size = sizeof(*header);
    header->record_size = sizeof(struct opennpux_npu_weight_residency_record);
    header->capacity = capacity;
    return 0;
}

static inline int
opennpux_npu_weight_residency_publish(void *storage, size_t storage_size,
                                     const struct opennpux_npu_page_fault *fault)
{
    if (storage == NULL || fault == NULL) {
        return -1;
    }
    struct opennpux_npu_weight_residency_header *header = storage;
    const size_t required = opennpux_npu_weight_residency_size(header->capacity);
    if (header->magic != OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC ||
        header->version != OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION ||
        header->header_size != sizeof(*header) ||
        header->record_size != sizeof(struct opennpux_npu_weight_residency_record) ||
        fault->cache_slot >= header->capacity || required > storage_size) {
        return -1;
    }
    struct opennpux_npu_weight_residency_record *records =
        (struct opennpux_npu_weight_residency_record *)(header + 1);
    struct opennpux_npu_weight_residency_record *record =
        &records[fault->cache_slot];
    const int was_valid = (record->flags &
        OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID) != 0;
    record->flags = 0;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    record->command_id = fault->command_id;
    record->role_id = fault->role_id;
    record->component_id = fault->component_id;
    record->shard_index = fault->shard_index;
    record->expert_id = fault->expert_id;
    record->range_file_offset = fault->range_file_offset;
    record->range_size = fault->range_size;
    record->page_file_offset = fault->file_offset;
    record->cache_slot = fault->cache_slot;
    record->page_size = fault->page_size;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    record->flags = OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID;
    if (!was_valid) {
        ++header->valid_records;
    }
    ++header->generation;
    return 0;
}

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_weight_residency_header) == 64);
static_assert(sizeof(struct opennpux_npu_weight_residency_record) == 64);
#else
_Static_assert(sizeof(struct opennpux_npu_weight_residency_header) == 64,
               "NPU residency header ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_weight_residency_record) == 64,
               "NPU residency record ABI size changed");
#endif

#endif
