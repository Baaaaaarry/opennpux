#ifndef OPENNPUX_NPU_WEIGHT_PAGER_H
#define OPENNPUX_NPU_WEIGHT_PAGER_H

#include "opennpux/model_package.h"
#include "opennpux/npu_weight_ranges.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_NPU_WEIGHT_PAGE_SIZE UINT32_C(4096)
#define OPENNPUX_NPU_PAGE_FAULT_MAGIC UINT32_C(0x4658504e)
#define OPENNPUX_NPU_PAGE_FAULT_VERSION UINT32_C(1)

enum opennpux_npu_page_fault_state {
    OPENNPUX_NPU_PAGE_FAULT_EMPTY = 0,
    OPENNPUX_NPU_PAGE_FAULT_PENDING = 1,
    OPENNPUX_NPU_PAGE_FAULT_READY = 2,
    OPENNPUX_NPU_PAGE_FAULT_ERROR = 3,
};

struct opennpux_npu_page_fault {
    uint32_t magic;
    uint32_t version;
    uint32_t struct_size;
    uint32_t state;
    uint64_t sequence;
    uint32_t command_id;
    uint32_t shard_index;
    uint64_t file_offset;
    uint64_t expert_id;
    uint32_t cache_slot;
    uint32_t error_code;
    uint64_t reserved;
};

struct opennpux_npu_weight_page_request {
    uint32_t command_id;
    uint32_t shard_index;
    uint64_t file_offset;
    uint64_t expert_id;
};

struct opennpux_npu_weight_page_cursor {
    const struct opennpux_npu_weight_range_record *records;
    uint32_t record_count;
    uint32_t record_index;
    uint64_t next_page_offset;
    uint64_t range_end;
    uint32_t command_id;
    const uint64_t *active_experts;
    uint32_t active_expert_count;
    uint32_t last_shard_index;
    uint64_t last_page_offset;
    uint32_t has_last_page;
};

struct opennpux_npu_weight_cache_entry {
    uint32_t valid;
    uint32_t shard_index;
    uint64_t file_offset;
    uint64_t last_use;
};

struct opennpux_npu_weight_cache_stats {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_read;
};

struct opennpux_npu_weight_cache {
    struct opennpux_npu_weight_cache_entry *entries;
    uint8_t *storage;
    uint32_t slot_count;
    uint64_t clock;
    struct opennpux_npu_weight_cache_stats stats;
};

int opennpux_npu_weight_page_cursor_begin(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    const uint64_t *active_experts, uint32_t active_expert_count,
    struct opennpux_npu_weight_page_cursor *cursor);
int opennpux_npu_weight_page_cursor_next(
    struct opennpux_npu_weight_page_cursor *cursor,
    struct opennpux_npu_weight_page_request *request);
int opennpux_npu_weight_page_read(
    const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_page_request *request,
    void *page, uint32_t page_size);
int opennpux_npu_weight_cache_init(
    struct opennpux_npu_weight_cache *cache,
    struct opennpux_npu_weight_cache_entry *entries, void *storage,
    uint32_t slot_count);
int opennpux_npu_weight_cache_acquire(
    struct opennpux_npu_weight_cache *cache, const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_page_request *request,
    const void **page, uint32_t *slot, uint32_t *cache_hit);
int opennpux_npu_page_fault_init(
    struct opennpux_npu_page_fault *fault, uint64_t sequence,
    const struct opennpux_npu_weight_page_request *request);
int opennpux_npu_page_fault_service(
    struct opennpux_npu_page_fault *fault,
    struct opennpux_npu_weight_cache *cache, const char *manifest_path,
    const struct opennpux_model_package_info *model, const void **page,
    uint32_t *cache_hit);

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_page_fault) == 64);
#else
_Static_assert(sizeof(struct opennpux_npu_page_fault) == 64,
               "NPU page fault ABI size changed");
#endif

#ifdef __cplusplus
}
#endif

#endif
