#ifndef OPENNPUX_NPU_WEIGHT_PAGER_H
#define OPENNPUX_NPU_WEIGHT_PAGER_H

#include "opennpux/model_package.h"
#include "opennpux/npu_weight_ranges.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_NPU_WEIGHT_PAGE_SIZE UINT32_C(4096)

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

#ifdef __cplusplus
}
#endif

#endif
