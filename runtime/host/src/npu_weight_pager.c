#include "opennpux/npu_weight_pager.h"

#include <errno.h>
#include <string.h>

static int
expert_active(const struct opennpux_npu_weight_page_cursor *cursor,
              uint64_t expert_id)
{
    if (expert_id == OPENNPUX_NPU_WEIGHT_EXPERT_NONE) {
        return 1;
    }
    for (uint32_t index = 0; index < cursor->active_expert_count; ++index) {
        if (cursor->active_experts[index] == expert_id) {
            return 1;
        }
    }
    return 0;
}

int
opennpux_npu_weight_page_cursor_begin(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    const uint64_t *active_experts, uint32_t active_expert_count,
    struct opennpux_npu_weight_page_cursor *cursor)
{
    if (cursor == NULL ||
        (active_expert_count != 0 && active_experts == NULL)) {
        errno = EINVAL;
        return -1;
    }
    memset(cursor, 0, sizeof(*cursor));
    if (opennpux_npu_weight_ranges_for_command(
            ranges, command_id, &cursor->records,
            &cursor->record_count) != 0) {
        return -1;
    }
    cursor->command_id = command_id;
    cursor->active_experts = active_experts;
    cursor->active_expert_count = active_expert_count;
    return 0;
}

int
opennpux_npu_weight_page_cursor_next(
    struct opennpux_npu_weight_page_cursor *cursor,
    struct opennpux_npu_weight_page_request *request)
{
    if (cursor == NULL || request == NULL) {
        errno = EINVAL;
        return -1;
    }
    while (cursor->record_index < cursor->record_count) {
        const struct opennpux_npu_weight_range_record *record =
            &cursor->records[cursor->record_index];
        if (!expert_active(cursor, record->expert_id)) {
            ++cursor->record_index;
            cursor->next_page_offset = 0;
            continue;
        }
        if (cursor->next_page_offset == 0) {
            cursor->next_page_offset = record->file_offset &
                ~(uint64_t)(OPENNPUX_NPU_WEIGHT_PAGE_SIZE - 1);
            cursor->range_end = record->file_offset + record->byte_size;
        }
        if (cursor->next_page_offset >= cursor->range_end) {
            ++cursor->record_index;
            cursor->next_page_offset = 0;
            continue;
        }
        const uint64_t page_offset = cursor->next_page_offset;
        cursor->next_page_offset += OPENNPUX_NPU_WEIGHT_PAGE_SIZE;
        if (cursor->has_last_page &&
            cursor->last_shard_index == record->shard_index &&
            cursor->last_page_offset == page_offset) {
            continue;
        }
        cursor->has_last_page = 1;
        cursor->last_shard_index = record->shard_index;
        cursor->last_page_offset = page_offset;
        request->command_id = cursor->command_id;
        request->shard_index = record->shard_index;
        request->file_offset = page_offset;
        request->expert_id = record->expert_id;
        return 1;
    }
    return 0;
}

int
opennpux_npu_weight_page_read(
    const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_page_request *request,
    void *page, uint32_t page_size)
{
    if (manifest_path == NULL || model == NULL || request == NULL ||
        page == NULL || page_size != OPENNPUX_NPU_WEIGHT_PAGE_SIZE ||
        request->shard_index >= model->shard_count ||
        request->file_offset >= model->shards[request->shard_index].size) {
        errno = EINVAL;
        return -1;
    }
    memset(page, 0, page_size);
    const uint64_t remaining =
        model->shards[request->shard_index].size - request->file_offset;
    const uint32_t bytes = remaining < page_size ? (uint32_t)remaining : page_size;
    return opennpux_model_package_read_shard_range(
        manifest_path, model, request->shard_index, request->file_offset,
        page, bytes);
}
