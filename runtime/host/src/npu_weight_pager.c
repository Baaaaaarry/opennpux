#include "opennpux/npu_weight_pager.h"

#include <errno.h>
#include <string.h>

static int
valid_page_size(uint32_t page_size)
{
    return page_size >= OPENNPUX_NPU_WEIGHT_PAGE_SIZE &&
        page_size <= OPENNPUX_NPU_WEIGHT_TRANSFER_MAX &&
        (page_size & (page_size - 1)) == 0;
}

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
    return opennpux_npu_weight_page_cursor_begin_sized(
        ranges, command_id, active_experts, active_expert_count,
        OPENNPUX_NPU_WEIGHT_PAGE_SIZE, cursor);
}

int
opennpux_npu_weight_page_cursor_begin_sized(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    const uint64_t *active_experts, uint32_t active_expert_count,
    uint32_t page_size, struct opennpux_npu_weight_page_cursor *cursor)
{
    if (cursor == NULL ||
        (active_expert_count != 0 && active_experts == NULL) ||
        !valid_page_size(page_size)) {
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
    cursor->page_size = page_size;
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
            cursor->pages_in_record = 0;
            continue;
        }
        if (cursor->next_page_offset == 0) {
            cursor->next_page_offset = record->file_offset &
                ~(uint64_t)(cursor->page_size - 1);
            cursor->range_end = record->file_offset + record->byte_size;
        }
        if (cursor->next_page_offset >= cursor->range_end ||
            (cursor->max_pages_per_record != 0 &&
             cursor->pages_in_record >= cursor->max_pages_per_record)) {
            ++cursor->record_index;
            cursor->next_page_offset = 0;
            cursor->pages_in_record = 0;
            continue;
        }
        const uint64_t page_offset = cursor->next_page_offset;
        cursor->next_page_offset += cursor->page_size;
        ++cursor->pages_in_record;
        request->command_id = cursor->command_id;
        request->shard_index = record->shard_index;
        request->file_offset = page_offset;
        request->expert_id = record->expert_id;
        request->role_id = record->role_id;
        request->component_id = record->component_id;
        request->range_file_offset = record->file_offset;
        request->range_size = record->byte_size;
        request->page_size = cursor->page_size;
        return 1;
    }
    return 0;
}

int
opennpux_npu_weight_page_cursor_limit_records(
    struct opennpux_npu_weight_page_cursor *cursor,
    uint32_t max_pages_per_record)
{
    if (cursor == NULL || cursor->next_page_offset != 0 ||
        cursor->record_index != 0) {
        errno = EINVAL;
        return -1;
    }
    cursor->max_pages_per_record = max_pages_per_record;
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
        page == NULL || !valid_page_size(page_size) ||
        request->page_size != page_size ||
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

int
opennpux_npu_weight_cache_init(
    struct opennpux_npu_weight_cache *cache,
    struct opennpux_npu_weight_cache_entry *entries, void *storage,
    uint32_t slot_count)
{
    return opennpux_npu_weight_cache_init_sized(
        cache, entries, storage, slot_count,
        OPENNPUX_NPU_WEIGHT_PAGE_SIZE);
}

int
opennpux_npu_weight_cache_init_sized(
    struct opennpux_npu_weight_cache *cache,
    struct opennpux_npu_weight_cache_entry *entries, void *storage,
    uint32_t slot_count, uint32_t page_size)
{
    if (cache == NULL || entries == NULL || storage == NULL || slot_count == 0 ||
        !valid_page_size(page_size)) {
        errno = EINVAL;
        return -1;
    }
    memset(cache, 0, sizeof(*cache));
    memset(entries, 0, slot_count * sizeof(*entries));
    cache->entries = entries;
    cache->storage = storage;
    cache->slot_count = slot_count;
    cache->page_size = page_size;
    return 0;
}

int
opennpux_npu_weight_cache_acquire(
    struct opennpux_npu_weight_cache *cache, const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_page_request *request,
    const void **page, uint32_t *slot, uint32_t *cache_hit)
{
    if (cache == NULL || cache->entries == NULL || cache->storage == NULL ||
        request == NULL || page == NULL || slot == NULL || cache_hit == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (request->page_size != cache->page_size) {
        errno = EINVAL;
        return -1;
    }
    ++cache->clock;
    uint32_t victim = 0;
    for (uint32_t index = 0; index < cache->slot_count; ++index) {
        struct opennpux_npu_weight_cache_entry *entry = &cache->entries[index];
        if (entry->valid && entry->shard_index == request->shard_index &&
            entry->file_offset == request->file_offset) {
            entry->last_use = cache->clock;
            ++cache->stats.hits;
            *page = cache->storage +
                (size_t)index * cache->page_size;
            *slot = index;
            *cache_hit = 1;
            return 0;
        }
        if (!entry->valid ||
            (cache->entries[victim].valid &&
             entry->last_use < cache->entries[victim].last_use)) {
            victim = index;
        }
    }
    struct opennpux_npu_weight_cache_entry *entry = &cache->entries[victim];
    if (entry->valid) {
        ++cache->stats.evictions;
    }
    uint8_t *destination = cache->storage +
        (size_t)victim * cache->page_size;
    if (opennpux_npu_weight_page_read(
            manifest_path, model, request, destination,
            cache->page_size) != 0) {
        return -1;
    }
    entry->valid = 1;
    entry->shard_index = request->shard_index;
    entry->file_offset = request->file_offset;
    entry->last_use = cache->clock;
    ++cache->stats.misses;
    cache->stats.bytes_read += cache->page_size;
    *page = destination;
    *slot = victim;
    *cache_hit = 0;
    return 0;
}

int
opennpux_npu_page_fault_init(
    struct opennpux_npu_page_fault *fault, uint64_t sequence,
    const struct opennpux_npu_weight_page_request *request)
{
    if (fault == NULL || request == NULL || sequence == 0) {
        errno = EINVAL;
        return -1;
    }
    memset(fault, 0, sizeof(*fault));
    fault->magic = OPENNPUX_NPU_PAGE_FAULT_MAGIC;
    fault->version = OPENNPUX_NPU_PAGE_FAULT_VERSION;
    fault->struct_size = sizeof(*fault);
    fault->state = OPENNPUX_NPU_PAGE_FAULT_PENDING;
    fault->sequence = sequence;
    fault->command_id = request->command_id;
    fault->shard_index = request->shard_index;
    fault->file_offset = request->file_offset;
    fault->expert_id = request->expert_id;
    fault->role_id = request->role_id;
    fault->component_id = request->component_id;
    fault->range_file_offset = request->range_file_offset;
    fault->range_size = request->range_size;
    fault->page_size = request->page_size;
    return 0;
}

int
opennpux_npu_page_fault_service(
    struct opennpux_npu_page_fault *fault,
    struct opennpux_npu_weight_cache *cache, const char *manifest_path,
    const struct opennpux_model_package_info *model, const void **page,
    uint32_t *cache_hit)
{
    if (fault == NULL || fault->magic != OPENNPUX_NPU_PAGE_FAULT_MAGIC ||
        fault->version != OPENNPUX_NPU_PAGE_FAULT_VERSION ||
        fault->struct_size != sizeof(*fault) ||
        fault->state != OPENNPUX_NPU_PAGE_FAULT_PENDING ||
        fault->sequence == 0 || page == NULL || cache_hit == NULL) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_weight_page_request request = {
        .command_id = fault->command_id,
        .shard_index = fault->shard_index,
        .file_offset = fault->file_offset,
        .expert_id = fault->expert_id,
        .role_id = fault->role_id,
        .component_id = fault->component_id,
        .range_file_offset = fault->range_file_offset,
        .range_size = fault->range_size,
        .page_size = fault->page_size,
    };
    uint32_t slot = 0;
    if (opennpux_npu_weight_cache_acquire(
            cache, manifest_path, model, &request, page, &slot,
            cache_hit) != 0) {
        fault->state = OPENNPUX_NPU_PAGE_FAULT_ERROR;
        fault->error_code = (uint32_t)errno;
        return -1;
    }
    fault->cache_slot = slot;
    fault->state = OPENNPUX_NPU_PAGE_FAULT_READY;
    return 0;
}
