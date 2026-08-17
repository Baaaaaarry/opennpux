#include "opennpux/model_package.h"
#include "opennpux/npu_weight_pager.h"
#include "opennpux/npu_weight_ranges.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s (errno=%d)\n", message, errno);
        exit(1);
    }
}

int
main(int argc, char **argv)
{
    check(argc == 3, "usage: npu_weight_pager_test <model.npxm> <model.npxr>");
    struct opennpux_model_package_info model;
    struct opennpux_npu_weight_ranges ranges;
    check(opennpux_model_package_load(argv[1], &model) == 0,
          "model package load failed");
    check(opennpux_npu_weight_ranges_load(argv[2], &ranges) == 0,
          "weight range load failed");

    struct opennpux_npu_weight_page_cursor cursor;
    struct opennpux_npu_weight_page_request request;
    check(opennpux_npu_weight_page_cursor_begin(
              &ranges, 0, NULL, 0, &cursor) == 0,
          "static pager begin failed");
    check(opennpux_npu_weight_page_cursor_next(&cursor, &request) == 1,
          "static page request missing");
    check(request.page_size == OPENNPUX_NPU_WEIGHT_PAGE_SIZE,
          "default page request size mismatch");
    check(request.role_id == ranges.records[0].role_id &&
              request.component_id == ranges.records[0].component_id &&
              request.range_file_offset == ranges.records[0].file_offset &&
              request.range_size == ranges.records[0].byte_size,
          "page request tensor identity mismatch");
    unsigned char page[OPENNPUX_NPU_WEIGHT_PAGE_SIZE];
    check(opennpux_npu_weight_page_read(
              argv[1], &model, &request, page, sizeof(page)) == 0,
          "static page read failed");
    const uint64_t tensor_offset = ranges.records[0].file_offset -
        request.file_offset;
    check(page[tensor_offset] == 0 && page[tensor_offset + 1] == 1,
          "static page payload mismatch");
    struct opennpux_npu_weight_cache cache;
    struct opennpux_npu_weight_cache_entry cache_entries[1];
    unsigned char cache_storage[OPENNPUX_NPU_WEIGHT_PAGE_SIZE];
    const void *cached_page;
    uint32_t cache_slot;
    uint32_t cache_hit;
    check(opennpux_npu_weight_cache_init(
              &cache, cache_entries, cache_storage, 1) == 0,
          "weight cache init failed");
    check(opennpux_npu_weight_cache_acquire(
              &cache, argv[1], &model, &request, &cached_page,
              &cache_slot, &cache_hit) == 0 && !cache_hit && cache_slot == 0,
          "first cache miss failed");
    check(opennpux_npu_weight_cache_acquire(
              &cache, argv[1], &model, &request, &cached_page,
              &cache_slot, &cache_hit) == 0 && cache_hit,
          "cache hit failed");

    uint32_t expert_command = UINT32_MAX;
    for (uint32_t index = 0; index < ranges.header->range_count; ++index) {
        if (ranges.records[index].expert_id == 7) {
            expert_command = ranges.records[index].command_id;
            break;
        }
    }
    check(expert_command != UINT32_MAX, "expert command missing");
    const uint64_t active[] = {7};
    check(opennpux_npu_weight_page_cursor_begin(
              &ranges, expert_command, active, 1, &cursor) == 0 &&
              opennpux_npu_weight_page_cursor_next(&cursor, &request) == 1 &&
              request.expert_id == 7,
          "active expert page missing");
    check(opennpux_npu_weight_cache_acquire(
              &cache, argv[1], &model, &request, &cached_page,
              &cache_slot, &cache_hit) == 0 && !cache_hit,
          "expert page cache miss failed");
    check(cache.stats.hits == 1 && cache.stats.misses == 2 &&
              cache.stats.evictions == 1 &&
              cache.stats.bytes_read == 2 * OPENNPUX_NPU_WEIGHT_PAGE_SIZE,
          "cache statistics mismatch");
    struct opennpux_npu_page_fault fault;
    check(opennpux_npu_page_fault_init(&fault, 11, &request) == 0 &&
              fault.state == OPENNPUX_NPU_PAGE_FAULT_PENDING &&
              fault.page_size == OPENNPUX_NPU_WEIGHT_PAGE_SIZE &&
              fault.role_id == request.role_id &&
              fault.component_id == request.component_id &&
              fault.range_file_offset == request.range_file_offset &&
              fault.range_size == request.range_size,
          "page fault publication failed");
    check(opennpux_npu_page_fault_service(
              &fault, &cache, argv[1], &model, &cached_page,
              &cache_hit) == 0 &&
              fault.state == OPENNPUX_NPU_PAGE_FAULT_READY &&
              fault.cache_slot == 0 && cache_hit,
          "page fault service failed");
    const uint64_t inactive[] = {0};
    check(opennpux_npu_weight_page_cursor_begin(
              &ranges, expert_command, inactive, 1, &cursor) == 0 &&
              opennpux_npu_weight_page_cursor_next(&cursor, &request) == 0,
          "inactive expert page was scheduled");

    const uint32_t transfer_size = UINT32_C(65536);
    unsigned char *transfer_storage = malloc(transfer_size);
    check(transfer_storage != NULL, "large transfer allocation failed");
    check(opennpux_npu_weight_page_cursor_begin_sized(
              &ranges, 0, NULL, 0, transfer_size, &cursor) == 0 &&
              opennpux_npu_weight_page_cursor_next(&cursor, &request) == 1 &&
              request.page_size == transfer_size &&
              request.file_offset % transfer_size == 0,
          "large transfer request failed");
    check(opennpux_npu_weight_cache_init_sized(
              &cache, cache_entries, transfer_storage, 1,
              transfer_size) == 0,
          "large transfer cache init failed");
    check(opennpux_npu_weight_cache_acquire(
              &cache, argv[1], &model, &request, &cached_page,
              &cache_slot, &cache_hit) == 0 && !cache_hit &&
              cache.stats.bytes_read == transfer_size,
          "large transfer cache fill failed");
    free(transfer_storage);

    opennpux_npu_weight_ranges_unload(&ranges);
    puts("PASS: NPU weight pager tests");
    return 0;
}
