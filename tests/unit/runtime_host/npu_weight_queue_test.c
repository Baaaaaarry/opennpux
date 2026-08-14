#include "opennpux/model_package.h"
#include "opennpux/npu_weight_queue.h"
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
    check(argc == 3, "usage: npu_weight_queue_test <model.npxm> <model.npxr>");
    struct opennpux_model_package_info model;
    struct opennpux_npu_weight_ranges ranges;
    check(opennpux_model_package_load(argv[1], &model) == 0,
          "model package load failed");
    check(opennpux_npu_weight_ranges_load(argv[2], &ranges) == 0,
          "weight range load failed");

    struct opennpux_npu_weight_page_cursor cursor;
    struct opennpux_npu_weight_page_request request;
    check(opennpux_npu_weight_page_cursor_begin_sized(
              &ranges, 0, NULL, 0, UINT32_C(65536), &cursor) == 0 &&
              opennpux_npu_weight_page_cursor_next(&cursor, &request) == 1,
          "weight request missing");

    size_t queue_size = 0;
    check(opennpux_npu_weight_queue_size(2, &queue_size) == 0,
          "queue size failed");
    void *queue_storage = malloc(queue_size);
    unsigned char *cache_storage = malloc(UINT32_C(65536));
    check(queue_storage != NULL && cache_storage != NULL,
          "queue allocation failed");
    struct opennpux_npu_weight_queue queue;
    check(opennpux_npu_weight_queue_init(
              queue_storage, queue_size, 2, &queue) == 0,
          "queue init failed");
    check(opennpux_npu_weight_queue_publish(&queue, 1, &request) == 0 &&
              opennpux_npu_weight_queue_publish(&queue, 2, &request) == 0,
          "queue publish failed");
    check(opennpux_npu_weight_queue_publish(&queue, 3, &request) != 0 &&
              errno == EAGAIN && queue.header->backpressure_count == 1,
          "queue backpressure missing");

    struct opennpux_npu_weight_cache cache;
    struct opennpux_npu_weight_cache_entry cache_entry;
    check(opennpux_npu_weight_cache_init_sized(
              &cache, &cache_entry, cache_storage, 1, UINT32_C(65536)) == 0,
          "cache init failed");
    const void *page = NULL;
    uint32_t cache_hit = 0;
    check(opennpux_npu_weight_queue_service_next(
              &queue, &cache, argv[1], &model, &page, &cache_hit) == 1 &&
              cache_hit == 0,
          "first queue service failed");
    struct opennpux_npu_page_fault completion;
    check(opennpux_npu_weight_queue_consume(&queue, &completion) == 1 &&
              completion.sequence == 1 &&
              completion.state == OPENNPUX_NPU_PAGE_FAULT_READY &&
              completion.page_size == UINT32_C(65536),
          "first queue completion failed");
    check(opennpux_npu_weight_queue_service_next(
              &queue, &cache, argv[1], &model, &page, &cache_hit) == 1 &&
              cache_hit == 1 &&
              opennpux_npu_weight_queue_consume(&queue, &completion) == 1 &&
              completion.sequence == 2,
          "second queue completion failed");
    check(queue.header->producer_index == 2 &&
              queue.header->service_index == 2 &&
              queue.header->retire_index == 2,
          "queue indexes mismatch");
    check(opennpux_npu_weight_queue_publish(&queue, 3, &request) == 0,
          "queue slot was not reusable");

    free(cache_storage);
    free(queue_storage);
    opennpux_npu_weight_ranges_unload(&ranges);
    puts("PASS: NPU weight fault queue tests");
    return 0;
}
