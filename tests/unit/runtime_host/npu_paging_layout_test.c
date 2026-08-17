#include "opennpux/npu_paging_layout.h"

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
main(void)
{
    struct opennpux_npu_paging_layout layout;
    check(opennpux_npu_paging_layout_plan(
              UINT64_C(65536), UINT64_C(8) * 1024 * 1024,
              OPENNPUX_NPU_PAGING_QUEUE_CAPACITY_DEFAULT,
              OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT,
              OPENNPUX_NPU_PAGING_TRANSFER_DEFAULT, &layout) == 0,
          "default paging layout failed");
    check(layout.queue_offset == UINT64_C(65536) &&
              layout.queue_size == UINT64_C(4160) &&
              layout.cache_offset == UINT64_C(131072) &&
              layout.cache_size == UINT64_C(4194304) &&
              layout.required_size == UINT64_C(4325376),
          "default paging offsets mismatch");
    check(opennpux_npu_paging_layout_plan(
              UINT64_C(65536), UINT64_C(4) * 1024 * 1024,
              OPENNPUX_NPU_PAGING_QUEUE_CAPACITY_DEFAULT,
              OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT,
              OPENNPUX_NPU_PAGING_TRANSFER_DEFAULT, &layout) != 0 &&
              errno == ENOSPC,
          "undersized window was accepted");
    check(opennpux_npu_paging_layout_plan(
              UINT64_C(65536), UINT64_C(8) * 1024 * 1024,
              OPENNPUX_NPU_PAGING_QUEUE_CAPACITY_DEFAULT,
              OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT,
              OPENNPUX_NPU_PAGING_TRANSFER_DEFAULT, &layout) == 0,
          "paging layout restore failed");

    check(opennpux_npu_paging_layout_plan(
              UINT64_C(0x18000), UINT64_C(8) * 1024 * 1024,
              OPENNPUX_NPU_PAGING_QUEUE_CAPACITY_DEFAULT,
              OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT,
              OPENNPUX_NPU_PAGING_TRANSFER_DEFAULT, &layout) == 0,
          "large invocation paging layout failed");
    check(layout.queue_offset == UINT64_C(0x18000) &&
              layout.queue_offset >= layout.control_size &&
              layout.cache_offset == UINT64_C(0x20000) &&
              layout.cache_offset >= layout.queue_offset + layout.queue_size,
          "large invocation overlaps paging resources");

    check(opennpux_npu_paging_layout_plan(
              UINT64_C(65536), UINT64_C(8) * 1024 * 1024,
              OPENNPUX_NPU_PAGING_QUEUE_CAPACITY_DEFAULT,
              OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT,
              OPENNPUX_NPU_PAGING_TRANSFER_DEFAULT, &layout) == 0,
          "default paging layout second restore failed");

    void *window = calloc(1, (size_t)layout.required_size);
    check(window != NULL, "paging window allocation failed");
    struct opennpux_npu_weight_queue queue;
    check(opennpux_npu_paging_layout_init_queue(
              window, (size_t)layout.required_size, &layout, &queue) == 0 &&
              queue.header->capacity ==
                  OPENNPUX_NPU_PAGING_QUEUE_CAPACITY_DEFAULT,
          "paging queue placement failed");
    struct opennpux_npu_tensor_binding queue_binding;
    struct opennpux_npu_tensor_binding cache_binding;
    check(opennpux_npu_paging_layout_bindings(
              &layout, UINT64_C(0x20000000), 5, 6,
              &queue_binding, &cache_binding) == 0,
          "paging binding construction failed");
    check(queue_binding.device_address == UINT64_C(0x20010000) &&
              queue_binding.byte_size == layout.queue_size &&
              (queue_binding.flags & OPENNPUX_NPU_BIND_PAGE_QUEUE) != 0 &&
              cache_binding.device_address == UINT64_C(0x20020000) &&
              cache_binding.byte_size == layout.cache_size &&
              (cache_binding.flags & OPENNPUX_NPU_BIND_PAGE_CACHE) != 0,
          "paging binding values mismatch");
    free(window);
    puts("PASS: NPU shared paging layout tests");
    return 0;
}
