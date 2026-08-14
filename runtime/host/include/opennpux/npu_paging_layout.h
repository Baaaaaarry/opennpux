#ifndef OPENNPUX_NPU_PAGING_LAYOUT_H
#define OPENNPUX_NPU_PAGING_LAYOUT_H

#include "opennpux/npu_submission.h"
#include "opennpux/npu_weight_queue.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_NPU_PAGING_QUEUE_CAPACITY_DEFAULT UINT32_C(64)
#define OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT UINT32_C(64)
#define OPENNPUX_NPU_PAGING_TRANSFER_DEFAULT UINT32_C(65536)

struct opennpux_npu_paging_layout {
    uint64_t control_offset;
    uint64_t control_size;
    uint64_t queue_offset;
    uint64_t queue_size;
    uint64_t cache_offset;
    uint64_t cache_size;
    uint64_t required_size;
    uint32_t queue_capacity;
    uint32_t cache_slots;
    uint32_t transfer_size;
    uint32_t reserved;
};

int opennpux_npu_paging_layout_plan(
    uint64_t control_size, uint64_t window_size, uint32_t queue_capacity,
    uint32_t cache_slots, uint32_t transfer_size,
    struct opennpux_npu_paging_layout *layout);
int opennpux_npu_paging_layout_init_queue(
    void *window, size_t window_size,
    const struct opennpux_npu_paging_layout *layout,
    struct opennpux_npu_weight_queue *queue);
int opennpux_npu_paging_layout_bindings(
    const struct opennpux_npu_paging_layout *layout,
    uint64_t device_window_base, uint32_t queue_tensor_id,
    uint32_t cache_tensor_id,
    struct opennpux_npu_tensor_binding *queue_binding,
    struct opennpux_npu_tensor_binding *cache_binding);

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_paging_layout) == 72);
#else
_Static_assert(sizeof(struct opennpux_npu_paging_layout) == 72,
               "NPU paging layout size changed");
#endif

#ifdef __cplusplus
}
#endif

#endif
