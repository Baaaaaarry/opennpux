#include "opennpux/npu_paging_layout.h"

#include <errno.h>
#include <string.h>

static int
align_up(uint64_t value, uint64_t alignment, uint64_t *result)
{
    if (alignment == 0 || (alignment & (alignment - 1)) != 0 ||
        value > UINT64_MAX - (alignment - 1)) {
        errno = EOVERFLOW;
        return -1;
    }
    *result = (value + alignment - 1) & ~(alignment - 1);
    return 0;
}

int
opennpux_npu_paging_layout_plan(
    uint64_t control_size, uint64_t window_size, uint32_t queue_capacity,
    uint32_t cache_slots, uint32_t transfer_size,
    struct opennpux_npu_paging_layout *layout)
{
    if (control_size == 0 || window_size == 0 || queue_capacity == 0 ||
        cache_slots == 0 || transfer_size < OPENNPUX_NPU_WEIGHT_PAGE_SIZE ||
        transfer_size > OPENNPUX_NPU_WEIGHT_TRANSFER_MAX ||
        (transfer_size & (transfer_size - 1)) != 0 || layout == NULL) {
        errno = EINVAL;
        return -1;
    }
    size_t queue_size = 0;
    uint64_t queue_offset = 0;
    uint64_t cache_offset = 0;
    if (opennpux_npu_weight_queue_size(queue_capacity, &queue_size) != 0 ||
        align_up(control_size, OPENNPUX_NPU_RECORD_ALIGNMENT,
                 &queue_offset) != 0 ||
        queue_offset > UINT64_MAX - queue_size ||
        align_up(queue_offset + queue_size, transfer_size,
                 &cache_offset) != 0 ||
        cache_slots > UINT64_MAX / transfer_size) {
        errno = EOVERFLOW;
        return -1;
    }
    const uint64_t cache_size = (uint64_t)cache_slots * transfer_size;
    if (cache_offset > UINT64_MAX - cache_size) {
        errno = EOVERFLOW;
        return -1;
    }
    const uint64_t required_size = cache_offset + cache_size;
    if (required_size > window_size) {
        errno = ENOSPC;
        return -1;
    }
    memset(layout, 0, sizeof(*layout));
    layout->control_size = control_size;
    layout->queue_offset = queue_offset;
    layout->queue_size = queue_size;
    layout->cache_offset = cache_offset;
    layout->cache_size = cache_size;
    layout->required_size = required_size;
    layout->queue_capacity = queue_capacity;
    layout->cache_slots = cache_slots;
    layout->transfer_size = transfer_size;
    return 0;
}

int
opennpux_npu_paging_layout_init_queue(
    void *window, size_t window_size,
    const struct opennpux_npu_paging_layout *layout,
    struct opennpux_npu_weight_queue *queue)
{
    if (window == NULL || layout == NULL || queue == NULL ||
        layout->queue_offset > window_size ||
        layout->queue_size > window_size - layout->queue_offset ||
        layout->required_size > window_size) {
        errno = EINVAL;
        return -1;
    }
    return opennpux_npu_weight_queue_init(
        (uint8_t *)window + layout->queue_offset,
        (size_t)layout->queue_size, layout->queue_capacity, queue);
}

int
opennpux_npu_paging_layout_bindings(
    const struct opennpux_npu_paging_layout *layout,
    uint64_t device_window_base, uint32_t queue_tensor_id,
    uint32_t cache_tensor_id,
    struct opennpux_npu_tensor_binding *queue_binding,
    struct opennpux_npu_tensor_binding *cache_binding)
{
    if (layout == NULL || queue_binding == NULL || cache_binding == NULL ||
        device_window_base > UINT64_MAX - layout->required_size) {
        errno = EINVAL;
        return -1;
    }
    memset(queue_binding, 0, sizeof(*queue_binding));
    queue_binding->tensor_id = queue_tensor_id;
    queue_binding->flags = OPENNPUX_NPU_BIND_READ | OPENNPUX_NPU_BIND_WRITE |
        OPENNPUX_NPU_BIND_PAGE_QUEUE;
    queue_binding->data_type = OPENNPUX_NPU_DTYPE_INT8;
    queue_binding->rank = 1;
    queue_binding->device_address = device_window_base + layout->queue_offset;
    queue_binding->byte_size = layout->queue_size;
    queue_binding->dimensions[0] = (uint32_t)layout->queue_size;

    memset(cache_binding, 0, sizeof(*cache_binding));
    cache_binding->tensor_id = cache_tensor_id;
    cache_binding->flags = OPENNPUX_NPU_BIND_READ | OPENNPUX_NPU_BIND_WRITE |
        OPENNPUX_NPU_BIND_WEIGHT | OPENNPUX_NPU_BIND_PAGE_CACHE;
    cache_binding->data_type = OPENNPUX_NPU_DTYPE_INT8;
    cache_binding->rank = 1;
    cache_binding->device_address = device_window_base + layout->cache_offset;
    cache_binding->byte_size = layout->cache_size;
    if (layout->cache_size > UINT32_MAX) {
        errno = EOVERFLOW;
        return -1;
    }
    cache_binding->dimensions[0] = (uint32_t)layout->cache_size;
    return 0;
}
