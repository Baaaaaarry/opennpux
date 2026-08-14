#include "opennpux/npu_weight_queue.h"

#include <errno.h>
#include <string.h>

static uint64_t
load_acquire(const uint64_t *value)
{
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

static void
store_release(uint64_t *target, uint64_t value)
{
    __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

static uint32_t
load_state(const uint32_t *value)
{
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

static void
store_state(uint32_t *target, uint32_t value)
{
    __atomic_store_n(target, value, __ATOMIC_RELEASE);
}

int
opennpux_npu_weight_queue_size(uint32_t capacity, size_t *size)
{
    if (capacity == 0 || size == NULL) {
        errno = EINVAL;
        return -1;
    }
    const size_t entries_size =
        (size_t)capacity * sizeof(struct opennpux_npu_page_fault);
    if (entries_size / sizeof(struct opennpux_npu_page_fault) != capacity ||
        entries_size > SIZE_MAX -
            sizeof(struct opennpux_npu_weight_queue_header)) {
        errno = EOVERFLOW;
        return -1;
    }
    *size = sizeof(struct opennpux_npu_weight_queue_header) + entries_size;
    return 0;
}

int
opennpux_npu_weight_queue_attach(
    void *storage, size_t storage_size,
    struct opennpux_npu_weight_queue *queue)
{
    if (storage == NULL || queue == NULL ||
        storage_size < sizeof(struct opennpux_npu_weight_queue_header) ||
        (uintptr_t)storage %
            _Alignof(struct opennpux_npu_weight_queue_header) != 0) {
        errno = EINVAL;
        return -1;
    }
    struct opennpux_npu_weight_queue_header *header = storage;
    size_t required = 0;
    if (header->magic != OPENNPUX_NPU_WEIGHT_QUEUE_MAGIC ||
        header->version != OPENNPUX_NPU_WEIGHT_QUEUE_VERSION ||
        header->header_size != sizeof(*header) ||
        header->entry_size != sizeof(struct opennpux_npu_page_fault) ||
        opennpux_npu_weight_queue_size(header->capacity, &required) != 0 ||
        required > storage_size || header->service_index > header->producer_index ||
        header->retire_index > header->service_index ||
        header->producer_index - header->retire_index > header->capacity) {
        errno = EINVAL;
        return -1;
    }
    queue->header = header;
    queue->entries = (struct opennpux_npu_page_fault *)(header + 1);
    return 0;
}

int
opennpux_npu_weight_queue_init(
    void *storage, size_t storage_size, uint32_t capacity,
    struct opennpux_npu_weight_queue *queue)
{
    size_t required = 0;
    if (opennpux_npu_weight_queue_size(capacity, &required) != 0 ||
        storage == NULL || storage_size < required) {
        errno = EINVAL;
        return -1;
    }
    memset(storage, 0, required);
    struct opennpux_npu_weight_queue_header *header = storage;
    header->magic = OPENNPUX_NPU_WEIGHT_QUEUE_MAGIC;
    header->version = OPENNPUX_NPU_WEIGHT_QUEUE_VERSION;
    header->header_size = sizeof(*header);
    header->entry_size = sizeof(struct opennpux_npu_page_fault);
    header->capacity = capacity;
    return opennpux_npu_weight_queue_attach(storage, storage_size, queue);
}

int
opennpux_npu_weight_queue_publish(
    struct opennpux_npu_weight_queue *queue, uint64_t sequence,
    const struct opennpux_npu_weight_page_request *request)
{
    if (queue == NULL || queue->header == NULL || queue->entries == NULL) {
        errno = EINVAL;
        return -1;
    }
    const uint64_t producer = load_acquire(&queue->header->producer_index);
    const uint64_t retire = load_acquire(&queue->header->retire_index);
    if (producer - retire >= queue->header->capacity) {
        __atomic_add_fetch(&queue->header->backpressure_count, 1,
                           __ATOMIC_RELAXED);
        errno = EAGAIN;
        return -1;
    }
    struct opennpux_npu_page_fault fault;
    if (opennpux_npu_page_fault_init(&fault, sequence, request) != 0) {
        return -1;
    }
    struct opennpux_npu_page_fault *entry =
        &queue->entries[producer % queue->header->capacity];
    if (load_state(&entry->state) != OPENNPUX_NPU_PAGE_FAULT_EMPTY) {
        errno = EBUSY;
        return -1;
    }
    fault.state = OPENNPUX_NPU_PAGE_FAULT_EMPTY;
    *entry = fault;
    store_state(&entry->state, OPENNPUX_NPU_PAGE_FAULT_PENDING);
    store_release(&queue->header->producer_index, producer + 1);
    return 0;
}

int
opennpux_npu_weight_queue_service_next(
    struct opennpux_npu_weight_queue *queue,
    struct opennpux_npu_weight_cache *cache, const char *manifest_path,
    const struct opennpux_model_package_info *model, const void **page,
    uint32_t *cache_hit)
{
    if (queue == NULL || queue->header == NULL || queue->entries == NULL ||
        page == NULL || cache_hit == NULL) {
        errno = EINVAL;
        return -1;
    }
    const uint64_t service = load_acquire(&queue->header->service_index);
    const uint64_t producer = load_acquire(&queue->header->producer_index);
    if (service == producer) {
        return 0;
    }
    struct opennpux_npu_page_fault *entry =
        &queue->entries[service % queue->header->capacity];
    if (load_state(&entry->state) != OPENNPUX_NPU_PAGE_FAULT_PENDING) {
        errno = EPROTO;
        return -1;
    }
    const int result = opennpux_npu_page_fault_service(
        entry, cache, manifest_path, model, page, cache_hit);
    const uint32_t final_state = result == 0 ?
        OPENNPUX_NPU_PAGE_FAULT_READY : OPENNPUX_NPU_PAGE_FAULT_ERROR;
    store_state(&entry->state, final_state);
    store_release(&queue->header->service_index, service + 1);
    return 1;
}

int
opennpux_npu_weight_queue_consume(
    struct opennpux_npu_weight_queue *queue,
    struct opennpux_npu_page_fault *completion)
{
    if (queue == NULL || queue->header == NULL || queue->entries == NULL ||
        completion == NULL) {
        errno = EINVAL;
        return -1;
    }
    const uint64_t retire = load_acquire(&queue->header->retire_index);
    const uint64_t service = load_acquire(&queue->header->service_index);
    if (retire == service) {
        return 0;
    }
    struct opennpux_npu_page_fault *entry =
        &queue->entries[retire % queue->header->capacity];
    const uint32_t state = load_state(&entry->state);
    if (state != OPENNPUX_NPU_PAGE_FAULT_READY &&
        state != OPENNPUX_NPU_PAGE_FAULT_ERROR) {
        errno = EPROTO;
        return -1;
    }
    *completion = *entry;
    store_state(&entry->state, OPENNPUX_NPU_PAGE_FAULT_EMPTY);
    store_release(&queue->header->retire_index, retire + 1);
    return 1;
}
