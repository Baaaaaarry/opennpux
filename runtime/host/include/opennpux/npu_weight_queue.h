#ifndef OPENNPUX_NPU_WEIGHT_QUEUE_H
#define OPENNPUX_NPU_WEIGHT_QUEUE_H

#include "opennpux/npu_weight_pager.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_NPU_WEIGHT_QUEUE_MAGIC UINT32_C(0x5158504e)
#define OPENNPUX_NPU_WEIGHT_QUEUE_VERSION UINT32_C(3)

struct opennpux_npu_weight_queue_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t entry_size;
    uint32_t capacity;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
    uint32_t producer_index;
    uint32_t service_index;
    uint32_t retire_index;
    uint32_t backpressure_count;
    uint64_t reserved[2];
};

struct opennpux_npu_weight_queue {
    struct opennpux_npu_weight_queue_header *header;
    struct opennpux_npu_page_fault *entries;
};

int opennpux_npu_weight_queue_size(uint32_t capacity, size_t *size);
int opennpux_npu_weight_queue_init(
    void *storage, size_t storage_size, uint32_t capacity,
    struct opennpux_npu_weight_queue *queue);
int opennpux_npu_weight_queue_attach(
    void *storage, size_t storage_size,
    struct opennpux_npu_weight_queue *queue);
int opennpux_npu_weight_queue_publish(
    struct opennpux_npu_weight_queue *queue, uint64_t sequence,
    const struct opennpux_npu_weight_page_request *request);
int opennpux_npu_weight_queue_service_next(
    struct opennpux_npu_weight_queue *queue,
    struct opennpux_npu_weight_cache *cache, const char *manifest_path,
    const struct opennpux_model_package_info *model, const void **page,
    uint32_t *cache_hit);
int opennpux_npu_weight_queue_consume(
    struct opennpux_npu_weight_queue *queue,
    struct opennpux_npu_page_fault *completion);

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_weight_queue_header) == 64);
#else
_Static_assert(sizeof(struct opennpux_npu_weight_queue_header) == 64,
               "NPU weight queue header ABI size changed");
#endif

#ifdef __cplusplus
}
#endif

#endif
