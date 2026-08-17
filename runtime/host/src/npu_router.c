#include "opennpux/npu_router.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static int
route_precedes(float logit, uint32_t expert_id,
               const opennpux_npu_route *route)
{
    return logit > route->logit ||
        (logit == route->logit && expert_id < route->expert_id);
}

int
opennpux_npu_router_topk(
    const float *logits, uint32_t expert_count, uint32_t active_count,
    opennpux_npu_route *routes)
{
    if (logits == NULL || routes == NULL || expert_count == 0 ||
        active_count == 0 || active_count > expert_count) {
        return -1;
    }
    for (uint32_t index = 0; index < active_count; ++index) {
        routes[index].expert_id = UINT32_MAX;
        routes[index].reserved = 0;
        routes[index].logit = -INFINITY;
        routes[index].weight = 0.0f;
    }

    for (uint32_t expert = 0; expert < expert_count; ++expert) {
        const float logit = logits[expert];
        if (!isfinite(logit)) {
            return -1;
        }
        uint32_t position = active_count;
        for (uint32_t index = 0; index < active_count; ++index) {
            if (route_precedes(logit, expert, &routes[index])) {
                position = index;
                break;
            }
        }
        if (position == active_count) {
            continue;
        }
        for (uint32_t index = active_count - 1; index > position; --index) {
            routes[index] = routes[index - 1];
        }
        routes[position].expert_id = expert;
        routes[position].logit = logit;
    }

    const float maximum = routes[0].logit;
    float sum = 0.0f;
    for (uint32_t index = 0; index < active_count; ++index) {
        routes[index].weight = expf(routes[index].logit - maximum);
        sum += routes[index].weight;
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        return -1;
    }
    for (uint32_t index = 0; index < active_count; ++index) {
        routes[index].weight /= sum;
    }
    return 0;
}

uint32_t
opennpux_npu_route_table_checksum(const void *buffer, size_t size)
{
    if (buffer == NULL || size == 0) {
        return 0;
    }
    const uint8_t *bytes = buffer;
    const size_t checksum_offset =
        offsetof(struct opennpux_npu_route_table_header, checksum);
    uint32_t checksum = UINT32_C(2166136261);
    for (size_t index = 0; index < size; ++index) {
        const uint8_t value = index >= checksum_offset &&
                index < checksum_offset + sizeof(uint32_t) ? 0 : bytes[index];
        checksum ^= value;
        checksum *= UINT32_C(16777619);
    }
    return checksum;
}

int
opennpux_npu_route_table_build(
    const struct opennpux_npu_route_record *records, uint32_t record_count,
    void *buffer, size_t capacity, size_t *table_size)
{
    if (records == NULL || record_count == 0 ||
        record_count > OPENNPUX_NPU_MAX_ACTIVE_EXPERTS || buffer == NULL ||
        table_size == NULL) {
        return -1;
    }
    const size_t size = sizeof(struct opennpux_npu_route_table_header) +
        (size_t)record_count * sizeof(*records);
    if (size > capacity || size > UINT32_MAX) {
        return -1;
    }
    memset(buffer, 0, size);
    struct opennpux_npu_route_table_header *header = buffer;
    header->magic = OPENNPUX_NPU_ROUTE_TABLE_MAGIC;
    header->version = OPENNPUX_NPU_ROUTE_TABLE_VERSION;
    header->header_size = sizeof(*header);
    header->record_size = sizeof(*records);
    header->record_count = record_count;
    header->total_size = (uint32_t)size;
    memcpy(header + 1, records, (size_t)record_count * sizeof(*records));
    header->checksum = opennpux_npu_route_table_checksum(buffer, size);
    *table_size = size;
    return 0;
}

int
opennpux_npu_route_table_validate(const void *buffer, size_t size)
{
    if (buffer == NULL || size < sizeof(struct opennpux_npu_route_table_header)) {
        return -1;
    }
    const struct opennpux_npu_route_table_header *header = buffer;
    if (header->magic != OPENNPUX_NPU_ROUTE_TABLE_MAGIC ||
        header->version != OPENNPUX_NPU_ROUTE_TABLE_VERSION ||
        header->header_size != sizeof(*header) ||
        header->record_size != sizeof(struct opennpux_npu_route_record) ||
        header->record_count == 0 ||
        header->record_count > OPENNPUX_NPU_MAX_ACTIVE_EXPERTS ||
        header->total_size != sizeof(*header) +
            (size_t)header->record_count * sizeof(struct opennpux_npu_route_record) ||
        header->total_size > size ||
        header->checksum !=
            opennpux_npu_route_table_checksum(buffer, header->total_size)) {
        return -1;
    }
    const struct opennpux_npu_route_record *records =
        (const struct opennpux_npu_route_record *)(header + 1);
    float weight_sum = 0.0f;
    for (uint32_t index = 0; index < header->record_count; ++index) {
        if (!isfinite(records[index].logit) ||
            !isfinite(records[index].weight) || records[index].weight < 0.0f) {
            return -1;
        }
        for (uint32_t previous = 0; previous < index; ++previous) {
            if (records[previous].expert_id == records[index].expert_id) {
                return -1;
            }
        }
        weight_sum += records[index].weight;
    }
    return isfinite(weight_sum) && fabsf(weight_sum - 1.0f) <= 1e-5f ? 0 : -1;
}
