#include "opennpux/npu_weight_ranges.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t
range_checksum(const void *buffer, size_t size)
{
    const uint8_t *bytes = buffer;
    const size_t begin =
        offsetof(struct opennpux_npu_weight_range_header, checksum);
    uint32_t value = UINT32_C(2166136261);
    for (size_t index = 0; index < size; ++index) {
        const uint8_t byte = index >= begin && index < begin + sizeof(uint32_t) ?
            0 : bytes[index];
        value ^= byte;
        value *= UINT32_C(16777619);
    }
    return value;
}

int
opennpux_npu_weight_ranges_load(
    const char *path, struct opennpux_npu_weight_ranges *ranges)
{
    if (path == NULL || ranges == NULL) {
        errno = EINVAL;
        return -1;
    }
    memset(ranges, 0, sizeof(*ranges));
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    const long length = ftell(file);
    if (length < (long)sizeof(struct opennpux_npu_weight_range_header) ||
        (unsigned long)length > SIZE_MAX || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        errno = EINVAL;
        return -1;
    }
    void *storage = malloc((size_t)length);
    if (storage == NULL) {
        fclose(file);
        return -1;
    }
    if (fread(storage, 1, (size_t)length, file) != (size_t)length) {
        free(storage);
        fclose(file);
        errno = EIO;
        return -1;
    }
    fclose(file);

    const struct opennpux_npu_weight_range_header *header = storage;
    const uint64_t records_size =
        (uint64_t)header->range_count * sizeof(struct opennpux_npu_weight_range_record);
    if (header->magic == OPENNPUX_NPU_WEIGHT_RANGE_MAGIC &&
        header->version != OPENNPUX_NPU_WEIGHT_RANGE_VERSION) {
        free(storage);
        errno = EPROTONOSUPPORT;
        return -1;
    }
    if (header->magic != OPENNPUX_NPU_WEIGHT_RANGE_MAGIC ||
        header->version != OPENNPUX_NPU_WEIGHT_RANGE_VERSION ||
        header->header_size != sizeof(*header) ||
        header->record_size != sizeof(struct opennpux_npu_weight_range_record) ||
        header->command_count == 0 || header->range_count == 0 ||
        header->shard_count == 0 || header->executable_id == 0 ||
        header->total_size != (uint64_t)length ||
        header->record_offset > header->total_size ||
        records_size > header->total_size - header->record_offset ||
        header->record_offset + records_size != header->total_size ||
        header->checksum != range_checksum(storage, (size_t)length)) {
        free(storage);
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_weight_range_record *records =
        (const struct opennpux_npu_weight_range_record *)(
            (const uint8_t *)storage + header->record_offset);
    uint32_t previous_command = 0;
    for (uint32_t index = 0; index < header->range_count; ++index) {
        if (records[index].command_id >= header->command_count ||
            records[index].shard_index >= header->shard_count ||
            records[index].byte_size == 0 ||
            records[index].file_offset > UINT64_MAX - records[index].byte_size ||
            records[index].tensor_id == 0 ||
            records[index].parameter_symbol == 0 ||
            (index != 0 && records[index].command_id < previous_command)) {
            free(storage);
            errno = EINVAL;
            return -1;
        }
        previous_command = records[index].command_id;
    }
    ranges->storage = storage;
    ranges->storage_size = (size_t)length;
    ranges->header = header;
    ranges->records = records;
    return 0;
}

void
opennpux_npu_weight_ranges_unload(struct opennpux_npu_weight_ranges *ranges)
{
    if (ranges != NULL) {
        free(ranges->storage);
        memset(ranges, 0, sizeof(*ranges));
    }
}

int
opennpux_npu_weight_ranges_for_command(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    const struct opennpux_npu_weight_range_record **records,
    uint32_t *record_count)
{
    if (ranges == NULL || ranges->header == NULL || records == NULL ||
        record_count == NULL || command_id >= ranges->header->command_count) {
        errno = EINVAL;
        return -1;
    }
    uint32_t begin = 0;
    uint32_t end = ranges->header->range_count;
    while (begin < end) {
        const uint32_t middle = begin + (end - begin) / 2;
        if (ranges->records[middle].command_id < command_id) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    const uint32_t first = begin;
    while (begin < ranges->header->range_count &&
           ranges->records[begin].command_id == command_id) {
        ++begin;
    }
    *records = ranges->records + first;
    *record_count = begin - first;
    return 0;
}

int
opennpux_npu_weight_range_find(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    uint32_t role_id, uint32_t component_id, uint64_t expert_id,
    const struct opennpux_npu_weight_range_record **record)
{
    if (record == NULL || role_id == 0 || component_id == 0) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_weight_range_record *records;
    uint32_t record_count;
    if (opennpux_npu_weight_ranges_for_command(
            ranges, command_id, &records, &record_count) != 0) {
        return -1;
    }
    const struct opennpux_npu_weight_range_record *match = NULL;
    for (uint32_t index = 0; index < record_count; ++index) {
        if (records[index].role_id != role_id ||
            records[index].component_id != component_id ||
            records[index].expert_id != expert_id) {
            continue;
        }
        if (match != NULL) {
            errno = EEXIST;
            return -1;
        }
        match = &records[index];
    }
    if (match == NULL) {
        errno = ENOENT;
        return -1;
    }
    *record = match;
    return 0;
}

int
opennpux_npu_weight_ranges_find_gptq(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    uint32_t role_id, uint64_t expert_id, uint32_t slot_id,
    struct opennpux_npu_gptq_weight_ranges *gptq)
{
    if (gptq == NULL) {
        errno = EINVAL;
        return -1;
    }
    memset(gptq, 0, sizeof(*gptq));
    if (opennpux_npu_weight_range_find_slot(
            ranges, command_id, role_id,
            OPENNPUX_NPU_WEIGHT_COMPONENT_QWEIGHT, expert_id,
            slot_id,
            &gptq->qweight) != 0 ||
        opennpux_npu_weight_range_find_slot(
            ranges, command_id, role_id,
            OPENNPUX_NPU_WEIGHT_COMPONENT_QZEROS, expert_id,
            slot_id,
            &gptq->qzeros) != 0 ||
        opennpux_npu_weight_range_find_slot(
            ranges, command_id, role_id,
            OPENNPUX_NPU_WEIGHT_COMPONENT_SCALES, expert_id,
            slot_id,
            &gptq->scales) != 0) {
        memset(gptq, 0, sizeof(*gptq));
        return -1;
    }
    if (opennpux_npu_weight_range_find_slot(
            ranges, command_id, role_id,
            OPENNPUX_NPU_WEIGHT_COMPONENT_G_IDX, expert_id,
            slot_id,
            &gptq->g_idx) != 0) {
        if (errno != ENOENT) {
            memset(gptq, 0, sizeof(*gptq));
            return -1;
        }
        gptq->g_idx = NULL;
    }
    return 0;
}

int
opennpux_npu_weight_range_find_slot(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    uint32_t role_id, uint32_t component_id, uint64_t expert_id,
    uint32_t slot_id,
    const struct opennpux_npu_weight_range_record **record)
{
    const struct opennpux_npu_weight_range_record *records;
    uint32_t record_count;
    if (record == NULL || role_id == 0 || component_id == 0 ||
        slot_id > OPENNPUX_NPU_WEIGHT_SLOT_MASK) {
        errno = EINVAL;
        return -1;
    }
    if (opennpux_npu_weight_ranges_for_command(
            ranges, command_id, &records, &record_count) != 0) {
        return -1;
    }
    const struct opennpux_npu_weight_range_record *match = NULL;
    for (uint32_t index = 0; index < record_count; ++index) {
        if (records[index].role_id != role_id ||
            records[index].component_id != component_id ||
            records[index].expert_id != expert_id ||
            (records[index].flags & OPENNPUX_NPU_WEIGHT_SLOT_MASK) != slot_id) {
            continue;
        }
        if (match != NULL) {
            errno = EEXIST;
            return -1;
        }
        match = &records[index];
    }
    if (match == NULL) {
        errno = ENOENT;
        return -1;
    }
    *record = match;
    return 0;
}
