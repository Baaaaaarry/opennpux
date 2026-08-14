#include "opennpux/model_package.h"
#include "opennpux/npu_weight_pager.h"
#include "opennpux/npu_weight_ranges.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define CACHE_SLOTS UINT32_C(64)

static int
parse_transfer_size(const char *text, uint32_t *size)
{
    char *end = NULL;
    errno = 0;
    const unsigned long value = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' ||
        value < OPENNPUX_NPU_WEIGHT_PAGE_SIZE ||
        value > OPENNPUX_NPU_WEIGHT_TRANSFER_MAX ||
        (value & (value - 1)) != 0) {
        errno = EINVAL;
        return -1;
    }
    *size = (uint32_t)value;
    return 0;
}

static uint32_t
fnv_word(uint32_t checksum, const void *page)
{
    const unsigned char *bytes = page;
    for (uint32_t index = 0; index < sizeof(uint32_t); ++index) {
        checksum ^= bytes[index];
        checksum *= UINT32_C(16777619);
    }
    return checksum;
}

int
main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        fprintf(stderr,
                "usage: %s <model.npxm> <model.npxr> [transfer-bytes]\n",
                argv[0]);
        return 2;
    }
    uint32_t transfer_size = OPENNPUX_NPU_WEIGHT_PAGE_SIZE;
    if (argc == 4 && parse_transfer_size(argv[3], &transfer_size) != 0) {
        fprintf(stderr, "npu-weight-inspect: invalid transfer size '%s'\n",
                argv[3]);
        return 2;
    }
    struct opennpux_model_package_info model;
    struct opennpux_npu_weight_ranges ranges;
    if (opennpux_model_package_load(argv[1], &model) != 0 ||
        opennpux_model_package_validate_shards(argv[1], &model) != 0 ||
        opennpux_npu_weight_ranges_load(argv[2], &ranges) != 0) {
        perror("npu-weight-inspect load");
        return 1;
    }
    if (ranges.header->shard_count != model.shard_count) {
        fprintf(stderr, "npu-weight-inspect: model/range shard mismatch\n");
        opennpux_npu_weight_ranges_unload(&ranges);
        return 1;
    }
    uint64_t *active_experts = calloc(
        model.experts_per_token == 0 ? 1 : model.experts_per_token,
        sizeof(*active_experts));
    struct opennpux_npu_weight_cache_entry *entries = calloc(
        CACHE_SLOTS, sizeof(*entries));
    void *storage = malloc(
        (size_t)CACHE_SLOTS * transfer_size);
    if (active_experts == NULL || entries == NULL || storage == NULL) {
        perror("npu-weight-inspect allocate");
        free(active_experts);
        free(entries);
        free(storage);
        opennpux_npu_weight_ranges_unload(&ranges);
        return 1;
    }
    for (uint32_t index = 0; index < model.experts_per_token; ++index) {
        active_experts[index] = index;
    }
    struct opennpux_npu_weight_cache cache;
    if (opennpux_npu_weight_cache_init_sized(
            &cache, entries, storage, CACHE_SLOTS, transfer_size) != 0) {
        perror("npu-weight-inspect cache");
        return 1;
    }

    uint64_t page_requests = 0;
    uint32_t mapped_commands = 0;
    uint32_t sampled_commands = 0;
    uint32_t sample_checksum = UINT32_C(2166136261);
    for (uint32_t command = 0; command < ranges.header->command_count; ++command) {
        struct opennpux_npu_weight_page_cursor cursor;
        if (opennpux_npu_weight_page_cursor_begin_sized(
                &ranges, command, active_experts, model.experts_per_token,
                transfer_size, &cursor) != 0) {
            perror("npu-weight-inspect cursor");
            return 1;
        }
        struct opennpux_npu_weight_page_request request;
        int next = opennpux_npu_weight_page_cursor_next(&cursor, &request);
        if (next < 0) {
            perror("npu-weight-inspect request");
            return 1;
        }
        if (next != 0) {
            ++mapped_commands;
            const void *page;
            uint32_t slot;
            uint32_t hit;
            if (opennpux_npu_weight_cache_acquire(
                    &cache, argv[1], &model, &request, &page, &slot,
                    &hit) != 0) {
                perror("npu-weight-inspect sample");
                return 1;
            }
            (void)slot;
            (void)hit;
            sample_checksum = fnv_word(sample_checksum, page);
            ++sampled_commands;
            ++page_requests;
        }
        while ((next = opennpux_npu_weight_page_cursor_next(
                    &cursor, &request)) > 0) {
            ++page_requests;
        }
        if (next < 0) {
            perror("npu-weight-inspect request");
            return 1;
        }
    }
    printf("weight_inspect_commands=%" PRIu32 "\n", ranges.header->command_count);
    printf("weight_inspect_range_records=%" PRIu32 "\n", ranges.header->range_count);
    printf("weight_inspect_active_experts=%" PRIu32 "\n", model.experts_per_token);
    printf("weight_inspect_transfer_bytes=%" PRIu32 "\n", transfer_size);
    printf("weight_inspect_mapped_commands=%" PRIu32 "\n", mapped_commands);
    printf("weight_inspect_page_requests=%" PRIu64 "\n", page_requests);
    printf("weight_inspect_request_bytes=%" PRIu64 "\n",
           page_requests * transfer_size);
    printf("weight_inspect_sampled_commands=%" PRIu32 "\n", sampled_commands);
    printf("weight_inspect_sample_checksum=0x%08" PRIx32 "\n", sample_checksum);
    printf("weight_inspect_cache_hits=%" PRIu64 "\n", cache.stats.hits);
    printf("weight_inspect_cache_misses=%" PRIu64 "\n", cache.stats.misses);
    printf("weight_inspect_cache_evictions=%" PRIu64 "\n", cache.stats.evictions);
    printf("weight_inspect_bytes_read=%" PRIu64 "\n", cache.stats.bytes_read);
    puts("weight_inspect=PASS");

    free(active_experts);
    free(entries);
    free(storage);
    opennpux_npu_weight_ranges_unload(&ranges);
    return 0;
}
