#include "opennpux/model_package.h"
#include "opennpux/npu_page_bundle.h"
#include "opennpux/npu_weight_pager.h"
#include "opennpux/npu_weight_ranges.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
parse_u32(const char *text, uint32_t *value)
{
    char *end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0' || parsed > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static int
write_exact(FILE *file, const void *data, size_t size)
{
    return fwrite(data, 1, size, file) == size ? 0 : -1;
}

int
main(int argc, char **argv)
{
    if (argc < 4 || argc > 6) {
        fprintf(stderr, "usage: %s <model.npxm> <model.npxr> <output.npxb> "
                "[transfer-bytes] [max-pages-per-range]\n", argv[0]);
        return 2;
    }
    uint32_t transfer_size = UINT32_C(65536);
    uint32_t max_pages = 1;
    if ((argc >= 5 && parse_u32(argv[4], &transfer_size) != 0) ||
        (argc == 6 && parse_u32(argv[5], &max_pages) != 0) ||
        transfer_size < OPENNPUX_NPU_WEIGHT_PAGE_SIZE ||
        transfer_size > OPENNPUX_NPU_WEIGHT_TRANSFER_MAX ||
        (transfer_size & (transfer_size - 1)) != 0) {
        fprintf(stderr, "npu-page-bundle: invalid paging arguments\n");
        return 2;
    }
    struct opennpux_model_package_info model;
    struct opennpux_npu_weight_ranges ranges;
    if (opennpux_model_package_load(argv[1], &model) != 0 ||
        opennpux_model_package_validate_shards(argv[1], &model) != 0 ||
        opennpux_npu_weight_ranges_load(argv[2], &ranges) != 0) {
        perror("npu-page-bundle load");
        return 1;
    }
    const uint32_t active_count = model.experts_per_token < 8 ?
        model.experts_per_token : 8;
    uint64_t active_experts[8] = {0};
    for (uint32_t index = 0; index < active_count; ++index) {
        active_experts[index] = index;
    }
    uint8_t *page = malloc(transfer_size);
    FILE *output = fopen(argv[3], "wb+");
    if (page == NULL || output == NULL) {
        perror("npu-page-bundle output");
        free(page);
        opennpux_npu_weight_ranges_unload(&ranges);
        return 1;
    }
    struct opennpux_npu_page_bundle_header header = {
        .magic = OPENNPUX_NPU_PAGE_BUNDLE_MAGIC,
        .version = OPENNPUX_NPU_PAGE_BUNDLE_VERSION,
        .header_size = sizeof(header),
        .record_size = sizeof(struct opennpux_npu_page_bundle_record),
        .transfer_size = transfer_size,
        .command_count = ranges.header->command_count,
        .active_expert_count = active_count,
        .max_pages_per_record = max_pages,
    };
    if (write_exact(output, &header, sizeof(header)) != 0) {
        perror("npu-page-bundle header");
        return 1;
    }
    for (uint32_t command = 0; command < ranges.header->command_count;
         ++command) {
        struct opennpux_npu_weight_page_cursor cursor;
        if (opennpux_npu_weight_page_cursor_begin_sized(
                &ranges, command, active_experts, active_count,
                transfer_size, &cursor) != 0 ||
            opennpux_npu_weight_page_cursor_limit_records(
                &cursor, max_pages) != 0) {
            perror("npu-page-bundle cursor");
            return 1;
        }
        struct opennpux_npu_weight_page_request request;
        int next;
        while ((next = opennpux_npu_weight_page_cursor_next(
                    &cursor, &request)) > 0) {
            struct opennpux_npu_weight_page_cursor probe = cursor;
            struct opennpux_npu_weight_page_request ignored;
            const int has_next = opennpux_npu_weight_page_cursor_next(
                &probe, &ignored);
            if (has_next < 0 || opennpux_npu_weight_page_read(
                    argv[1], &model, &request, page, transfer_size) != 0) {
                perror("npu-page-bundle page");
                return 1;
            }
            const struct opennpux_npu_page_bundle_record record = {
                .command_id = request.command_id,
                .shard_index = request.shard_index,
                .role_id = request.role_id,
                .component_id = request.component_id,
                .expert_id = request.expert_id,
                .file_offset = request.file_offset,
                .range_file_offset = request.range_file_offset,
                .range_size = request.range_size,
                .page_size = transfer_size,
                .flags = has_next == 0 ? OPENNPUX_NPU_PAGE_BUNDLE_LAST : 0,
            };
            if (write_exact(output, &record, sizeof(record)) != 0 ||
                write_exact(output, page, transfer_size) != 0) {
                perror("npu-page-bundle write");
                return 1;
            }
            ++header.record_count;
            header.payload_bytes += transfer_size;
        }
        if (next < 0) {
            perror("npu-page-bundle request");
            return 1;
        }
    }
    if (fseek(output, 0, SEEK_SET) != 0 ||
        write_exact(output, &header, sizeof(header)) != 0 ||
        fclose(output) != 0) {
        perror("npu-page-bundle finalize");
        return 1;
    }
    printf("sim_host_bundle_records=%" PRIu32 "\n", header.record_count);
    printf("sim_host_bundle_transfer_size=%" PRIu32 "\n", transfer_size);
    printf("sim_host_bundle_payload_bytes=%" PRIu64 "\n", header.payload_bytes);
    puts("sim_host_bundle=PASS");
    free(page);
    opennpux_npu_weight_ranges_unload(&ranges);
    return 0;
}
