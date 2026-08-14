#include "opennpux/model_package.h"
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
    check(argc == 3,
          "usage: npu_weight_ranges_test <model.npxm> <model.npxr>");
    struct opennpux_model_package_info model;
    check(opennpux_model_package_load(argv[1], &model) == 0,
          "model package load failed");
    struct opennpux_npu_weight_ranges ranges;
    check(opennpux_npu_weight_ranges_load(argv[2], &ranges) == 0,
          "range index load failed");
    check(ranges.header->range_count == 8, "range count mismatch");
    int found_expert = 0;
    for (uint32_t command = 0; command < ranges.header->command_count; ++command) {
        const struct opennpux_npu_weight_range_record *records;
        uint32_t count;
        check(opennpux_npu_weight_ranges_for_command(
                  &ranges, command, &records, &count) == 0,
              "command range lookup failed");
        for (uint32_t index = 0; index < count; ++index) {
            check(records[index].command_id == command,
                  "lookup returned another command");
            found_expert |= records[index].expert_id == 7;
        }
    }
    check(found_expert, "routed expert metadata missing");
    unsigned char sample[4];
    check(opennpux_model_package_read_shard_range(
              argv[1], &model, ranges.records[0].shard_index,
              ranges.records[0].file_offset, sample, sizeof(sample)) == 0,
          "indexed shard range read failed");
    check(sample[0] == 0 && sample[1] == 1 &&
              sample[2] == 2 && sample[3] == 3,
          "indexed shard payload mismatch");
    opennpux_npu_weight_ranges_unload(&ranges);
    puts("PASS: NPU weight range loader tests");
    return 0;
}
