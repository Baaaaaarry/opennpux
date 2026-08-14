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
    check(argc == 2, "usage: npu_weight_ranges_test <model.npxr>");
    struct opennpux_npu_weight_ranges ranges;
    check(opennpux_npu_weight_ranges_load(argv[1], &ranges) == 0,
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
    opennpux_npu_weight_ranges_unload(&ranges);
    puts("PASS: NPU weight range loader tests");
    return 0;
}
