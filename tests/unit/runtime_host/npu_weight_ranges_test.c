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
    check(ranges.header->range_count == 33, "range count mismatch");
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
    uint32_t q_projection_command = UINT32_MAX;
    for (uint32_t index = 0; index < ranges.header->range_count; ++index) {
        if (ranges.records[index].role_id ==
                OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ &&
            ranges.records[index].component_id ==
                OPENNPUX_NPU_WEIGHT_COMPONENT_QWEIGHT) {
            q_projection_command = ranges.records[index].command_id;
            break;
        }
    }
    check(q_projection_command != UINT32_MAX,
          "Q projection GPTQ range missing");
    const struct opennpux_npu_weight_range_record *qweight;
    check(opennpux_npu_weight_range_find(
              &ranges, q_projection_command,
              OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ,
              OPENNPUX_NPU_WEIGHT_COMPONENT_QWEIGHT,
              OPENNPUX_NPU_WEIGHT_EXPERT_NONE, &qweight) == 0 &&
              qweight->byte_size == 288,
          "Q projection GPTQ range lookup failed");
    struct opennpux_npu_gptq_weight_ranges gptq;
    check(opennpux_npu_weight_ranges_find_gptq(
              &ranges, q_projection_command,
              OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ,
              OPENNPUX_NPU_WEIGHT_EXPERT_NONE,
              OPENNPUX_NPU_WEIGHT_SLOT_Q_PROJ, &gptq) == 0 &&
              gptq.qweight != NULL && gptq.qzeros != NULL &&
              gptq.scales != NULL && gptq.g_idx != NULL,
          "complete GPTQ component set lookup failed");
    check(opennpux_npu_weight_ranges_find_gptq(
              &ranges, q_projection_command,
              OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_PROJ,
              OPENNPUX_NPU_WEIGHT_EXPERT_NONE,
              OPENNPUX_NPU_WEIGHT_SLOT_K_PROJ, &gptq) == 0 &&
              gptq.qweight != NULL && gptq.qzeros != NULL &&
              gptq.scales != NULL && gptq.g_idx != NULL,
          "K projection GPTQ component set lookup failed");
    check(opennpux_npu_weight_ranges_find_gptq(
              &ranges, q_projection_command,
              OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_V_PROJ,
              OPENNPUX_NPU_WEIGHT_EXPERT_NONE,
              OPENNPUX_NPU_WEIGHT_SLOT_V_PROJ, &gptq) == 0 &&
              gptq.qweight != NULL && gptq.qzeros != NULL &&
              gptq.scales != NULL && gptq.g_idx != NULL,
          "V projection GPTQ component set lookup failed");
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
