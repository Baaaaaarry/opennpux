#include "opennpux/model_package.h"
#include "opennpux/npu_gptq_weights.h"
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
          "usage: npu_gptq_weights_test <model.npxm> <model.npxr>");
    struct opennpux_model_package_info model;
    struct opennpux_npu_weight_ranges ranges;
    check(opennpux_model_package_load(argv[1], &model) == 0,
          "model package load failed");
    check(opennpux_npu_weight_ranges_load(argv[2], &ranges) == 0,
          "weight ranges load failed");
    uint32_t command_id = UINT32_MAX;
    for (uint32_t index = 0; index < ranges.header->range_count; ++index) {
        if (ranges.records[index].role_id ==
                OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ) {
            command_id = ranges.records[index].command_id;
            break;
        }
    }
    check(command_id != UINT32_MAX, "Q projection command missing");

    struct opennpux_npu_gptq_weights weights;
    check(opennpux_npu_gptq_weights_load(
              argv[1], &model, &ranges, command_id,
              OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ,
              OPENNPUX_NPU_WEIGHT_EXPERT_NONE,
              OPENNPUX_NPU_WEIGHT_SLOT_Q_PROJ, 4096, &weights) == 0,
          "GPTQ components load failed");
    check(weights.qweight.size == 8 && weights.qzeros.size == 4 &&
              weights.scales.size == 8 && weights.g_idx.size == 8,
          "GPTQ component sizes mismatch");
    check(((const unsigned char *)weights.qweight.data)[1] == 1 &&
              ((const unsigned char *)weights.qzeros.data)[3] == 3,
          "GPTQ component payload mismatch");
    opennpux_npu_gptq_weights_unload(&weights);
    check(weights.qweight.data == NULL && weights.qzeros.data == NULL,
          "GPTQ component unload failed");

    check(opennpux_npu_gptq_weights_load(
              argv[1], &model, &ranges, command_id,
              OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ,
              OPENNPUX_NPU_WEIGHT_EXPERT_NONE,
              OPENNPUX_NPU_WEIGHT_SLOT_Q_PROJ, 4, &weights) != 0 &&
              errno == EFBIG,
          "GPTQ component size limit was ignored");
    opennpux_npu_weight_ranges_unload(&ranges);
    puts("PASS: NPU GPTQ component loader tests");
    return 0;
}
