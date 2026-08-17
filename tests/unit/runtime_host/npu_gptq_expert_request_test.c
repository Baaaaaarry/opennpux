#include "opennpux/npu_gptq_request.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s (errno=%d)\n", message, errno);
        exit(1);
    }
}

static uint8_t *
read_file(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    check(file != NULL, "expert image missing");
    check(fseek(file, 0, SEEK_END) == 0, "expert image seek failed");
    const long length = ftell(file);
    check(length > 0 && fseek(file, 0, SEEK_SET) == 0,
          "expert image is empty");
    uint8_t *data = malloc((size_t)length);
    check(data != NULL, "expert image allocation failed");
    check(fread(data, 1, (size_t)length, file) == (size_t)length,
          "expert image read failed");
    fclose(file);
    *size = (size_t)length;
    return data;
}

int
main(int argc, char **argv)
{
    check(argc == 4,
          "usage: npu_gptq_expert_request_test <model.npxm> <model.npxr> "
          "<gptq-expert.bin>");
    struct opennpux_model_package_info model;
    struct opennpux_npu_weight_ranges ranges;
    check(opennpux_model_package_load(argv[1], &model) == 0,
          "model package load failed");
    check(opennpux_npu_weight_ranges_load(argv[2], &ranges) == 0,
          "weight range index load failed");

    uint32_t command_id = UINT32_MAX;
    for (uint32_t index = 0; index < ranges.header->range_count; ++index) {
        const struct opennpux_npu_weight_range_record *record =
            &ranges.records[index];
        if (record->role_id == OPENNPUX_NPU_WEIGHT_ROLE_ROUTED_EXPERT &&
            record->expert_id == 0 &&
            (record->flags & OPENNPUX_NPU_WEIGHT_SLOT_MASK) ==
                OPENNPUX_NPU_WEIGHT_SLOT_GATE_PROJ) {
            command_id = record->command_id;
            break;
        }
    }
    check(command_id != UINT32_MAX, "complete routed expert missing");

    const struct opennpux_npu_gptq_expert_selector selector = {
        command_id, OPENNPUX_NPU_WEIGHT_ROLE_ROUTED_EXPERT, 0};
    const struct opennpux_npu_gptq_expert_shape shape = {
        1, model.hidden_size, model.moe_intermediate_size,
        model.quantization_group_size, 1};
    const size_t input_floats = shape.hidden_columns;
    float *input = malloc(input_floats * sizeof(*input));
    check(input != NULL, "expert input allocation failed");
    for (size_t index = 0; index < input_floats; ++index) {
        input[index] = (float)(((int)(index % 17) - 8) / 16.0);
    }

    size_t expected_size = 0;
    uint8_t *expected = read_file(argv[3], &expected_size);
    uint8_t *image = malloc(expected_size);
    check(image != NULL, "expert staging allocation failed");
    struct opennpux_npu_gptq_expert_layout layout;
    check(opennpux_npu_gptq_expert_stage(
              argv[1], &model, &ranges, &selector, &shape, input,
              input_floats, UINT32_C(0x20000000), image, expected_size,
              &layout) == 0,
          "expert staging failed");
    check(layout.total_size == expected_size,
          "runtime and offline expert sizes differ");
    check(layout.output_bytes == model.hidden_size * sizeof(float),
          "expert output layout is invalid");
    check(memcmp(image, expected, expected_size) == 0,
          "runtime expert differs from offline materializer");
    check(opennpux_npu_gptq_expert_stage(
              argv[1], &model, &ranges, &selector, &shape, input,
              input_floats, UINT32_C(0x20000000), image, expected_size - 1,
              &layout) != 0 && errno == ENOSPC,
          "undersized expert image was accepted");

    free(image);
    free(expected);
    free(input);
    opennpux_npu_weight_ranges_unload(&ranges);
    puts("PASS: NPU GPTQ expert staging tests");
    return 0;
}
