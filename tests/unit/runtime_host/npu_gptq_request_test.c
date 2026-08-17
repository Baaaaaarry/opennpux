#include "opennpux/npu_gptq_reference.h"
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
    check(file != NULL, "reference image missing");
    check(fseek(file, 0, SEEK_END) == 0, "reference image seek failed");
    const long length = ftell(file);
    check(length > 0 && fseek(file, 0, SEEK_SET) == 0,
          "reference image is empty");
    uint8_t *data = malloc((size_t)length);
    check(data != NULL, "reference image allocation failed");
    check(fread(data, 1, (size_t)length, file) == (size_t)length,
          "reference image read failed");
    fclose(file);
    *size = (size_t)length;
    return data;
}

int
main(int argc, char **argv)
{
    check(argc == 4,
          "usage: npu_gptq_request_test <model.npxm> <model.npxr> "
          "<gptq-projection.bin>");
    struct opennpux_model_package_info model;
    struct opennpux_npu_weight_ranges ranges;
    check(opennpux_model_package_load(argv[1], &model) == 0,
          "model package load failed");
    check(opennpux_npu_weight_ranges_load(argv[2], &ranges) == 0,
          "weight range index load failed");

    uint32_t command_id = UINT32_MAX;
    for (uint32_t index = 0; index < ranges.header->range_count; ++index) {
        if (ranges.records[index].role_id ==
                OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ) {
            command_id = ranges.records[index].command_id;
            break;
        }
    }
    check(command_id != UINT32_MAX, "Q projection command missing");

    const struct opennpux_npu_gptq_projection_selector selector = {
        command_id, OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ,
        OPENNPUX_NPU_WEIGHT_EXPERT_NONE, OPENNPUX_NPU_WEIGHT_SLOT_Q_PROJ};
    const struct opennpux_npu_gptq_projection_shape shape = {
        1, model.hidden_size, model.head_count * model.head_dim,
        model.quantization_group_size, 1};
    struct opennpux_npu_gptq_request_layout layout;
    check(opennpux_npu_gptq_request_layout(&shape, 2, 1, &layout) == 0,
          "projection layout failed");
    check(layout.input_offset == sizeof(struct coral_gptq_matmul_request),
          "projection layout does not start after the request record");
    check(layout.total_size % OPENNPUX_NPU_GPTQ_REQUEST_ALIGNMENT == 0,
          "projection layout is not aligned");

    /*
     * The offline materializer writes the same deterministic input, so the
     * staged image must match it byte for byte.
     */
    const size_t input_floats = (size_t)shape.rows * shape.input_columns;
    float *input = malloc(input_floats * sizeof(*input));
    check(input != NULL, "projection input allocation failed");
    for (size_t index = 0; index < input_floats; ++index) {
        input[index] = (float)(((int)(index % 17) - 8) / 16.0);
    }

    size_t expected_size = 0;
    uint8_t *expected = read_file(argv[3], &expected_size);
    /* Oversized so the shape checks below are not masked by ENOSPC. */
    const size_t image_size = expected_size + 4096;
    uint8_t *image = malloc(image_size);
    check(image != NULL, "projection image allocation failed");
    check(opennpux_npu_gptq_request_stage(
              argv[1], &model, &ranges, &selector, &shape, input, input_floats,
              OPENNPUX_NPU_GPTQ_REFERENCE_EXTMEM_BASE, image, expected_size,
              &layout) == 0,
          "projection staging failed");
    check(layout.total_size == expected_size,
          "staged projection size differs from the materialized image");
    check(layout.scale_data_type == CORAL_GPTQ_SCALE_FLOAT16,
          "staged projection scale dtype mismatch");
    check(memcmp(image, expected, expected_size) == 0,
          "staged projection differs from the materialized image");

    struct opennpux_npu_gptq_reference_result result;
    check(opennpux_npu_gptq_reference_run(
              image, layout.total_size,
              OPENNPUX_NPU_GPTQ_REFERENCE_EXTMEM_BASE, NULL, 0, &result) == 0,
          "staged projection is not executable");
    printf("gptq_request_staged_checksum=0x%08x\n", result.output_checksum);

    check(opennpux_npu_gptq_request_stage(
              argv[1], &model, &ranges, &selector, &shape, input, input_floats,
              OPENNPUX_NPU_GPTQ_REFERENCE_EXTMEM_BASE, image,
              layout.total_size - 1, &layout) != 0 && errno == ENOSPC,
          "undersized projection image was accepted");
    check(opennpux_npu_gptq_request_stage(
              argv[1], &model, &ranges, &selector, &shape, input,
              input_floats - 1, OPENNPUX_NPU_GPTQ_REFERENCE_EXTMEM_BASE, image,
              image_size, &layout) != 0 && errno == EINVAL,
          "mismatched input length was accepted");
    struct opennpux_npu_gptq_projection_shape wide = shape;
    wide.output_columns += 8;
    check(opennpux_npu_gptq_request_stage(
              argv[1], &model, &ranges, &selector, &wide, input, input_floats,
              OPENNPUX_NPU_GPTQ_REFERENCE_EXTMEM_BASE, image, image_size,
              &layout) != 0 && errno == EPROTO,
          "projection shape was not checked against the model components");

    free(image);
    free(expected);
    free(input);
    opennpux_npu_weight_ranges_unload(&ranges);
    puts("PASS: NPU GPTQ projection staging tests");
    return 0;
}
