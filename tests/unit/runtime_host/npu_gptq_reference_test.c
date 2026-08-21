#include "opennpux/npu_gptq_reference.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SYNTHETIC_BASE UINT64_C(0x20000000)
#define SYNTHETIC_SIZE 448u
#define REQUEST_OFFSET 0u
#define INPUT_OFFSET 128u
#define QWEIGHT_OFFSET 192u
#define QZEROS_OFFSET 256u
#define SCALES_OFFSET 320u
#define OUTPUT_OFFSET 384u

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s (errno=%d)\n", message, errno);
        exit(1);
    }
}

/*
 * Deterministic 8x8 projection: quantized nibble k, stored zero 0 plus a zero
 * bias of 1, unit scales and unit inputs. Every output column therefore
 * accumulates sum(k - 1) for k in [0, 8) == 20.
 */
static void
build_synthetic(uint8_t *image)
{
    memset(image, 0, SYNTHETIC_SIZE);
    struct coral_gptq_matmul_request request;
    memset(&request, 0, sizeof(request));
    request.magic = CORAL_GPTQ_MATMUL_MAGIC;
    request.version = CORAL_GPTQ_MATMUL_VERSION;
    request.struct_size = sizeof(request);
    request.state = CORAL_GPTQ_MATMUL_PENDING;
    request.rows = 1;
    request.input_columns = 8;
    request.output_columns = 8;
    request.group_size = 8;
    request.zero_bias = 1;
    request.scale_data_type = CORAL_GPTQ_SCALE_FLOAT32;
    request.input_address = (uint32_t)(SYNTHETIC_BASE + INPUT_OFFSET);
    request.qweight_address = (uint32_t)(SYNTHETIC_BASE + QWEIGHT_OFFSET);
    request.qzeros_address = (uint32_t)(SYNTHETIC_BASE + QZEROS_OFFSET);
    request.scales_address = (uint32_t)(SYNTHETIC_BASE + SCALES_OFFSET);
    request.output_address = (uint32_t)(SYNTHETIC_BASE + OUTPUT_OFFSET);
    memcpy(image + REQUEST_OFFSET, &request, sizeof(request));

    for (unsigned index = 0; index < 8; ++index) {
        const float one = 1.0f;
        const uint32_t packed = UINT32_C(0x76543210);
        memcpy(image + INPUT_OFFSET + index * sizeof(float), &one,
               sizeof(one));
        memcpy(image + QWEIGHT_OFFSET + index * sizeof(packed), &packed,
               sizeof(packed));
        memcpy(image + SCALES_OFFSET + index * sizeof(float), &one,
               sizeof(one));
    }
}

static void
run_synthetic(void)
{
    _Alignas(64) uint8_t image[SYNTHETIC_SIZE];
    build_synthetic(image);
    float output[8];
    struct opennpux_npu_gptq_reference_result result;
    check(opennpux_npu_gptq_reference_run(
              image, sizeof(image), SYNTHETIC_BASE, output,
              sizeof(output) / sizeof(output[0]), &result) == 0,
          "synthetic projection reference failed");
    for (unsigned index = 0; index < 8; ++index) {
        check(output[index] == 20.0f, "synthetic projection value mismatch");
    }
    check(result.operations == 128 && result.bytes_read == 100 &&
              result.bytes_written == 32 && result.modeled_cycles == 73,
          "synthetic projection statistics mismatch");
    check(result.has_g_idx == 0, "synthetic projection reported g_idx");
    check(result.output_checksum ==
              opennpux_npu_gptq_reference_checksum(output, 8),
          "streaming checksum does not match the output buffer");
    /* Shared with TestHostReferenceVector in gem5_gptq_kernels_test.cc. */
    check(result.output_checksum == UINT32_C(0x5bea2f85),
          "synthetic projection checksum drifted from the bridge kernel");

    struct coral_gptq_matmul_request request;
    memcpy(&request, image, sizeof(request));
    const uint32_t scales_address = request.scales_address;
    request.scales_address = 0;
    memcpy(image, &request, sizeof(request));
    check(opennpux_npu_gptq_reference_run(
              image, sizeof(image), SYNTHETIC_BASE, NULL, 0, &result) != 0,
          "missing scales operand was accepted");
    request.scales_address = scales_address;
    request.magic = 0;
    memcpy(image, &request, sizeof(request));
    check(opennpux_npu_gptq_reference_run(
              image, sizeof(image), SYNTHETIC_BASE, NULL, 0, &result) != 0 &&
              errno == EPROTO,
          "invalid request magic was accepted");
    request.magic = CORAL_GPTQ_MATMUL_MAGIC;
    memcpy(image, &request, sizeof(request));
    check(opennpux_npu_gptq_reference_run(
              image, OUTPUT_OFFSET + 4, SYNTHETIC_BASE, NULL, 0,
              &result) != 0 && errno == ERANGE,
          "truncated output region was accepted");
    check(opennpux_npu_gptq_reference_run(
              image, sizeof(image), SYNTHETIC_BASE, output, 4, &result) != 0 &&
              errno == ENOSPC,
          "undersized caller output buffer was accepted");
}

static void
run_image(const char *path)
{
    FILE *file = fopen(path, "rb");
    check(file != NULL, "staged projection image missing");
    check(fseek(file, 0, SEEK_END) == 0, "staged projection image seek failed");
    const long size = ftell(file);
    check(size > 0 && fseek(file, 0, SEEK_SET) == 0,
          "staged projection image is empty");
    uint8_t *image = malloc((size_t)size);
    check(image != NULL, "staged projection image allocation failed");
    check(fread(image, 1, (size_t)size, file) == (size_t)size,
          "staged projection image read failed");
    fclose(file);

    struct opennpux_npu_gptq_reference_result result;
    check(opennpux_npu_gptq_reference_run(
              image, (size_t)size, OPENNPUX_NPU_GPTQ_REFERENCE_EXTMEM_BASE,
              NULL, 0, &result) == 0,
          "staged projection reference failed");
    check(result.rows == 1 && result.input_columns == 18 &&
              result.output_columns == 48,
          "staged projection shape mismatch");
    check(result.scale_data_type == CORAL_GPTQ_SCALE_FLOAT16,
          "staged projection scale dtype mismatch");
    check(result.has_g_idx == 1, "staged projection is missing g_idx");
    check(result.operations == 2u * 18u * 48u,
          "staged projection operation count mismatch");
    check(result.output_checksum != 0,
          "staged projection checksum is unset");
    printf("gptq_reference_staged_checksum=0x%08x\n", result.output_checksum);
    free(image);
}

int
main(int argc, char **argv)
{
    check(argc == 2, "usage: npu_gptq_reference_test <gptq-projection.bin>");
    run_synthetic();
    run_image(argv[1]);
    puts("PASS: NPU GPTQ projection reference tests");
    return 0;
}
