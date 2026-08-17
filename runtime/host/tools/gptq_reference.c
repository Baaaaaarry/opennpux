#include "opennpux/npu_gptq_reference.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

static void *
read_image(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long file_size = ftell(file);
    if (file_size <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        errno = EINVAL;
        return NULL;
    }
    void *payload = malloc((size_t)file_size);
    if (payload == NULL) {
        fclose(file);
        return NULL;
    }
    const size_t read = fread(payload, 1, (size_t)file_size, file);
    const int failed = read != (size_t)file_size || ferror(file);
    fclose(file);
    if (failed) {
        free(payload);
        errno = EIO;
        return NULL;
    }
    *size = (size_t)file_size;
    return payload;
}

int
main(int argc, char **argv)
{
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s <gptq-projection.bin> [device-base]\n",
                argv[0]);
        return 2;
    }
    uint64_t device_base = OPENNPUX_NPU_GPTQ_REFERENCE_EXTMEM_BASE;
    if (argc == 3) {
        char *end = NULL;
        errno = 0;
        const unsigned long long parsed = strtoull(argv[2], &end, 0);
        if (errno != 0 || end == argv[2] || *end != '\0' ||
            parsed > UINT32_MAX) {
            fprintf(stderr, "invalid device base: %s\n", argv[2]);
            return 2;
        }
        device_base = (uint64_t)parsed;
    }

    size_t image_size = 0;
    void *image = read_image(argv[1], &image_size);
    if (image == NULL) {
        perror("gptq reference image");
        return 1;
    }
    struct opennpux_npu_gptq_reference_result result;
    const int rc = opennpux_npu_gptq_reference_run(
        image, image_size, device_base, NULL, 0, &result);
    free(image);
    if (rc != 0) {
        perror("gptq reference");
        return 1;
    }
    printf("gptq_reference_shape=%" PRIu32 "x%" PRIu32 "x%" PRIu32 "\n",
           result.rows, result.input_columns, result.output_columns);
    printf("gptq_reference_group_size=%" PRIu32 "\n", result.group_size);
    printf("gptq_reference_zero_bias=%" PRIu32 "\n", result.zero_bias);
    printf("gptq_reference_scale_dtype=%" PRIu32 "\n", result.scale_data_type);
    printf("gptq_reference_g_idx=%" PRIu32 "\n", result.has_g_idx);
    printf("gptq_reference_operations=%" PRIu64 "\n", result.operations);
    printf("gptq_reference_operations_low=0x%08" PRIx32 "\n",
           (uint32_t)result.operations);
    printf("gptq_reference_bytes_read=%" PRIu64 "\n", result.bytes_read);
    printf("gptq_reference_bytes_written=%" PRIu64 "\n", result.bytes_written);
    printf("gptq_reference_modeled_cycles=%" PRIu64 "\n",
           result.modeled_cycles);
    printf("gptq_reference_checksum=0x%08" PRIx32 "\n", result.output_checksum);
    printf("gptq_reference=PASS\n");
    return 0;
}
