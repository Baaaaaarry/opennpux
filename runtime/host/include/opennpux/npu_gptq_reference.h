#ifndef OPENNPUX_NPU_GPTQ_REFERENCE_H
#define OPENNPUX_NPU_GPTQ_REFERENCE_H

#include "opennpux/coral_gptq_matmul.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Bit-exact host reference for the packed int4 GPTQ MatMul that the Coral
 * bridge executes for CORAL_OPERATOR_OP_GPTQ_MATMUL_INT4. It consumes the same
 * staged EXTMEM image the device consumes, so a full-system run can be gated on
 * an exact output checksum instead of a non-zero one.
 *
 * The reference accumulates in float32 in the same order as the bridge kernel.
 * It must therefore be compiled with -ffp-contract=off: fusing the inner
 * multiply-add changes the rounding and breaks the exact checksum comparison on
 * targets whose baseline includes FMA.
 */

#define OPENNPUX_NPU_GPTQ_REFERENCE_EXTMEM_BASE UINT64_C(0x20000000)

struct opennpux_npu_gptq_reference_result {
    uint32_t rows;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t group_size;
    uint32_t zero_bias;
    uint32_t scale_data_type;
    uint32_t has_g_idx;
    uint32_t output_checksum;
    uint64_t operations;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t modeled_cycles;
};

/*
 * Validate the request header and every operand extent inside a staged image,
 * then recompute the projection. `device_base` is the Coral address that maps
 * to image offset 0. `output` may be NULL; otherwise it receives
 * rows * output_columns floats and `output_floats` bounds it.
 */
int opennpux_npu_gptq_reference_run(
    const void *image, size_t image_size, uint64_t device_base,
    float *output, size_t output_floats,
    struct opennpux_npu_gptq_reference_result *result);

/* FNV-1a over the raw float bytes, matching the bridge output checksum. */
uint32_t opennpux_npu_gptq_reference_checksum(
    const float *values, size_t count);

#ifdef __cplusplus
}
#endif

#endif
