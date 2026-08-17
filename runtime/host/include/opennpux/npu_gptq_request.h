#ifndef OPENNPUX_NPU_GPTQ_REQUEST_H
#define OPENNPUX_NPU_GPTQ_REQUEST_H

#include "opennpux/coral_gptq_matmul.h"
#include "opennpux/coral_gptq_expert.h"
#include "opennpux/model_package.h"
#include "opennpux/npu_weight_ranges.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stage one packed int4 GPTQ projection for the Coral CUSTOM_0 path.
 *
 * This is the runtime-owned counterpart of the offline
 * tools/models/materialize_gptq_projection.py harness: the CPU resolves a
 * command's exact weight components, binds live input and output buffers, and
 * writes the request record the firmware consumes. Both producers emit the same
 * image, so an offline-materialized projection and a runtime-staged one remain
 * interchangeable.
 *
 * The shape is supplied by the caller rather than derived from tensor names.
 * Projection geometry belongs to the compiler frontend; the runtime only checks
 * that the resolved components match the shape it was given.
 */

#define OPENNPUX_NPU_GPTQ_REQUEST_ALIGNMENT UINT32_C(64)

struct opennpux_npu_gptq_projection_selector {
    uint32_t command_id;
    uint32_t role_id;
    uint64_t expert_id;
    uint32_t slot_id;
};

struct opennpux_npu_gptq_projection_shape {
    uint32_t rows;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t group_size;
    uint32_t zero_bias;
};

struct opennpux_npu_gptq_request_layout {
    uint32_t input_offset;
    uint32_t qweight_offset;
    uint32_t qzeros_offset;
    uint32_t scales_offset;
    uint32_t g_idx_offset;
    uint32_t output_offset;
    uint32_t output_bytes;
    uint32_t total_size;
    uint32_t scale_data_type;
};

struct opennpux_npu_gptq_expert_selector {
    uint32_t command_id;
    uint32_t role_id;
    uint64_t expert_id;
};

struct opennpux_npu_gptq_expert_shape {
    uint32_t rows;
    uint32_t hidden_columns;
    uint32_t intermediate_columns;
    uint32_t group_size;
    uint32_t zero_bias;
};

struct opennpux_npu_gptq_weight_layout {
    uint32_t qweight_offset;
    uint32_t qzeros_offset;
    uint32_t scales_offset;
    uint32_t g_idx_offset;
    uint32_t scale_data_type;
};

struct opennpux_npu_gptq_expert_layout {
    uint32_t input_offset;
    uint32_t gate_output_offset;
    uint32_t up_output_offset;
    uint32_t activated_offset;
    uint32_t output_offset;
    struct opennpux_npu_gptq_weight_layout gate;
    struct opennpux_npu_gptq_weight_layout up;
    struct opennpux_npu_gptq_weight_layout down;
    uint32_t output_bytes;
    uint32_t total_size;
};

/*
 * Compute the image layout without touching the model. `scale_element_size`
 * must be 2 for FP16/BF16 scales and 4 for FP32 scales, and `has_g_idx`
 * reserves the optional group-index operand.
 */
int opennpux_npu_gptq_request_layout(
    const struct opennpux_npu_gptq_projection_shape *shape,
    uint32_t scale_element_size, uint32_t has_g_idx,
    struct opennpux_npu_gptq_request_layout *layout);

/*
 * Resolve the selected command's components, validate them against `shape`,
 * and write the complete request image into `image`. `device_base` is the Coral
 * address that maps to image offset 0. The output region is zeroed.
 */
int opennpux_npu_gptq_request_stage(
    const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_ranges *ranges,
    const struct opennpux_npu_gptq_projection_selector *selector,
    const struct opennpux_npu_gptq_projection_shape *shape,
    const float *input, size_t input_floats, uint64_t device_base,
    void *image, size_t image_size,
    struct opennpux_npu_gptq_request_layout *layout);

/* Stage a complete gate/up/SiLU-Mul/down expert from live CPU input. */
int opennpux_npu_gptq_expert_stage(
    const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_ranges *ranges,
    const struct opennpux_npu_gptq_expert_selector *selector,
    const struct opennpux_npu_gptq_expert_shape *shape,
    const float *input, size_t input_floats, uint64_t device_base,
    void *image, size_t image_size,
    struct opennpux_npu_gptq_expert_layout *layout);

#ifdef __cplusplus
}
#endif

#endif
