#ifndef OPENNPUX_NPU_GPTQ_TILE_PLAN_H
#define OPENNPUX_NPU_GPTQ_TILE_PLAN_H

#include "opennpux/npu_functional_request.h"
#include "opennpux/npu_submission.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A strided source view consumed by one device-side tile operation. */
struct opennpux_npu_gptq_component_view {
    uint32_t address;
    uint32_t row_count;
    uint32_t row_stride_bytes;
    uint32_t row_bytes;
};

struct opennpux_npu_gptq_tile_plan {
    uint32_t rows;
    uint32_t input_columns;
    uint32_t output_columns;
    uint32_t group_count;
    uint32_t scale_element_bytes;
    uint32_t tile_columns;
    uint32_t tile_count;
    uint32_t scratch_address;
    uint32_t scratch_size;
    uint32_t has_g_idx;
};

struct opennpux_npu_gptq_tile {
    uint32_t column_base;
    uint32_t column_count;
    uint32_t dequantized_address;
    uint32_t dequantized_bytes;
    struct opennpux_npu_gptq_component_view qweight;
    struct opennpux_npu_gptq_component_view qzeros;
    struct opennpux_npu_gptq_component_view scales;
    struct opennpux_npu_gptq_component_view g_idx;
    struct opennpux_npu_gptq_component_view output;
};

/*
 * Validate one materialized GPTQ MatMul and choose the largest N tile that
 * fits the caller-owned dequantization scratch region. Non-final tiles are
 * aligned to eight output columns so packed qzeros never straddle tiles.
 */
int opennpux_npu_gptq_plan_tiles(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, struct opennpux_npu_gptq_tile_plan *plan);

/* Materialize the strided component ranges for one planned output tile. */
int opennpux_npu_gptq_get_tile(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_gptq_tile_plan *plan, uint32_t tile_index,
    struct opennpux_npu_gptq_tile *tile);

#ifdef __cplusplus
}
#endif

#endif
