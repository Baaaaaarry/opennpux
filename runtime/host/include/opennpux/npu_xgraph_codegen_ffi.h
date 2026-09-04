#ifndef OPENNPUX_NPU_XGRAPH_CODEGEN_FFI_H
#define OPENNPUX_NPU_XGRAPH_CODEGEN_FFI_H

#include "opennpux/xopennpux_graph.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_XGRAPH_DENSE_SPEC_SIZE UINT32_C(56)

struct opennpux_xgraph_dense_spec {
    uint32_t struct_size;
    uint32_t extmem_base;
    uint32_t extmem_size;
    uint32_t lhs_offset;
    uint32_t lhs_bytes;
    uint32_t rhs_offset;
    uint32_t rhs_bytes;
    uint32_t output_offset;
    uint32_t output_bytes;
    uint32_t rows;
    uint32_t input_features;
    uint32_t output_features;
    uint32_t transpose_rhs;
    uint32_t first_command_id;
};

/* Stable compiler FFI: lower one dense MatMul through the runtime C tiler. */
int opennpux_xgraph_codegen_dense_matmul(
    const struct opennpux_xgraph_dense_spec *spec,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count);

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_xgraph_dense_spec) ==
              OPENNPUX_XGRAPH_DENSE_SPEC_SIZE);
#else
_Static_assert(sizeof(struct opennpux_xgraph_dense_spec) ==
                   OPENNPUX_XGRAPH_DENSE_SPEC_SIZE,
               "XGraph dense compiler spec ABI size changed");
#endif

#ifdef __cplusplus
}
#endif

#endif
