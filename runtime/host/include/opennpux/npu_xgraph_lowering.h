#ifndef OPENNPUX_NPU_XGRAPH_LOWERING_H
#define OPENNPUX_NPU_XGRAPH_LOWERING_H

#include "opennpux/npu_functional_request.h"
#include "opennpux/npu_submission.h"
#include "opennpux/xopennpux_graph.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum opennpux_npu_xgraph_rope_layout {
    OPENNPUX_NPU_XGRAPH_ROPE_ADJACENT = 0,
    OPENNPUX_NPU_XGRAPH_ROPE_HALF_SPLIT = 1,
};

enum opennpux_npu_xgraph_activation {
    OPENNPUX_NPU_XGRAPH_ACTIVATION_NONE = 0,
    OPENNPUX_NPU_XGRAPH_ACTIVATION_SILU = 1,
};

struct opennpux_npu_xgraph_lowering_options {
    uint32_t rope_layout;
    uint32_t activation;
    uint32_t topk_packed_address;
    uint32_t topk_packed_size;
};

/*
 * Lower one materialized generic command to one XOpenNPUX primitive record.
 * Composite generic commands return ENOTSUP and must first pass through a
 * decomposition pass. All operand addresses must belong to the NPU EXTMEM
 * aperture. GPTQ MatMul also requires decomposition/dequantization and is not
 * silently treated as FP32 TMMA.
 */
int opennpux_npu_xgraph_lower_primitive(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_xgraph_lowering_options *options,
    uint32_t extmem_base, uint32_t extmem_size,
    struct opennpux_xgraph_command *command);

#ifdef __cplusplus
}
#endif

#endif
