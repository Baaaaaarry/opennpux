#ifndef OPENNPUX_NPU_XGRAPH_LOWERING_H
#define OPENNPUX_NPU_XGRAPH_LOWERING_H

#include "opennpux/npu_functional_request.h"
#include "opennpux/npu_gptq_tile_plan.h"
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

enum opennpux_npu_xgraph_tensor_layout {
    OPENNPUX_NPU_XGRAPH_LAYOUT_UNSPECIFIED = 0,
    OPENNPUX_NPU_XGRAPH_LAYOUT_NHWC = 1,
    OPENNPUX_NPU_XGRAPH_LAYOUT_OHWI = 2,
};

struct opennpux_npu_xgraph_convolution_options {
    uint32_t input_height;
    uint32_t input_width;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t output_channels;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t stride_height;
    uint32_t stride_width;
    uint32_t padding_top;
    uint32_t padding_bottom;
    uint32_t padding_left;
    uint32_t padding_right;
    uint32_t dilation_height;
    uint32_t dilation_width;
    uint32_t groups;
    uint32_t input_layout;
    uint32_t weight_layout;
    uint32_t output_layout;
};

struct opennpux_npu_xgraph_lowering_options {
    uint32_t rope_layout;
    uint32_t activation;
    uint32_t topk_packed_address;
    uint32_t topk_packed_size;
    struct opennpux_npu_xgraph_convolution_options convolution;
};

struct opennpux_npu_xgraph_lowering_failure {
    uint32_t command_index;
    uint32_t command_id;
    uint32_t opcode;
    int32_t error_code;
};

/*
 * Lower one materialized generic command to one XOpenNPUX primitive record.
 * Generic commands with primitive-equivalent semantics may be canonicalized
 * (for example COMBINE to TADD). Stateful causal depthwise convolution is
 * represented by one TCAUSALCONV record with auxiliary state offsets. Generic
 * FP32 Conv2D is represented by one TCONV record when explicit NHWC/OHWI
 * convolution options are present. Other
 * composite commands return ENOTSUP and
 * must first pass through a decomposition pass. All operand addresses must
 * belong to the NPU EXTMEM aperture. GPTQ MatMul also requires
 * decomposition/dequantization and is not silently treated as FP32 TMMA.
 */
int opennpux_npu_xgraph_lower_primitive(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_xgraph_lowering_options *options,
    uint32_t extmem_base, uint32_t extmem_size,
    struct opennpux_xgraph_command *command);

/*
 * Lower an ordered materialized command sequence without model-specific
 * interpretation. Command IDs must be dense and match sequence order because
 * firmware uses them as the retirement/completion index. On failure, no
 * command after the reported index is emitted.
 */
int opennpux_npu_xgraph_lower_sequence(
    const struct opennpux_npu_functional_request *requests,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_xgraph_lowering_options *options,
    uint32_t command_count, uint32_t extmem_base, uint32_t extmem_size,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    struct opennpux_npu_xgraph_lowering_failure *failure);

/*
 * Decompose one GPTQ MatMul into executable XOpenNPUX records. Each N/K tile is
 * dequantized once. One-row TMMA records preserve strided output views; later
 * K tiles write a reusable partial buffer and TADD accumulates into output.
 */
int opennpux_npu_xgraph_lower_gptq_matmul(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count);

/*
 * Decompose one GPTQ gated expert into three tiled projections and the
 * intervening SiLU/multiply activation. Gate, up, and down projections reuse
 * the dequant scratch sequentially; their tensor outputs remain explicit
 * operands so the sequence is independent of model-specific memory layouts.
 */
int opennpux_npu_xgraph_lower_gptq_expert(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count);

/*
 * Lower one KV-cache update into two contiguous TDMA records. The generic
 * request supplies key/value inputs and a [2, kv_length, kv_heads, head_dim]
 * destination state. Each record updates the visible tail of one state plane.
 */
int opennpux_npu_xgraph_lower_dma(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count);

/*
 * Lower one recurrent state update. Basic copy semantics use two contiguous
 * TDMA records. Gated-delta semantics use one TRECURRENT record carrying QKV,
 * alpha, beta, persistent state, A-log, and dt-bias through generic operands.
 */
int opennpux_npu_xgraph_lower_recurrent_update(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count);

/*
 * Lower one dense FP32 MoE router into projection, selection, normalization,
 * and split result writeback. Scratch contains the transient logits followed
 * by the packed Top-K values/indices. The five records are one atomic batch
 * item and expose no model-specific expert policy to the NPU ISA.
 */
int opennpux_npu_xgraph_lower_router(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count);

/*
 * Lower as many ordered generic commands as fit in one bounded XGraph batch.
 * Primitive commands emit one record; composite GPTQ MatMul commands emit a
 * tiled TDEQUANT/TMMA/TADD sequence, GPTQ Expert emits three such projections
 * around TSILU/TMUL, DMA and basic recurrent state updates emit two TDMA
 * records, and Router emits TMMA/TTOPK/TSOFTMAX plus two TDMA writebacks.
 * Output command IDs are local to the batch and dense from zero.
 * command_origins, when non-NULL, maps each emitted record back to its input
 * request command_id.
 *
 * A full next request is never split across batches. If at least one request
 * was emitted, an ENOSPC result for the next request ends the batch normally.
 * If the first request cannot fit, the call fails with ENOSPC.
 */
int opennpux_npu_xgraph_lower_batch(
    const struct opennpux_npu_functional_request *requests,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_xgraph_lowering_options *options,
    uint32_t request_count, uint32_t extmem_base, uint32_t extmem_size,
    uint32_t scratch_address, uint32_t scratch_size,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_origins, uint32_t *requests_consumed,
    uint32_t *commands_emitted,
    struct opennpux_npu_xgraph_lowering_failure *failure);

#ifdef __cplusplus
}
#endif

#endif
