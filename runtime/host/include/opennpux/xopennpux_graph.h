#ifndef OPENNPUX_XOPENNPUX_GRAPH_H
#define OPENNPUX_XOPENNPUX_GRAPH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_XGRAPH_MAGIC UINT32_C(0x5847504e)
#define OPENNPUX_XGRAPH_VERSION UINT32_C(2)
#define OPENNPUX_XGRAPH_OFFSET UINT32_C(0x00010000)
#define OPENNPUX_XGRAPH_DATA_OFFSET UINT32_C(0x00020000)
/* 768 records fit before DATA_OFFSET and cover the current 524-command graph. */
#define OPENNPUX_XGRAPH_MAX_COMMANDS UINT32_C(768)

#define OPENNPUX_XGRAPH_STATE_EMPTY UINT32_C(0)
#define OPENNPUX_XGRAPH_STATE_READY UINT32_C(1)
#define OPENNPUX_XGRAPH_STATE_RUNNING UINT32_C(2)
#define OPENNPUX_XGRAPH_STATE_COMPLETE UINT32_C(3)
#define OPENNPUX_XGRAPH_STATE_ERROR UINT32_C(0x80000000)

#define OPENNPUX_XGRAPH_ERROR_NONE UINT32_C(0)
#define OPENNPUX_XGRAPH_ERROR_ABI UINT32_C(1)
#define OPENNPUX_XGRAPH_ERROR_OPCODE UINT32_C(2)
#define OPENNPUX_XGRAPH_ERROR_BOUNDS UINT32_C(3)
#define OPENNPUX_XGRAPH_ERROR_RESULT UINT32_C(4)

/* Batch metadata carried in opennpux_xgraph_header::reserved. */
#define OPENNPUX_XGRAPH_BATCH_SEQUENCE UINT32_C(0)
#define OPENNPUX_XGRAPH_BATCH_FIRST_REQUEST UINT32_C(1)
#define OPENNPUX_XGRAPH_BATCH_REQUEST_COUNT UINT32_C(2)
#define OPENNPUX_XGRAPH_BATCH_FLAGS UINT32_C(3)
#define OPENNPUX_XGRAPH_BATCH_FIRST_COMMAND UINT32_C(4)
#define OPENNPUX_XGRAPH_BATCH_FLAG_FINAL UINT32_C(1)

enum opennpux_xgraph_opcode {
    OPENNPUX_XGRAPH_OP_TMMA = 1,
    OPENNPUX_XGRAPH_OP_TADD = 2,
    OPENNPUX_XGRAPH_OP_TMUL = 3,
    OPENNPUX_XGRAPH_OP_TRMSNORM = 4,
    OPENNPUX_XGRAPH_OP_TSOFTMAX = 5,
    OPENNPUX_XGRAPH_OP_TROPE = 6,
    OPENNPUX_XGRAPH_OP_TSILU = 7,
    OPENNPUX_XGRAPH_OP_TGATHER = 8,
    OPENNPUX_XGRAPH_OP_TTOPK = 9,
    OPENNPUX_XGRAPH_OP_TDEQUANT = 10,
    OPENNPUX_XGRAPH_OP_TDMA = 11,
    OPENNPUX_XGRAPH_OP_TCAUSALCONV = 12,
    OPENNPUX_XGRAPH_OP_TATTENTION = 13,
    OPENNPUX_XGRAPH_OP_TRECURRENT = 14,
    OPENNPUX_XGRAPH_OP_TCONV = 15,
    OPENNPUX_XGRAPH_OP_TSIGMOID = 16,
    OPENNPUX_XGRAPH_OP_TROW_SCALE = 17,
};

#define OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL UINT32_C(1)
#define OPENNPUX_XGRAPH_TCAUSALCONV_SILU UINT32_C(2)
#define OPENNPUX_XGRAPH_TATTENTION_GATED UINT32_C(1)
#define OPENNPUX_XGRAPH_TTOPK_SPLIT_OUTPUT UINT32_C(1)
#define OPENNPUX_XGRAPH_TMMA_TRANSPOSE_RHS UINT32_C(1)
#define OPENNPUX_XGRAPH_TMMA_ACCUMULATE UINT32_C(2)
#define OPENNPUX_XGRAPH_TRMSNORM_WEIGHT_OFFSET UINT32_C(1)
#define OPENNPUX_XGRAPH_TRMSNORM_BFLOAT16_INPUT UINT32_C(2)

enum opennpux_xgraph_data_type {
    OPENNPUX_XGRAPH_DTYPE_FP32 = 2,
};

/*
 * A compiler/runtime command. Addresses are byte offsets in the shared DMA
 * window, not host pointers or model-specific tensor identifiers. The Coral
 * firmware lowers each record to custom CSR writes plus one 32-bit
 * XOpenNPUX instruction.
 */
struct opennpux_xgraph_command {
    uint32_t opcode;
    uint32_t flags;
    uint32_t destination_offset;
    uint32_t source0_offset;
    uint32_t source1_offset;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    uint32_t scalar0;
    uint32_t data_type;
    uint32_t command_id;
    uint32_t reserved[5];
};

/*
 * TDEQUANT reuses the fixed command fields as follows:
 *   destination/source0/source1: FP32 scratch/qweight/qzeros offsets
 *   dim0/dim1/dim2: 1/tile-N/input-K
 *   scalar0: XOpenNPUX quant config
 *   flags: global quant group count `[31:16]`, tile group base `[15:0]`
 *   reserved[0..4]: scales offset, optional g_idx offset, then qweight,
 *                   qzeros, and scales row strides in bytes.
 *
 * TMMA uses reserved[0..2] as lhs, rhs and destination row strides in bytes.
 * Zero selects the contiguous stride implied by dim2/dim1. The transpose-RHS
 * flag interprets source1 as row-major [N,K], and accumulate initializes each
 * output accumulator from its current destination value.
 *
 * TCAUSALCONV uses dim0/dim1/dim2 as rows/features/kernel width. For a
 * stateful command, reserved[0] and reserved[1] contain previous-state and
 * next-state offsets. The state has shape [kernel_width - 1, features].
 *
 * TATTENTION uses dim0/dim1/dim2 as query rows, query heads, and head
 * dimension. scalar0 is the KV-head count and flags is the KV length.
 * source1 contains K followed by V, each [kv_length, kv_heads, head_dim].
 * reserved[0] is an optional FP32 sigmoid-gate tensor and reserved[1] carries
 * OPENNPUX_XGRAPH_TATTENTION_* flags.
 *
 * TRECURRENT uses dim0/dim1/dim2 as rows, key heads and key dimension.
 * scalar0 packs value heads [15:0] and value dimension [31:16]. source0/source1
 * are QKV/alpha, destination is output, and reserved[0..3] contain beta,
 * persistent state, A-log and dt-bias offsets.
 *
 * TCONV uses dim0/dim1/dim2/scalar0 as batch, input height, input width and
 * input channels. flags packs output channels [15:0] and groups [31:16].
 * source0/source1/destination are NHWC input, OHWI weights and NHWC output.
 * reserved[0] is an optional FP32 bias offset. reserved[1]/[2] pack output
 * and kernel H/W as 16-bit fields. reserved[3] packs stride H/W and dilation
 * H/W as four 8-bit fields; reserved[4] packs T/B/L/R padding as four 8-bit
 * fields. The firmware expands these into the dedicated custom CSRs.
 *
 * TTOPK normally writes packed FP32 values followed by uint32 indices. With
 * OPENNPUX_XGRAPH_TTOPK_SPLIT_OUTPUT, destination is the values tensor and
 * reserved[0] is the independent indices tensor offset.
 *
 * TROW_SCALE multiplies each FP32 source0 row by the corresponding scalar in
 * source1. dim0/dim1 are row and feature counts; destination has the same
 * shape as source0 and source1 contains dim0 FP32 values.
 */

struct opennpux_xgraph_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t command_size;
    uint32_t command_count;
    uint32_t total_size;
    uint32_t state;
    uint32_t error;
    uint32_t completed_commands;
    uint32_t output_offset;
    uint32_t output_bytes;
    uint32_t output_checksum;
    uint64_t operation_count;
    uint64_t modeled_cycles;
    uint32_t reserved[8];
};

#ifdef __cplusplus
static_assert(sizeof(opennpux_xgraph_command) == 64,
              "XOpenNPUX graph command ABI changed");
static_assert(sizeof(opennpux_xgraph_header) == 96,
              "XOpenNPUX graph header ABI changed");
static_assert(OPENNPUX_XGRAPH_OFFSET + sizeof(opennpux_xgraph_header) +
                  OPENNPUX_XGRAPH_MAX_COMMANDS *
                      sizeof(opennpux_xgraph_command) <=
              OPENNPUX_XGRAPH_DATA_OFFSET,
              "XOpenNPUX command region overlaps tensor data");
#else
_Static_assert(sizeof(struct opennpux_xgraph_command) == 64,
               "XOpenNPUX graph command ABI changed");
_Static_assert(sizeof(struct opennpux_xgraph_header) == 96,
               "XOpenNPUX graph header ABI changed");
_Static_assert(OPENNPUX_XGRAPH_OFFSET +
                       sizeof(struct opennpux_xgraph_header) +
                       OPENNPUX_XGRAPH_MAX_COMMANDS *
                           sizeof(struct opennpux_xgraph_command) <=
                   OPENNPUX_XGRAPH_DATA_OFFSET,
               "XOpenNPUX command region overlaps tensor data");
#endif

#ifdef __cplusplus
}
#endif

#endif
