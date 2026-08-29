#include "opennpux/npu_gptq_tile_plan.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#define BASE UINT32_C(0x20000000)
#define SIZE UINT32_C(0x00800000)

static void
add_operand(struct opennpux_npu_functional_request *request, uint32_t role,
            uint32_t offset, uint32_t bytes)
{
    struct opennpux_npu_functional_operand *operand =
        &request->operands[request->operand_count++];
    *operand = (struct opennpux_npu_functional_operand){
        role, BASE + offset, bytes, 0};
}

static void
initialize(struct opennpux_npu_functional_request *request,
           struct opennpux_npu_operator_parameters *parameters,
           uint32_t output_columns, int has_g_idx)
{
    memset(request, 0, sizeof(*request));
    request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
    request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
    request->struct_size = sizeof(*request);
    request->opcode = OPENNPUX_NPU_OP_MATMUL;
    request->rows = 3;
    request->features = 32;
    add_operand(request, OPENNPUX_NPU_OPERAND_INPUT, 0x1000, 3 * 32 * 4);
    add_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x2000,
                3 * output_columns * 4);
    add_operand(request, OPENNPUX_NPU_OPERAND_QWEIGHT, 0x3000,
                4 * output_columns * 4);
    add_operand(request, OPENNPUX_NPU_OPERAND_QZEROS, 0x5000,
                2 * ((output_columns + 7) / 8) * 4);
    add_operand(request, OPENNPUX_NPU_OPERAND_SCALES, 0x6000,
                2 * output_columns * 2);
    if (has_g_idx) {
        add_operand(request, OPENNPUX_NPU_OPERAND_G_IDX, 0x7000, 32 * 4);
    }

    memset(parameters, 0, sizeof(*parameters));
    parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
    parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
    parameters->struct_size = sizeof(*parameters);
    parameters->opcode = OPENNPUX_NPU_OP_MATMUL;
    parameters->flags = OPENNPUX_NPU_PARAMETER_GPTQ;
    parameters->input_features = 32;
    parameters->output_features = output_columns;
    parameters->quantization_bits = 4;
    parameters->quantization_group_size = 16;
    parameters->scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT16;
    parameters->quantized_zero_bias = 1;
}

static void
test_strided_tiles(void)
{
    struct opennpux_npu_functional_request request;
    struct opennpux_npu_operator_parameters parameters;
    struct opennpux_npu_gptq_tile_plan plan;
    struct opennpux_npu_gptq_tile tile;
    initialize(&request, &parameters, 18, 1);
    assert(opennpux_npu_gptq_plan_tiles(
               &request, &parameters, BASE, SIZE, BASE + 0x10000,
               32 * 8 * 4, &plan) == 0);
    assert(plan.tile_columns == 8);
    assert(plan.tile_count == 3);
    assert(plan.group_count == 2);
    assert(plan.has_g_idx == 1);

    assert(opennpux_npu_gptq_get_tile(&request, &plan, 1, &tile) == 0);
    assert(tile.column_base == 8 && tile.column_count == 8);
    assert(tile.dequantized_bytes == 32 * 8 * 4);
    assert(tile.qweight.address == BASE + 0x3000 + 8 * 4);
    assert(tile.qweight.row_count == 4);
    assert(tile.qweight.row_stride_bytes == 18 * 4);
    assert(tile.qweight.row_bytes == 8 * 4);
    assert(tile.qzeros.address == BASE + 0x5000 + 4);
    assert(tile.qzeros.row_count == 2);
    assert(tile.qzeros.row_stride_bytes == 3 * 4);
    assert(tile.qzeros.row_bytes == 4);
    assert(tile.scales.address == BASE + 0x6000 + 8 * 2);
    assert(tile.scales.row_stride_bytes == 18 * 2);
    assert(tile.scales.row_bytes == 8 * 2);
    assert(tile.g_idx.row_bytes == 32 * 4);
    assert(tile.output.address == BASE + 0x2000 + 8 * 4);
    assert(tile.output.row_stride_bytes == 18 * 4);

    assert(opennpux_npu_gptq_get_tile(&request, &plan, 2, &tile) == 0);
    assert(tile.column_base == 16 && tile.column_count == 2);
    assert(tile.qzeros.address == BASE + 0x5000 + 8);
    assert(tile.qzeros.row_bytes == 4);
    assert(tile.dequantized_bytes == 32 * 2 * 4);
}

static void
test_capacity_and_validation(void)
{
    struct opennpux_npu_functional_request request;
    struct opennpux_npu_operator_parameters parameters;
    struct opennpux_npu_gptq_tile_plan plan;
    initialize(&request, &parameters, 7, 0);
    assert(opennpux_npu_gptq_plan_tiles(
               &request, &parameters, BASE, SIZE, BASE + 0x10000,
               32 * 7 * 4, &plan) == 0);
    assert(plan.tile_columns == 7 && plan.tile_count == 1);
    assert(plan.has_g_idx == 0);

    initialize(&request, &parameters, 18, 0);
    errno = 0;
    assert(opennpux_npu_gptq_plan_tiles(
               &request, &parameters, BASE, SIZE, BASE + 0x10000,
               32 * 7 * 4, &plan) == -1);
    assert(errno == ENOSPC);

    parameters.quantization_bits = 8;
    errno = 0;
    assert(opennpux_npu_gptq_plan_tiles(
               &request, &parameters, BASE, SIZE, BASE + 0x10000,
               32 * 8 * 4, &plan) == -1);
    assert(errno == EINVAL);

    parameters.quantization_bits = 4;
    request.operands[2].byte_size--;
    errno = 0;
    assert(opennpux_npu_gptq_plan_tiles(
               &request, &parameters, BASE, SIZE, BASE + 0x10000,
               32 * 8 * 4, &plan) == -1);
    assert(errno == ENOSPC);
}

static void
test_large_k_tiles(void)
{
    struct opennpux_npu_functional_request request;
    struct opennpux_npu_operator_parameters parameters;
    struct opennpux_npu_gptq_tile_plan plan;
    struct opennpux_npu_gptq_tile tile;
    initialize(&request, &parameters, 18, 1);
    request.rows = 2;
    request.features = 2048;
    parameters.input_features = 2048;
    parameters.quantization_group_size = 128;
    request.operands[0].byte_size = 2 * 2048 * 4;
    request.operands[1].byte_size = 2 * 18 * 4;
    request.operands[2].byte_size = 256 * 18 * 4;
    request.operands[3].byte_size = 16 * 3 * 4;
    request.operands[4].byte_size = 16 * 18 * 2;
    request.operands[5].byte_size = 2048 * 4;

    assert(opennpux_npu_gptq_plan_tiles(
               &request, &parameters, BASE, SIZE, BASE + 0x10000,
               (896 + 1) * 8 * 4, &plan) == 0);
    assert(plan.input_tile_columns == 896);
    assert(plan.input_tile_count == 3);
    assert(plan.tile_columns == 8 && plan.tile_count == 3);
    assert(plan.partial_address == BASE + 0x10000 + 896 * 8 * 4);
    assert(plan.partial_size == 8 * 4);

    assert(opennpux_npu_gptq_get_tile_2d(&request, &plan, 1, 1, &tile) == 0);
    assert(tile.column_base == 8 && tile.input_base == 896);
    assert(tile.input_count == 896 && tile.group_base == 7);
    assert(tile.group_count == 16);
    assert(tile.qweight.address ==
           BASE + 0x3000 + (896 / 8) * 18 * 4 + 8 * 4);
    assert(tile.g_idx.address == BASE + 0x7000 + 896 * 4);
    assert(tile.dequantized_bytes == 896 * 8 * 4);

    assert(opennpux_npu_gptq_get_tile_2d(&request, &plan, 1, 2, &tile) == 0);
    assert(tile.input_base == 1792 && tile.input_count == 256);
    assert(tile.group_base == 14);
}

int
main(void)
{
    test_strided_tiles();
    test_capacity_and_validation();
    test_large_k_tiles();
    puts("NPU GPTQ tile plan test: PASS");
    return 0;
}
