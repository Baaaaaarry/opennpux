#include "opennpux/npu_gptq_tile_plan.h"

#include <errno.h>
#include <string.h>

static uint32_t
ceil_div(uint32_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1u : 0u);
}

static uint32_t
greatest_common_divisor(uint32_t first, uint32_t second)
{
    while (second != 0) {
        const uint32_t remainder = first % second;
        first = second;
        second = remainder;
    }
    return first;
}

static const struct opennpux_npu_functional_operand *
find_operand(const struct opennpux_npu_functional_request *request,
             uint32_t role)
{
    for (uint32_t index = 0; index < request->operand_count; ++index) {
        if (request->operands[index].role == role) {
            return &request->operands[index];
        }
    }
    return NULL;
}

static int
product3(uint32_t first, uint32_t second, uint32_t third, uint64_t *value)
{
    const uint64_t product = (uint64_t)first * second;
    if (third != 0 && product > UINT64_MAX / third) {
        errno = EOVERFLOW;
        return -1;
    }
    *value = product * third;
    return 0;
}

static int
validate_region(uint32_t address, uint64_t bytes, uint32_t extmem_base,
                uint32_t extmem_size)
{
    if (address < extmem_base) {
        errno = ERANGE;
        return -1;
    }
    const uint64_t offset = (uint64_t)address - extmem_base;
    if (bytes == 0 || offset > extmem_size || bytes > extmem_size - offset) {
        errno = ERANGE;
        return -1;
    }
    return 0;
}

static int
validate_operand(const struct opennpux_npu_functional_operand *operand,
                 uint64_t required, uint32_t extmem_base,
                 uint32_t extmem_size)
{
    if (operand == NULL || required > operand->byte_size) {
        errno = operand == NULL ? EINVAL : ENOSPC;
        return -1;
    }
    return validate_region(operand->address, required, extmem_base,
                           extmem_size);
}

static uint32_t
scale_element_size(uint32_t data_type)
{
    if (data_type == OPENNPUX_NPU_DTYPE_FLOAT16 ||
        data_type == OPENNPUX_NPU_DTYPE_BFLOAT16) {
        return 2;
    }
    return data_type == OPENNPUX_NPU_DTYPE_FLOAT32 ? 4u : 0u;
}

int
opennpux_npu_gptq_plan_tiles(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, struct opennpux_npu_gptq_tile_plan *plan)
{
    if (request == NULL || parameters == NULL || plan == NULL ||
        extmem_size == 0 || request->magic != OPENNPUX_NPU_FUNCTIONAL_MAGIC ||
        request->version != OPENNPUX_NPU_FUNCTIONAL_VERSION ||
        request->struct_size != sizeof(*request) ||
        request->operand_count > OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS ||
        request->opcode != OPENNPUX_NPU_OP_MATMUL || request->rows == 0 ||
        parameters->magic != OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC ||
        parameters->version != OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION ||
        parameters->struct_size != sizeof(*parameters) ||
        parameters->opcode != OPENNPUX_NPU_OP_MATMUL ||
        (parameters->flags & OPENNPUX_NPU_PARAMETER_GPTQ) == 0 ||
        parameters->input_features == 0 || parameters->output_features == 0 ||
        parameters->quantization_bits != 4 ||
        parameters->quantization_group_size == 0 ||
        parameters->quantized_zero_bias > 15) {
        errno = EINVAL;
        return -1;
    }

    const uint32_t scale_bytes =
        scale_element_size(parameters->scale_data_type);
    if (scale_bytes == 0) {
        errno = EINVAL;
        return -1;
    }
    const uint32_t rows = request->rows;
    const uint32_t input_columns = parameters->input_features;
    const uint32_t output_columns = parameters->output_features;
    const uint32_t weight_rows = ceil_div(input_columns, 8);
    const uint32_t groups =
        ceil_div(input_columns, parameters->quantization_group_size);
    if (groups > UINT16_MAX) {
        errno = EOVERFLOW;
        return -1;
    }
    const uint32_t zero_columns = ceil_div(output_columns, 8);
    uint64_t input_bytes;
    uint64_t output_bytes;
    uint64_t qweight_bytes;
    uint64_t qzeros_bytes;
    uint64_t scales_bytes;
    uint64_t g_idx_bytes;
    if (product3(rows, input_columns, 4, &input_bytes) != 0 ||
        product3(rows, output_columns, 4, &output_bytes) != 0 ||
        product3(weight_rows, output_columns, 4, &qweight_bytes) != 0 ||
        product3(groups, zero_columns, 4, &qzeros_bytes) != 0 ||
        product3(groups, output_columns, scale_bytes, &scales_bytes) != 0 ||
        product3(input_columns, 1, 4, &g_idx_bytes) != 0) {
        return -1;
    }

    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *output =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const struct opennpux_npu_functional_operand *qweight =
        find_operand(request, OPENNPUX_NPU_OPERAND_QWEIGHT);
    const struct opennpux_npu_functional_operand *qzeros =
        find_operand(request, OPENNPUX_NPU_OPERAND_QZEROS);
    const struct opennpux_npu_functional_operand *scales =
        find_operand(request, OPENNPUX_NPU_OPERAND_SCALES);
    const struct opennpux_npu_functional_operand *g_idx =
        find_operand(request, OPENNPUX_NPU_OPERAND_G_IDX);
    if (validate_operand(input, input_bytes, extmem_base, extmem_size) != 0 ||
        validate_operand(output, output_bytes, extmem_base, extmem_size) != 0 ||
        validate_operand(qweight, qweight_bytes, extmem_base, extmem_size) != 0 ||
        validate_operand(qzeros, qzeros_bytes, extmem_base, extmem_size) != 0 ||
        validate_operand(scales, scales_bytes, extmem_base, extmem_size) != 0 ||
        (g_idx != NULL &&
         validate_operand(g_idx, g_idx_bytes, extmem_base, extmem_size) != 0) ||
        validate_region(scratch_address, scratch_size, extmem_base,
                        extmem_size) != 0) {
        return -1;
    }

    uint32_t input_tile_columns = input_columns;
    if (input_tile_columns > 1023) {
        const uint32_t divisor =
            greatest_common_divisor(8, parameters->quantization_group_size);
        const uint64_t alignment =
            (uint64_t)8 * parameters->quantization_group_size / divisor;
        if (alignment > 1023) {
            errno = ENOTSUP;
            return -1;
        }
        input_tile_columns =
            (uint32_t)(1023 / alignment) * (uint32_t)alignment;
    }
    const uint32_t input_tile_count =
        ceil_div(input_columns, input_tile_columns);
    const uint64_t bytes_per_column =
        ((uint64_t)input_tile_columns +
         (input_tile_count > 1 ? UINT64_C(1) : UINT64_C(0))) *
        4;
    uint32_t tile_columns = (uint32_t)(scratch_size / bytes_per_column);
    if (tile_columns > output_columns) {
        tile_columns = output_columns;
    }
    if (output_columns > 8) {
        tile_columns &= ~UINT32_C(7);
    }
    if (tile_columns == 0 || (output_columns > 8 && tile_columns < 8)) {
        errno = ENOSPC;
        return -1;
    }

    memset(plan, 0, sizeof(*plan));
    plan->rows = rows;
    plan->input_columns = input_columns;
    plan->output_columns = output_columns;
    plan->group_count = groups;
    plan->group_size = parameters->quantization_group_size;
    plan->scale_element_bytes = scale_bytes;
    plan->tile_columns = tile_columns;
    plan->tile_count = ceil_div(output_columns, tile_columns);
    plan->input_tile_columns = input_tile_columns;
    plan->input_tile_count = input_tile_count;
    plan->scratch_address = scratch_address;
    plan->scratch_size = scratch_size;
    plan->partial_address =
        scratch_address + input_tile_columns * tile_columns * 4;
    plan->partial_size = input_tile_count > 1 ? tile_columns * 4 : 0;
    plan->has_g_idx = g_idx != NULL;
    return 0;
}

int
opennpux_npu_gptq_get_tile(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_gptq_tile_plan *plan, uint32_t tile_index,
    struct opennpux_npu_gptq_tile *tile)
{
    return opennpux_npu_gptq_get_tile_2d(request, plan, tile_index, 0, tile);
}

int
opennpux_npu_gptq_get_tile_2d(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_gptq_tile_plan *plan, uint32_t tile_index,
    uint32_t input_tile_index, struct opennpux_npu_gptq_tile *tile)
{
    if (request == NULL || plan == NULL || tile == NULL ||
        tile_index >= plan->tile_count ||
        input_tile_index >= plan->input_tile_count ||
        plan->tile_columns == 0 || plan->input_tile_columns == 0) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_functional_operand *output =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const struct opennpux_npu_functional_operand *qweight =
        find_operand(request, OPENNPUX_NPU_OPERAND_QWEIGHT);
    const struct opennpux_npu_functional_operand *qzeros =
        find_operand(request, OPENNPUX_NPU_OPERAND_QZEROS);
    const struct opennpux_npu_functional_operand *scales =
        find_operand(request, OPENNPUX_NPU_OPERAND_SCALES);
    const struct opennpux_npu_functional_operand *g_idx =
        find_operand(request, OPENNPUX_NPU_OPERAND_G_IDX);
    if (output == NULL || qweight == NULL || qzeros == NULL || scales == NULL ||
        (plan->has_g_idx != 0 && g_idx == NULL)) {
        errno = EINVAL;
        return -1;
    }

    const uint32_t column_base = tile_index * plan->tile_columns;
    const uint32_t column_count =
        plan->output_columns - column_base < plan->tile_columns
            ? plan->output_columns - column_base
            : plan->tile_columns;
    const uint32_t input_base = input_tile_index * plan->input_tile_columns;
    const uint32_t input_count =
        plan->input_columns - input_base < plan->input_tile_columns
            ? plan->input_columns - input_base
            : plan->input_tile_columns;
    const uint32_t weight_rows = ceil_div(input_count, 8);
    const uint32_t zero_columns = ceil_div(plan->output_columns, 8);
    const uint64_t dequantized_bytes =
        (uint64_t)input_count * column_count * 4;
    if (dequantized_bytes > plan->scratch_size) {
        errno = ENOSPC;
        return -1;
    }

    memset(tile, 0, sizeof(*tile));
    tile->column_base = column_base;
    tile->column_count = column_count;
    tile->input_base = input_base;
    tile->input_count = input_count;
    tile->group_base = input_base / plan->group_size;
    tile->group_count = plan->group_count;
    tile->dequantized_address = plan->scratch_address;
    tile->dequantized_bytes = (uint32_t)dequantized_bytes;
    tile->partial_address = plan->partial_address;
    tile->partial_bytes = plan->partial_size;
    tile->qweight = (struct opennpux_npu_gptq_component_view){
        qweight->address + (input_base / 8) * plan->output_columns * 4 +
            column_base * 4,
        weight_rows,
        plan->output_columns * 4,
        column_count * 4,
    };
    tile->qzeros = (struct opennpux_npu_gptq_component_view){
        qzeros->address + (column_base / 8) * 4,
        plan->group_count,
        zero_columns * 4,
        ceil_div(column_count, 8) * 4,
    };
    tile->scales = (struct opennpux_npu_gptq_component_view){
        scales->address + column_base * plan->scale_element_bytes,
        plan->group_count,
        plan->output_columns * plan->scale_element_bytes,
        column_count * plan->scale_element_bytes,
    };
    if (plan->has_g_idx != 0) {
        tile->g_idx = (struct opennpux_npu_gptq_component_view){
            g_idx->address + input_base * 4, 1, input_count * 4,
            input_count * 4,
        };
    }
    tile->output = (struct opennpux_npu_gptq_component_view){
        output->address + column_base * 4,
        plan->rows,
        plan->output_columns * 4,
        column_count * 4,
    };
    return 0;
}
