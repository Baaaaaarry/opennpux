#include "opennpux/npu_gptq_reference.h"

#include <errno.h>
#include <math.h>
#include <string.h>

struct operand {
    const void *data;
    uint64_t bytes;
};

static uint32_t
ceil_div(uint32_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

static int
multiply_fits(uint64_t lhs, uint64_t rhs)
{
    return rhs == 0 || lhs <= UINT64_MAX / rhs;
}

static int
product(uint64_t first, uint64_t second, uint64_t third, uint64_t *result)
{
    if (!multiply_fits(first, second) ||
        !multiply_fits(first * second, third)) {
        return -1;
    }
    *result = first * second * third;
    return 0;
}

static uint32_t
scale_element_size(uint32_t data_type)
{
    if (data_type == CORAL_GPTQ_SCALE_FLOAT16 ||
        data_type == CORAL_GPTQ_SCALE_BFLOAT16) {
        return (uint32_t)sizeof(uint16_t);
    }
    return data_type == CORAL_GPTQ_SCALE_FLOAT32 ? (uint32_t)sizeof(float) : 0;
}

static float
bits_to_float(uint32_t bits)
{
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float
float16_to_float(uint16_t value)
{
    const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
    const uint32_t exponent = (uint32_t)((value >> 10) & 0x1fu);
    uint32_t mantissa = (uint32_t)(value & 0x3ffu);
    if (exponent == 0) {
        if (mantissa == 0) {
            return bits_to_float(sign);
        }
        uint32_t normalized_exponent = 113;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --normalized_exponent;
        }
        return bits_to_float(sign | (normalized_exponent << 23) |
                             ((mantissa & 0x3ffu) << 13));
    }
    if (exponent == 0x1f) {
        return bits_to_float(sign | UINT32_C(0x7f800000) | (mantissa << 13));
    }
    return bits_to_float(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

static float
read_scale(const void *scales, size_t index, uint32_t data_type)
{
    if (data_type == CORAL_GPTQ_SCALE_FLOAT16) {
        return float16_to_float(((const uint16_t *)scales)[index]);
    }
    if (data_type == CORAL_GPTQ_SCALE_BFLOAT16) {
        return bits_to_float(
            (uint32_t)((const uint16_t *)scales)[index] << 16);
    }
    return ((const float *)scales)[index];
}

/*
 * Map one device address onto the staged image. Operand pointers are cast to
 * their element type, so the resolved offset must carry the element alignment.
 */
static int
resolve(const uint8_t *image, size_t image_size, uint64_t device_base,
        uint32_t address, uint64_t bytes, uint32_t alignment,
        struct operand *operand)
{
    if (address == 0 || bytes == 0 || address < device_base) {
        errno = EINVAL;
        return -1;
    }
    const uint64_t offset = (uint64_t)address - device_base;
    if (offset % alignment != 0 || offset > image_size ||
        bytes > (uint64_t)image_size - offset) {
        errno = ERANGE;
        return -1;
    }
    operand->data = image + offset;
    operand->bytes = bytes;
    return 0;
}

uint32_t
opennpux_npu_gptq_reference_checksum(const float *values, size_t count)
{
    uint32_t checksum = UINT32_C(2166136261);
    const uint8_t *bytes = (const uint8_t *)values;
    for (size_t index = 0; index < count * sizeof(float); ++index) {
        checksum ^= bytes[index];
        checksum *= UINT32_C(16777619);
    }
    return checksum;
}

int
opennpux_npu_gptq_reference_run(
    const void *image, size_t image_size, uint64_t device_base,
    float *output, size_t output_floats,
    struct opennpux_npu_gptq_reference_result *result)
{
    if (image == NULL || result == NULL ||
        image_size < sizeof(struct coral_gptq_matmul_request) ||
        (uintptr_t)image % sizeof(uint32_t) != 0) {
        errno = EINVAL;
        return -1;
    }
    memset(result, 0, sizeof(*result));

    struct coral_gptq_matmul_request request;
    memcpy(&request, image, sizeof(request));
    if (request.magic != CORAL_GPTQ_MATMUL_MAGIC ||
        request.version != CORAL_GPTQ_MATMUL_VERSION ||
        request.struct_size != sizeof(request) ||
        (request.state != CORAL_GPTQ_MATMUL_PENDING &&
         request.state != CORAL_GPTQ_MATMUL_COMPLETE) ||
        request.rows == 0 || request.input_columns == 0 ||
        request.output_columns == 0 || request.group_size == 0 ||
        request.zero_bias > 15 ||
        scale_element_size(request.scale_data_type) == 0) {
        errno = EPROTO;
        return -1;
    }

    const uint32_t weight_rows = ceil_div(request.input_columns, 8);
    const uint32_t groups = ceil_div(request.input_columns, request.group_size);
    const uint32_t zero_columns = ceil_div(request.output_columns, 8);
    const uint32_t scale_bytes = scale_element_size(request.scale_data_type);
    uint64_t input_bytes = 0;
    uint64_t qweight_bytes = 0;
    uint64_t qzeros_bytes = 0;
    uint64_t scales_bytes = 0;
    uint64_t g_idx_bytes = 0;
    uint64_t output_bytes = 0;
    uint64_t multiply_accumulates = 0;
    if (product(request.rows, request.input_columns, sizeof(float),
                &input_bytes) != 0 ||
        product(weight_rows, request.output_columns, sizeof(uint32_t),
                &qweight_bytes) != 0 ||
        product(groups, zero_columns, sizeof(uint32_t), &qzeros_bytes) != 0 ||
        product(groups, request.output_columns, scale_bytes,
                &scales_bytes) != 0 ||
        product(request.input_columns, sizeof(uint32_t), 1,
                &g_idx_bytes) != 0 ||
        product(request.rows, request.output_columns, sizeof(float),
                &output_bytes) != 0 ||
        product(request.rows, request.input_columns, request.output_columns,
                &multiply_accumulates) != 0 ||
        !multiply_fits(multiply_accumulates, 2)) {
        errno = EOVERFLOW;
        return -1;
    }

    const uint8_t *bytes = (const uint8_t *)image;
    struct operand input = {NULL, 0};
    struct operand qweight = {NULL, 0};
    struct operand qzeros = {NULL, 0};
    struct operand scales = {NULL, 0};
    struct operand g_idx = {NULL, 0};
    struct operand output_region = {NULL, 0};
    if (resolve(bytes, image_size, device_base, request.input_address,
                input_bytes, (uint32_t)sizeof(float), &input) != 0 ||
        resolve(bytes, image_size, device_base, request.qweight_address,
                qweight_bytes, (uint32_t)sizeof(uint32_t), &qweight) != 0 ||
        resolve(bytes, image_size, device_base, request.qzeros_address,
                qzeros_bytes, (uint32_t)sizeof(uint32_t), &qzeros) != 0 ||
        resolve(bytes, image_size, device_base, request.scales_address,
                scales_bytes, scale_bytes, &scales) != 0 ||
        resolve(bytes, image_size, device_base, request.output_address,
                output_bytes, (uint32_t)sizeof(float), &output_region) != 0 ||
        (request.g_idx_address != 0 &&
         resolve(bytes, image_size, device_base, request.g_idx_address,
                 g_idx_bytes, (uint32_t)sizeof(uint32_t), &g_idx) != 0)) {
        return -1;
    }
    if (output != NULL &&
        output_floats < (size_t)request.rows * request.output_columns) {
        errno = ENOSPC;
        return -1;
    }

    const float *input_data = (const float *)input.data;
    const uint32_t *qweight_data = (const uint32_t *)qweight.data;
    const uint32_t *qzeros_data = (const uint32_t *)qzeros.data;
    const uint32_t *g_idx_data = (const uint32_t *)g_idx.data;
    uint32_t checksum = UINT32_C(2166136261);
    for (uint32_t row = 0; row < request.rows; ++row) {
        for (uint32_t column = 0; column < request.output_columns; ++column) {
            float accumulator = 0.0f;
            for (uint32_t k = 0; k < request.input_columns; ++k) {
                const uint32_t group = g_idx_data == NULL ?
                    k / request.group_size : g_idx_data[k];
                if (group >= groups) {
                    errno = EPROTO;
                    return -1;
                }
                const uint32_t packed_weight = qweight_data[
                    (size_t)(k / 8) * request.output_columns + column];
                const uint32_t quantized =
                    (packed_weight >> (4 * (k % 8))) & 0xfu;
                const uint32_t packed_zero = qzeros_data[
                    (size_t)group * zero_columns + column / 8];
                const uint32_t stored_zero =
                    (packed_zero >> (4 * (column % 8))) & 0xfu;
                const uint32_t zero = stored_zero + request.zero_bias > 15 ?
                    15 : stored_zero + request.zero_bias;
                const float scale = read_scale(
                    scales.data, (size_t)group * request.output_columns +
                    column, request.scale_data_type);
                if (!isfinite(scale)) {
                    errno = EPROTO;
                    return -1;
                }
                const float weight =
                    ((int32_t)quantized - (int32_t)zero) * scale;
                accumulator +=
                    input_data[(size_t)row * request.input_columns + k] *
                    weight;
            }
            if (output != NULL) {
                output[(size_t)row * request.output_columns + column] =
                    accumulator;
            }
            const uint8_t *value = (const uint8_t *)&accumulator;
            for (size_t index = 0; index < sizeof(accumulator); ++index) {
                checksum ^= value[index];
                checksum *= UINT32_C(16777619);
            }
        }
    }

    result->rows = request.rows;
    result->input_columns = request.input_columns;
    result->output_columns = request.output_columns;
    result->group_size = request.group_size;
    result->zero_bias = request.zero_bias;
    result->scale_data_type = request.scale_data_type;
    result->has_g_idx = g_idx_data != NULL;
    result->output_checksum = checksum;
    result->operations = multiply_accumulates * 2;
    result->bytes_read = input_bytes + qweight_bytes + qzeros_bytes +
        scales_bytes + (g_idx_data != NULL ? g_idx_bytes : 0);
    result->bytes_written = output_bytes;
    result->modeled_cycles = result->operations / 2 +
        (result->bytes_read + result->bytes_written + 15) / 16;
    return 0;
}
