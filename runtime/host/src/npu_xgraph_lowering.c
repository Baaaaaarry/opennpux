#include "opennpux/npu_xgraph_lowering.h"

#include <errno.h>
#include <string.h>

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
has_gptq_operands(const struct opennpux_npu_functional_request *request)
{
    return find_operand(request, OPENNPUX_NPU_OPERAND_QWEIGHT) != NULL ||
        find_operand(request, OPENNPUX_NPU_OPERAND_QZEROS) != NULL ||
        find_operand(request, OPENNPUX_NPU_OPERAND_SCALES) != NULL;
}

static int
has_gptq_expert_operands(
    const struct opennpux_npu_functional_request *request)
{
    return find_operand(request, OPENNPUX_NPU_OPERAND_GATE_QWEIGHT) != NULL ||
        find_operand(request, OPENNPUX_NPU_OPERAND_UP_QWEIGHT) != NULL ||
        find_operand(request, OPENNPUX_NPU_OPERAND_DOWN_QWEIGHT) != NULL;
}

static int
has_dense_multi_projection_operands(
    const struct opennpux_npu_functional_request *request)
{
    return find_operand(request, OPENNPUX_NPU_OPERAND_INPUT) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_LINEAR_QKV_WEIGHT) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_LINEAR_ALPHA_WEIGHT) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_LINEAR_BETA_WEIGHT) != NULL;
}

static int
has_attention_projection_operands(
    const struct opennpux_npu_functional_request *request)
{
    return find_operand(request, OPENNPUX_NPU_OPERAND_INPUT) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_ATTENTION_Q_WEIGHT) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_ATTENTION_K_WEIGHT) != NULL &&
        find_operand(request, OPENNPUX_NPU_OPERAND_ATTENTION_V_WEIGHT) != NULL &&
        find_operand(request,
                     OPENNPUX_NPU_OPERAND_ATTENTION_Q_NORM_WEIGHT) != NULL &&
        find_operand(request,
                     OPENNPUX_NPU_OPERAND_ATTENTION_K_NORM_WEIGHT) != NULL;
}

static int
operand_offset(const struct opennpux_npu_functional_operand *operand,
               uint32_t extmem_base, uint32_t extmem_size, uint32_t *offset)
{
    if (operand == NULL || offset == NULL || operand->address < extmem_base) {
        errno = EINVAL;
        return -1;
    }
    const uint64_t relative = (uint64_t)operand->address - extmem_base;
    if (relative > extmem_size || operand->byte_size > extmem_size - relative) {
        errno = ERANGE;
        return -1;
    }
    *offset = (uint32_t)relative;
    return 0;
}

static int
set_operands(struct opennpux_xgraph_command *command,
             const struct opennpux_npu_functional_operand *destination,
             const struct opennpux_npu_functional_operand *source0,
             const struct opennpux_npu_functional_operand *source1,
             uint32_t extmem_base, uint32_t extmem_size)
{
    if (operand_offset(destination, extmem_base, extmem_size,
                       &command->destination_offset) != 0 ||
        operand_offset(source0, extmem_base, extmem_size,
                       &command->source0_offset) != 0) {
        return -1;
    }
    if (source1 != NULL &&
        operand_offset(source1, extmem_base, extmem_size,
                       &command->source1_offset) != 0) {
        return -1;
    }
    return 0;
}

static int
validate_request(const struct opennpux_npu_functional_request *request,
                 const struct opennpux_npu_operator_parameters *parameters,
                 uint32_t extmem_size)
{
    if (request == NULL || parameters == NULL || extmem_size == 0 ||
        request->magic != OPENNPUX_NPU_FUNCTIONAL_MAGIC ||
        request->version != OPENNPUX_NPU_FUNCTIONAL_VERSION ||
        request->struct_size != sizeof(*request) ||
        request->operand_count > OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS ||
        request->rows == 0 || request->features == 0 ||
        parameters->magic != OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC ||
        parameters->version != OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION ||
        parameters->struct_size != sizeof(*parameters) ||
        parameters->opcode != request->opcode) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}

static int
address_offset(uint32_t address, uint32_t bytes, uint32_t extmem_base,
               uint32_t extmem_size, uint32_t *offset)
{
    const struct opennpux_npu_functional_operand operand = {
        0, address, bytes, 0};
    return operand_offset(&operand, extmem_base, extmem_size, offset);
}

static int
address_offset64(uint64_t address, uint64_t bytes, uint32_t extmem_base,
                 uint32_t extmem_size, uint32_t *offset)
{
    if (address > UINT32_MAX || bytes == 0 || bytes > UINT32_MAX) {
        errno = EOVERFLOW;
        return -1;
    }
    return address_offset((uint32_t)address, (uint32_t)bytes, extmem_base,
                          extmem_size, offset);
}

static uint32_t
xopennpux_scale_data_type(uint32_t data_type)
{
    switch (data_type) {
    case OPENNPUX_NPU_DTYPE_FLOAT16:
        return 0;
    case OPENNPUX_NPU_DTYPE_BFLOAT16:
        return 1;
    case OPENNPUX_NPU_DTYPE_FLOAT32:
        return 2;
    default:
        return UINT32_MAX;
    }
}

#define OPENNPUX_TMMA_TILE_LIMIT UINT32_C(1023)

static int
lower_dense_matmul_operands_strided(
    const struct opennpux_npu_functional_operand *input,
    const struct opennpux_npu_functional_operand *weight,
    const struct opennpux_npu_functional_operand *output,
    uint32_t rows, uint32_t input_features, uint32_t output_features,
    uint32_t input_row_stride, uint32_t weight_row_stride,
    uint32_t output_row_stride,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (command_count != NULL) {
        *command_count = 0;
    }
    const uint64_t minimum_input_stride =
        (uint64_t)input_features * sizeof(float);
    const uint64_t minimum_weight_stride = minimum_input_stride;
    const uint64_t minimum_output_stride =
        (uint64_t)output_features * sizeof(float);
    const uint64_t input_stride =
        input_row_stride == 0 ? minimum_input_stride : input_row_stride;
    const uint64_t weight_stride =
        weight_row_stride == 0 ? minimum_weight_stride : weight_row_stride;
    const uint64_t output_stride =
        output_row_stride == 0 ? minimum_output_stride : output_row_stride;
    const uint64_t input_bytes =
        (uint64_t)(rows - 1) * input_stride + minimum_input_stride;
    const uint64_t weight_bytes = (uint64_t)(output_features - 1) *
            weight_stride +
        minimum_weight_stride;
    const uint64_t output_bytes =
        (uint64_t)(rows - 1) * output_stride + minimum_output_stride;
    const uint64_t m_tiles =
        (rows + OPENNPUX_TMMA_TILE_LIMIT - 1) / OPENNPUX_TMMA_TILE_LIMIT;
    const uint64_t n_tiles =
        (output_features + OPENNPUX_TMMA_TILE_LIMIT - 1) /
        OPENNPUX_TMMA_TILE_LIMIT;
    const uint64_t k_tiles =
        (input_features + OPENNPUX_TMMA_TILE_LIMIT - 1) /
        OPENNPUX_TMMA_TILE_LIMIT;
    const uint64_t required_commands = m_tiles * n_tiles * k_tiles;
    if (input == NULL || weight == NULL || output == NULL || commands == NULL ||
        command_count == NULL || rows == 0 || input_features == 0 ||
        output_features == 0 || input_stride < minimum_input_stride ||
        weight_stride < minimum_weight_stride ||
        output_stride < minimum_output_stride || input_stride > UINT32_MAX ||
        weight_stride > UINT32_MAX || output_stride > UINT32_MAX ||
        input_bytes > UINT32_MAX ||
        weight_bytes > UINT32_MAX || output_bytes > UINT32_MAX ||
        input->byte_size < input_bytes || weight->byte_size < weight_bytes ||
        output->byte_size < output_bytes || required_commands == 0 ||
        required_commands > command_capacity ||
        required_commands > UINT32_MAX - first_command_id) {
        errno = required_commands > command_capacity ? ENOSPC : EINVAL;
        return -1;
    }

    uint32_t emitted = 0;
    for (uint32_t m_base = 0; m_base < rows;
         m_base += OPENNPUX_TMMA_TILE_LIMIT) {
        const uint32_t m = rows - m_base < OPENNPUX_TMMA_TILE_LIMIT
                               ? rows - m_base
                               : OPENNPUX_TMMA_TILE_LIMIT;
        for (uint32_t n_base = 0; n_base < output_features;
             n_base += OPENNPUX_TMMA_TILE_LIMIT) {
            const uint32_t n =
                output_features - n_base < OPENNPUX_TMMA_TILE_LIMIT
                    ? output_features - n_base
                    : OPENNPUX_TMMA_TILE_LIMIT;
            for (uint32_t k_base = 0; k_base < input_features;
                 k_base += OPENNPUX_TMMA_TILE_LIMIT) {
                const uint32_t k =
                    input_features - k_base < OPENNPUX_TMMA_TILE_LIMIT
                        ? input_features - k_base
                        : OPENNPUX_TMMA_TILE_LIMIT;
                struct opennpux_xgraph_command *command = &commands[emitted];
                memset(command, 0, sizeof(*command));
                command->command_id = first_command_id + emitted;
                command->opcode = OPENNPUX_XGRAPH_OP_TMMA;
                command->flags = OPENNPUX_XGRAPH_TMMA_TRANSPOSE_RHS |
                    (k_base == 0 ? 0 : OPENNPUX_XGRAPH_TMMA_ACCUMULATE);
                command->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
                command->dim0 = m;
                command->dim1 = n;
                command->dim2 = k;
                command->reserved[0] = (uint32_t)input_stride;
                command->reserved[1] = (uint32_t)weight_stride;
                command->reserved[2] = (uint32_t)output_stride;
                const uint64_t input_address =
                    (uint64_t)input->address +
                    ((uint64_t)m_base * input_features + k_base) *
                        sizeof(float);
                const uint64_t weight_address =
                    (uint64_t)weight->address +
                    ((uint64_t)n_base * input_features + k_base) *
                        sizeof(float);
                const uint64_t output_address =
                    (uint64_t)output->address +
                    ((uint64_t)m_base * output_features + n_base) *
                        sizeof(float);
                if (address_offset64(
                        input_address,
                        (uint64_t)(m - 1) * command->reserved[0] +
                            k * sizeof(float),
                        extmem_base, extmem_size,
                        &command->source0_offset) != 0 ||
                    address_offset64(
                        weight_address,
                        (uint64_t)(n - 1) * command->reserved[1] +
                            k * sizeof(float),
                        extmem_base, extmem_size,
                        &command->source1_offset) != 0 ||
                    address_offset64(
                        output_address,
                        (uint64_t)(m - 1) * command->reserved[2] +
                            n * sizeof(float),
                        extmem_base, extmem_size,
                        &command->destination_offset) != 0) {
                    return -1;
                }
                ++emitted;
            }
        }
    }
    *command_count = emitted;
    return 0;
}

static int
lower_dense_matmul_operands(
    const struct opennpux_npu_functional_operand *input,
    const struct opennpux_npu_functional_operand *weight,
    const struct opennpux_npu_functional_operand *output,
    uint32_t rows, uint32_t input_features, uint32_t output_features,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    return lower_dense_matmul_operands_strided(
        input, weight, output, rows, input_features, output_features, 0, 0, 0,
        extmem_base, extmem_size, first_command_id, commands,
        command_capacity, command_count);
}

int
opennpux_npu_xgraph_lower_dense_matmul(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (validate_request(request, parameters, extmem_size) != 0 ||
        request->opcode != OPENNPUX_NPU_OP_MATMUL ||
        has_gptq_operands(request)) {
        errno = EINVAL;
        return -1;
    }
    return lower_dense_matmul_operands(
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT),
        find_operand(request, OPENNPUX_NPU_OPERAND_WEIGHT),
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT), request->rows,
        parameters->input_features, parameters->output_features, extmem_base,
        extmem_size, first_command_id, commands, command_capacity,
        command_count);
}

int
opennpux_npu_xgraph_lower_dense_multi_projection(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (command_count != NULL) {
        *command_count = 0;
    }
    if (validate_request(request, parameters, extmem_size) != 0 ||
        request->opcode != OPENNPUX_NPU_OP_MATMUL || commands == NULL ||
        command_count == NULL || parameters->input_features == 0 ||
        !has_dense_multi_projection_operands(request)) {
        errno = EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const uint32_t weight_roles[] = {
        OPENNPUX_NPU_OPERAND_LINEAR_QKV_WEIGHT,
        OPENNPUX_NPU_OPERAND_LINEAR_ALPHA_WEIGHT,
        OPENNPUX_NPU_OPERAND_LINEAR_BETA_WEIGHT,
    };
    const uint32_t output_roles[] = {
        OPENNPUX_NPU_OPERAND_OUTPUT,
        OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY,
        OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY,
    };
    const uint64_t weight_row_bytes =
        (uint64_t)parameters->input_features * sizeof(float);
    uint32_t emitted = 0;
    for (uint32_t index = 0; index < 3; ++index) {
        const struct opennpux_npu_functional_operand *weight =
            find_operand(request, weight_roles[index]);
        const struct opennpux_npu_functional_operand *output =
            find_operand(request, output_roles[index]);
        if (weight_row_bytes == 0 || weight_row_bytes > UINT32_MAX ||
            weight->byte_size == 0 ||
            weight->byte_size % weight_row_bytes != 0) {
            errno = EINVAL;
            return -1;
        }
        const uint64_t output_features64 =
            weight->byte_size / weight_row_bytes;
        if (output_features64 == 0 || output_features64 > UINT32_MAX) {
            errno = EOVERFLOW;
            return -1;
        }
        uint32_t produced = 0;
        if (lower_dense_matmul_operands(
                input, weight, output, request->rows,
                parameters->input_features, (uint32_t)output_features64,
                extmem_base, extmem_size, first_command_id + emitted,
                commands + emitted, command_capacity - emitted,
                &produced) != 0) {
            return -1;
        }
        emitted += produced;
    }
    *command_count = emitted;
    return 0;
}

int
opennpux_npu_xgraph_lower_attention_projection(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (command_count != NULL) {
        *command_count = 0;
    }
    if (validate_request(request, parameters, extmem_size) != 0 ||
        request->opcode != OPENNPUX_NPU_OP_MATMUL || commands == NULL ||
        command_count == NULL || !has_attention_projection_operands(request) ||
        parameters->input_features == 0 || request->heads == 0 ||
        request->kv_heads == 0 || request->head_dim == 0) {
        errno = EINVAL;
        return -1;
    }
    const uint64_t query_features64 =
        (uint64_t)request->heads * request->head_dim;
    const uint64_t key_features64 =
        (uint64_t)request->kv_heads * request->head_dim;
    if (query_features64 > UINT32_MAX || key_features64 > UINT32_MAX ||
        request->features != query_features64) {
        errno = EINVAL;
        return -1;
    }
    const uint32_t query_features = (uint32_t)query_features64;
    const uint32_t key_features = (uint32_t)key_features64;
    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *query =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const struct opennpux_npu_functional_operand *key =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY);
    const struct opennpux_npu_functional_operand *value =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY);
    const struct opennpux_npu_functional_operand *gate =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_QUATERNARY);
    const struct opennpux_npu_functional_operand *q_weight =
        find_operand(request, OPENNPUX_NPU_OPERAND_ATTENTION_Q_WEIGHT);
    const struct opennpux_npu_functional_operand *k_weight =
        find_operand(request, OPENNPUX_NPU_OPERAND_ATTENTION_K_WEIGHT);
    const struct opennpux_npu_functional_operand *v_weight =
        find_operand(request, OPENNPUX_NPU_OPERAND_ATTENTION_V_WEIGHT);
    const struct opennpux_npu_functional_operand *q_norm = find_operand(
        request, OPENNPUX_NPU_OPERAND_ATTENTION_Q_NORM_WEIGHT);
    const struct opennpux_npu_functional_operand *k_norm = find_operand(
        request, OPENNPUX_NPU_OPERAND_ATTENTION_K_NORM_WEIGHT);
    const uint64_t input_stride64 =
        (uint64_t)parameters->input_features * sizeof(float);
    const uint64_t query_stride64 =
        (uint64_t)query_features * sizeof(float);
    const uint64_t key_stride64 = (uint64_t)key_features * sizeof(float);
    const uint64_t q_weight_features =
        q_weight->byte_size / sizeof(float) / parameters->input_features;
    const int gated_query = gate != NULL;
    if (input_stride64 > UINT32_MAX || query_stride64 > UINT32_MAX ||
        key_stride64 > UINT32_MAX ||
        q_weight->byte_size % sizeof(float) != 0 ||
        q_weight->byte_size / sizeof(float) % parameters->input_features != 0 ||
        q_weight_features != query_features * (gated_query ? 2u : 1u) ||
        k_weight->byte_size != input_stride64 * key_features ||
        v_weight->byte_size != input_stride64 * key_features ||
        q_norm->byte_size != request->head_dim * sizeof(float) ||
        k_norm->byte_size != request->head_dim * sizeof(float)) {
        errno = EINVAL;
        return -1;
    }

    uint32_t emitted = 0;
    uint32_t produced = 0;
    for (uint32_t head = 0; head < request->heads; ++head) {
        const uint32_t projection_count = gated_query ? 2 : 1;
        for (uint32_t projection = 0; projection < projection_count;
             ++projection) {
            const uint64_t weight_offset =
                ((uint64_t)head * projection_count + projection) *
                request->head_dim * input_stride64;
            const uint64_t output_offset =
                (uint64_t)head * request->head_dim * sizeof(float);
            struct opennpux_npu_functional_operand weight_view = *q_weight;
            struct opennpux_npu_functional_operand output_view =
                projection == 0 ? *query : *gate;
            weight_view.address += (uint32_t)weight_offset;
            weight_view.byte_size -= (uint32_t)weight_offset;
            output_view.address += (uint32_t)output_offset;
            output_view.byte_size -= (uint32_t)output_offset;
            if (lower_dense_matmul_operands_strided(
                    input, &weight_view, &output_view, request->rows,
                    parameters->input_features, request->head_dim,
                    (uint32_t)input_stride64, (uint32_t)input_stride64,
                    (uint32_t)query_stride64, extmem_base, extmem_size,
                    first_command_id + emitted, commands + emitted,
                    command_capacity - emitted, &produced) != 0) {
                return -1;
            }
            emitted += produced;
        }
    }
#define LOWER_ATTENTION_DENSE(WEIGHT, OUTPUT)                                 \
    do {                                                                       \
        if (lower_dense_matmul_operands(                                       \
                input, (WEIGHT), (OUTPUT), request->rows,                      \
                parameters->input_features, key_features, extmem_base,         \
                extmem_size, first_command_id + emitted, commands + emitted,   \
                command_capacity - emitted, &produced) != 0) {                 \
            return -1;                                                         \
        }                                                                      \
        emitted += produced;                                                   \
    } while (0)
    LOWER_ATTENTION_DENSE(k_weight, key);
    LOWER_ATTENTION_DENSE(v_weight, value);
#undef LOWER_ATTENTION_DENSE

    const uint64_t norm_commands =
        (uint64_t)request->rows * (request->heads + request->kv_heads);
    if (norm_commands > command_capacity - emitted) {
        errno = ENOSPC;
        return -1;
    }
    const struct {
        const struct opennpux_npu_functional_operand *tensor;
        const struct opennpux_npu_functional_operand *weight;
        uint32_t heads;
        uint32_t row_stride;
    } norm_groups[] = {
        {query, q_norm, request->heads, (uint32_t)query_stride64},
        {key, k_norm, request->kv_heads, (uint32_t)key_stride64},
    };
    for (uint32_t group = 0; group < 2; ++group) {
        for (uint32_t row = 0; row < request->rows; ++row) {
            for (uint32_t head = 0; head < norm_groups[group].heads; ++head) {
                const uint64_t tensor_offset =
                    (uint64_t)row * norm_groups[group].row_stride +
                    (uint64_t)head * request->head_dim * sizeof(float);
                struct opennpux_npu_functional_operand tensor_view =
                    *norm_groups[group].tensor;
                tensor_view.address += (uint32_t)tensor_offset;
                tensor_view.byte_size -= (uint32_t)tensor_offset;
                struct opennpux_xgraph_command *command = &commands[emitted++];
                memset(command, 0, sizeof(*command));
                command->command_id = first_command_id + emitted - 1;
                command->opcode = OPENNPUX_XGRAPH_OP_TRMSNORM;
                command->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
                command->dim0 = 1;
                command->dim1 = request->head_dim;
                command->dim2 = 1;
                command->flags =
                    ((parameters->flags &
                      OPENNPUX_NPU_PARAMETER_NORM_WEIGHT_OFFSET) != 0
                         ? OPENNPUX_XGRAPH_TRMSNORM_WEIGHT_OFFSET
                         : 0) |
                    ((parameters->flags &
                      OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE) != 0
                         ? OPENNPUX_XGRAPH_TRMSNORM_BFLOAT16_INPUT
                         : 0);
                memcpy(&command->scalar0, &request->epsilon,
                       sizeof(command->scalar0));
                if (set_operands(command, &tensor_view, &tensor_view,
                                 norm_groups[group].weight, extmem_base,
                                 extmem_size) != 0) {
                    return -1;
                }
            }
        }
    }
    *command_count = emitted;
    return 0;
}

int
opennpux_npu_xgraph_lower_gated_normalize(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (command_count != NULL) {
        *command_count = 0;
    }
    if (validate_request(request, parameters, extmem_size) != 0 ||
        request->opcode != OPENNPUX_NPU_OP_NORMALIZE || commands == NULL ||
        command_count == NULL ||
        (parameters->flags & OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET) == 0) {
        errno = EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *projection_input =
        find_operand(request, OPENNPUX_NPU_OPERAND_SECONDARY);
    const struct opennpux_npu_functional_operand *output =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const struct opennpux_npu_functional_operand *gate_weight =
        find_operand(request, OPENNPUX_NPU_OPERAND_LINEAR_GATE_WEIGHT);
    const struct opennpux_npu_functional_operand *norm_weight =
        find_operand(request, OPENNPUX_NPU_OPERAND_LINEAR_NORM_WEIGHT);
    const uint32_t input_features = parameters->input_features;
    const uint32_t output_features = parameters->output_features;
    const uint32_t head_dim =
        norm_weight != NULL &&
                norm_weight->byte_size % sizeof(float) == 0
            ? norm_weight->byte_size / sizeof(float)
            : 0;
    if (input == NULL || projection_input == NULL || output == NULL ||
        gate_weight == NULL || norm_weight == NULL || request->rows == 0 ||
        input_features == 0 || output_features == 0 || head_dim == 0 ||
        output_features % head_dim != 0) {
        errno = EINVAL;
        return -1;
    }

    const uint32_t heads = output_features / head_dim;
    const uint64_t tensor_elements =
        (uint64_t)request->rows * output_features;
    const uint64_t tensor_bytes = tensor_elements * sizeof(float);
    const uint64_t required_scratch = tensor_bytes * 2;
    const uint64_t tail_commands =
        UINT64_C(2) + (uint64_t)request->rows * heads;
    if (tensor_bytes > UINT32_MAX || required_scratch > scratch_size ||
        (uint64_t)scratch_address + required_scratch > UINT32_MAX ||
        tail_commands >= command_capacity) {
        errno = tail_commands >= command_capacity ? ENOSPC : ENOMEM;
        return -1;
    }
    if (input->byte_size < tensor_bytes ||
        projection_input->byte_size <
            (uint64_t)request->rows * input_features * sizeof(float) ||
        output->byte_size < tensor_bytes ||
        gate_weight->byte_size <
            (uint64_t)input_features * output_features * sizeof(float) ||
        norm_weight->byte_size < (uint64_t)head_dim * sizeof(float)) {
        errno = EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand gate = {
        OPENNPUX_NPU_OPERAND_OUTPUT, scratch_address,
        (uint32_t)tensor_bytes, 0,
    };
    const struct opennpux_npu_functional_operand normalized = {
        OPENNPUX_NPU_OPERAND_OUTPUT,
        scratch_address + (uint32_t)tensor_bytes,
        (uint32_t)tensor_bytes, 0,
    };
    uint32_t emitted = 0;
    uint32_t projection_commands = 0;
    if (lower_dense_matmul_operands_strided(
            projection_input, gate_weight, &gate, request->rows,
            input_features, output_features, input_features * sizeof(float),
            input_features * sizeof(float), output_features * sizeof(float),
            extmem_base, extmem_size, first_command_id, commands,
            command_capacity - (uint32_t)tail_commands,
            &projection_commands) != 0) {
        return -1;
    }
    emitted = projection_commands;

    for (uint32_t row = 0; row < request->rows; ++row) {
        for (uint32_t head = 0; head < heads; ++head) {
            const uint64_t element =
                (uint64_t)row * output_features + (uint64_t)head * head_dim;
            const uint64_t byte_offset = element * sizeof(float);
            struct opennpux_npu_functional_operand input_view = *input;
            struct opennpux_npu_functional_operand output_view = normalized;
            input_view.address += (uint32_t)byte_offset;
            input_view.byte_size = head_dim * sizeof(float);
            output_view.address += (uint32_t)byte_offset;
            output_view.byte_size = head_dim * sizeof(float);

            struct opennpux_xgraph_command *norm = &commands[emitted++];
            memset(norm, 0, sizeof(*norm));
            norm->opcode = OPENNPUX_XGRAPH_OP_TRMSNORM;
            norm->command_id = first_command_id + emitted - 1;
            norm->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
            norm->dim0 = 1;
            norm->dim1 = head_dim;
            norm->dim2 = 1;
            norm->flags =
                ((parameters->flags &
                  OPENNPUX_NPU_PARAMETER_NORM_WEIGHT_OFFSET) != 0
                     ? OPENNPUX_XGRAPH_TRMSNORM_WEIGHT_OFFSET
                     : 0) |
                ((parameters->flags &
                  OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE) != 0
                     ? OPENNPUX_XGRAPH_TRMSNORM_BFLOAT16_NORMALIZED
                     : 0);
            memcpy(&norm->scalar0, &request->epsilon, sizeof(norm->scalar0));
            if (set_operands(norm, &output_view, &input_view, norm_weight,
                             extmem_base, extmem_size) != 0) {
                return -1;
            }
        }
    }

    struct opennpux_xgraph_command *silu = &commands[emitted++];
    memset(silu, 0, sizeof(*silu));
    silu->opcode = OPENNPUX_XGRAPH_OP_TSILU;
    silu->command_id = first_command_id + emitted - 1;
    silu->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    silu->dim0 = request->rows;
    silu->dim1 = output_features;
    silu->dim2 = 1;
    silu->flags =
        (parameters->flags & OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE) != 0
            ? OPENNPUX_XGRAPH_TSILU_BFLOAT16_INPUT
            : 0;
    if (set_operands(silu, &gate, &gate, NULL, extmem_base, extmem_size) != 0) {
        return -1;
    }

    struct opennpux_xgraph_command *multiply = &commands[emitted++];
    memset(multiply, 0, sizeof(*multiply));
    multiply->opcode = OPENNPUX_XGRAPH_OP_TMUL;
    multiply->command_id = first_command_id + emitted - 1;
    multiply->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    multiply->dim0 = request->rows;
    multiply->dim1 = output_features;
    multiply->dim2 = 1;
    if (set_operands(multiply, output, &normalized, &gate,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }
    *command_count = emitted;
    return 0;
}

int
opennpux_npu_xgraph_lower_shared_expert(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (command_count != NULL) {
        *command_count = 0;
    }
    if (validate_request(request, parameters, extmem_size) != 0 ||
        request->opcode != OPENNPUX_NPU_OP_EXPERT || commands == NULL ||
        command_count == NULL || parameters->input_features == 0 ||
        parameters->intermediate_features == 0 ||
        parameters->output_features == 0) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *output =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const struct opennpux_npu_functional_operand *gate =
        find_operand(request, OPENNPUX_NPU_OPERAND_SHARED_GATE_WEIGHT);
    const struct opennpux_npu_functional_operand *up =
        find_operand(request, OPENNPUX_NPU_OPERAND_SHARED_UP_WEIGHT);
    const struct opennpux_npu_functional_operand *down =
        find_operand(request, OPENNPUX_NPU_OPERAND_SHARED_DOWN_WEIGHT);
    const struct opennpux_npu_functional_operand *router =
        find_operand(request, OPENNPUX_NPU_OPERAND_SHARED_ROUTER_WEIGHT);
    const uint64_t activation_bytes =
        (uint64_t)request->rows * parameters->intermediate_features *
        sizeof(float);
    const uint64_t router_bytes = (uint64_t)request->rows * sizeof(float);
    const uint64_t required_scratch = activation_bytes * 2 + router_bytes;
    if (input == NULL || output == NULL || gate == NULL || up == NULL ||
        down == NULL || router == NULL || activation_bytes > UINT32_MAX ||
        required_scratch > scratch_size || required_scratch > UINT32_MAX ||
        (uint64_t)scratch_address + required_scratch > UINT32_MAX) {
        errno = required_scratch > scratch_size ? ENOSPC : EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand gate_activation = {
        OPENNPUX_NPU_OPERAND_OUTPUT, scratch_address,
        (uint32_t)activation_bytes, 0};
    const struct opennpux_npu_functional_operand up_activation = {
        OPENNPUX_NPU_OPERAND_OUTPUT,
        scratch_address + (uint32_t)activation_bytes,
        (uint32_t)activation_bytes, 0};
    const struct opennpux_npu_functional_operand router_activation = {
        OPENNPUX_NPU_OPERAND_OUTPUT,
        scratch_address + (uint32_t)(activation_bytes * 2),
        (uint32_t)router_bytes, 0};
    uint32_t emitted = 0;
    uint32_t produced = 0;
#define LOWER_DENSE(INPUT, WEIGHT, OUTPUT, K, N)                              \
    do {                                                                       \
        if (lower_dense_matmul_operands(                                       \
                (INPUT), (WEIGHT), (OUTPUT), request->rows, (K), (N),          \
                extmem_base, extmem_size, first_command_id + emitted,          \
                commands + emitted, command_capacity - emitted, &produced) !=  \
            0) {                                                               \
            return -1;                                                         \
        }                                                                      \
        emitted += produced;                                                   \
    } while (0)
    LOWER_DENSE(input, gate, &gate_activation, parameters->input_features,
                parameters->intermediate_features);
    LOWER_DENSE(input, up, &up_activation, parameters->input_features,
                parameters->intermediate_features);
    if (command_capacity - emitted < 2) {
        errno = ENOSPC;
        return -1;
    }
    struct opennpux_xgraph_command *command = &commands[emitted++];
    memset(command, 0, sizeof(*command));
    command->command_id = first_command_id + emitted - 1;
    command->opcode = OPENNPUX_XGRAPH_OP_TSILU;
    command->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    command->dim0 = request->rows;
    command->dim1 = parameters->intermediate_features;
    command->dim2 = 1;
    if (set_operands(command, &gate_activation, &gate_activation, NULL,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }
    command = &commands[emitted++];
    memset(command, 0, sizeof(*command));
    command->command_id = first_command_id + emitted - 1;
    command->opcode = OPENNPUX_XGRAPH_OP_TMUL;
    command->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    command->dim0 = request->rows;
    command->dim1 = parameters->intermediate_features;
    command->dim2 = 1;
    if (set_operands(command, &gate_activation, &gate_activation,
                     &up_activation, extmem_base, extmem_size) != 0) {
        return -1;
    }
    LOWER_DENSE(&gate_activation, down, output,
                parameters->intermediate_features,
                parameters->output_features);
    LOWER_DENSE(input, router, &router_activation, parameters->input_features,
                1);
    if (command_capacity - emitted < 2) {
        errno = ENOSPC;
        return -1;
    }
    command = &commands[emitted++];
    memset(command, 0, sizeof(*command));
    command->command_id = first_command_id + emitted - 1;
    command->opcode = OPENNPUX_XGRAPH_OP_TSIGMOID;
    command->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    command->dim0 = request->rows;
    command->dim1 = 1;
    command->dim2 = 1;
    if (set_operands(command, &router_activation, &router_activation, NULL,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }
    command = &commands[emitted++];
    memset(command, 0, sizeof(*command));
    command->command_id = first_command_id + emitted - 1;
    command->opcode = OPENNPUX_XGRAPH_OP_TROW_SCALE;
    command->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    command->dim0 = request->rows;
    command->dim1 = parameters->output_features;
    command->dim2 = 1;
    if (set_operands(command, output, output, &router_activation, extmem_base,
                     extmem_size) != 0) {
        return -1;
    }
#undef LOWER_DENSE
    *command_count = emitted;
    return 0;
}

int
opennpux_npu_xgraph_lower_primitive(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_xgraph_lowering_options *options,
    uint32_t extmem_base, uint32_t extmem_size,
    struct opennpux_xgraph_command *command)
{
    if (command == NULL ||
        validate_request(request, parameters, extmem_size) != 0) {
        if (errno == 0) {
            errno = EINVAL;
        }
        return -1;
    }

    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *secondary =
        find_operand(request, OPENNPUX_NPU_OPERAND_SECONDARY);
    const struct opennpux_npu_functional_operand *weight =
        find_operand(request, OPENNPUX_NPU_OPERAND_WEIGHT);
    const struct opennpux_npu_functional_operand *indices =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT_INDICES);
    const struct opennpux_npu_functional_operand *output =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const struct opennpux_npu_functional_operand *output_secondary =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY);
    const struct opennpux_npu_functional_operand *output_indices =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_INDICES);

    memset(command, 0, sizeof(*command));
    command->command_id = request->command_id;
    command->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    command->dim0 = request->rows;
    command->dim1 = request->features;
    command->dim2 = 1;

    switch (request->opcode) {
    case OPENNPUX_NPU_OP_EMBED:
        command->opcode = OPENNPUX_XGRAPH_OP_TGATHER;
        command->scalar0 = parameters->input_features;
        return set_operands(command, output, weight, indices,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_MATMUL:
        if (has_gptq_operands(request)) {
            errno = ENOTSUP;
            return -1;
        }
        if (request->rows > OPENNPUX_TMMA_TILE_LIMIT ||
            parameters->output_features > OPENNPUX_TMMA_TILE_LIMIT ||
            parameters->input_features > OPENNPUX_TMMA_TILE_LIMIT) {
            errno = ENOTSUP;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TMMA;
        command->dim1 = parameters->output_features;
        command->dim2 = parameters->input_features;
        return set_operands(command, output, input, weight,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_ADD:
    case OPENNPUX_NPU_OP_COMBINE:
        command->opcode = OPENNPUX_XGRAPH_OP_TADD;
        return set_operands(command, output, input, secondary,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_MUL:
        command->opcode = OPENNPUX_XGRAPH_OP_TMUL;
        return set_operands(command, output, input, secondary,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_NORMALIZE:
        if ((parameters->flags &
             OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET) != 0) {
            errno = ENOTSUP;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TRMSNORM;
        command->flags =
            ((parameters->flags &
              OPENNPUX_NPU_PARAMETER_NORM_WEIGHT_OFFSET) != 0
                 ? OPENNPUX_XGRAPH_TRMSNORM_WEIGHT_OFFSET
                 : 0) |
            ((parameters->flags &
              OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE) != 0
                 ? OPENNPUX_XGRAPH_TRMSNORM_BFLOAT16_INPUT
                 : 0);
        memcpy(&command->scalar0, &request->epsilon,
               sizeof(command->scalar0));
        return set_operands(command, output, input, weight,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_ROPE:
        if (options == NULL ||
            options->rope_layout > OPENNPUX_NPU_XGRAPH_ROPE_HALF_SPLIT) {
            errno = EINVAL;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TROPE;
        command->scalar0 = options->rope_layout;
        return set_operands(command, output, input, secondary,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_SOFTMAX:
        command->opcode = OPENNPUX_XGRAPH_OP_TSOFTMAX;
        return set_operands(command, output, input, NULL,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_ACTIVATION:
        if (options == NULL ||
            options->activation != OPENNPUX_NPU_XGRAPH_ACTIVATION_SILU) {
            errno = ENOTSUP;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TSILU;
        return set_operands(command, output, input, NULL,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_TOPK: {
        if (request->top_k == 0) {
            errno = EINVAL;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TTOPK;
        command->scalar0 = request->top_k;
        const uint64_t full_plane_bytes =
            (uint64_t)request->rows * request->top_k * sizeof(uint32_t);
        const uint64_t row_plane_bytes =
            (uint64_t)request->top_k * sizeof(uint32_t);
        if (output != NULL && output_indices != NULL) {
            if (full_plane_bytes > UINT32_MAX ||
                output->byte_size < full_plane_bytes ||
                output_indices->byte_size < full_plane_bytes ||
                operand_offset(output_indices, extmem_base, extmem_size,
                               &command->reserved[0]) != 0) {
                if (errno == 0) {
                    errno = EINVAL;
                }
                return -1;
            }
            command->flags = OPENNPUX_XGRAPH_TTOPK_SPLIT_OUTPUT;
            return set_operands(command, output, input, NULL, extmem_base,
                                extmem_size);
        }
        if (output == NULL && output_indices != NULL) {
            const int last_row_only =
                output_indices->byte_size >= row_plane_bytes &&
                output_indices->byte_size < full_plane_bytes;
            const uint64_t required_indices =
                last_row_only ? row_plane_bytes : full_plane_bytes;
            const uint64_t input_bytes =
                (uint64_t)request->rows * request->features * sizeof(float);
            if (input == NULL || options == NULL ||
                options->topk_packed_size < required_indices ||
                required_indices > UINT32_MAX || input_bytes > UINT32_MAX ||
                input->byte_size < input_bytes ||
                output_indices->byte_size < required_indices) {
                errno = EINVAL;
                return -1;
            }
            struct opennpux_npu_functional_operand values = {
                OPENNPUX_NPU_OPERAND_OUTPUT,
                options->topk_packed_address,
                (uint32_t)required_indices, 0};
            struct opennpux_npu_functional_operand selected_input = *input;
            if (last_row_only) {
                const uint64_t row_bytes =
                    (uint64_t)request->features * sizeof(float);
                const uint64_t selected_address =
                    (uint64_t)input->address +
                    (uint64_t)(request->rows - 1) * row_bytes;
                if (row_bytes > UINT32_MAX || selected_address > UINT32_MAX) {
                    errno = EOVERFLOW;
                    return -1;
                }
                selected_input.address = (uint32_t)selected_address;
                selected_input.byte_size = (uint32_t)row_bytes;
                command->dim0 = 1;
            }
            command->flags = OPENNPUX_XGRAPH_TTOPK_SPLIT_OUTPUT;
            if (operand_offset(output_indices, extmem_base, extmem_size,
                               &command->reserved[0]) != 0) {
                return -1;
            }
            return set_operands(command, &values, &selected_input, NULL,
                                extmem_base, extmem_size);
        }
        if (options == NULL || options->topk_packed_size == 0 ||
            full_plane_bytes > UINT32_MAX / 2 ||
            full_plane_bytes * 2 > options->topk_packed_size) {
            errno = options == NULL || options->topk_packed_size == 0
                        ? EINVAL
                        : ENOSPC;
            return -1;
        }
        const struct opennpux_npu_functional_operand packed = {
            OPENNPUX_NPU_OPERAND_OUTPUT, options->topk_packed_address,
            options->topk_packed_size, 0};
        return set_operands(command, &packed, input, NULL, extmem_base,
                            extmem_size);
    }
    case OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION: {
        const uint32_t kernel_width = parameters->intermediate_features;
        const uint64_t tensor_bytes =
            (uint64_t)request->rows * request->features * sizeof(float);
        const uint64_t weight_bytes =
            (uint64_t)kernel_width * request->features * sizeof(float);
        const uint64_t state_bytes = kernel_width > 1
            ? (uint64_t)(kernel_width - 1) * request->features * sizeof(float)
            : 0;
        const int has_previous_state = secondary != NULL;
        const int has_next_state = output_secondary != NULL;
        if (kernel_width == 0 || tensor_bytes > UINT32_MAX ||
            weight_bytes > UINT32_MAX || state_bytes > UINT32_MAX ||
            input == NULL || weight == NULL || output == NULL ||
            input->byte_size < tensor_bytes ||
            weight->byte_size < weight_bytes ||
            output->byte_size < tensor_bytes ||
            has_previous_state != has_next_state ||
            ((has_previous_state != 0) &&
             (secondary->byte_size < state_bytes ||
              output_secondary->byte_size < state_bytes))) {
            errno = EINVAL;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TCAUSALCONV;
        command->dim2 = kernel_width;
        if (has_previous_state != 0) {
            command->flags |= OPENNPUX_XGRAPH_TCAUSALCONV_STATEFUL;
            if (operand_offset(secondary, extmem_base, extmem_size,
                               &command->reserved[0]) != 0 ||
                operand_offset(output_secondary, extmem_base, extmem_size,
                               &command->reserved[1]) != 0) {
                return -1;
            }
        }
        if ((parameters->flags & OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET) != 0) {
            command->flags |= OPENNPUX_XGRAPH_TCAUSALCONV_SILU;
        }
        return set_operands(command, output, input, weight,
                            extmem_base, extmem_size);
    }
    case OPENNPUX_NPU_OP_CONVOLUTION: {
        if (options == NULL) {
            errno = EINVAL;
            return -1;
        }
        const struct opennpux_npu_xgraph_convolution_options *conv =
            &options->convolution;
        const uint32_t input_channels = request->features;
        if (input == NULL || weight == NULL || output == NULL ||
            request->rows > UINT16_MAX || input_channels == 0 ||
            input_channels > UINT16_MAX || conv->input_height == 0 ||
            conv->input_width == 0 || conv->output_height == 0 ||
            conv->output_width == 0 || conv->output_channels == 0 ||
            conv->output_channels > UINT16_MAX || conv->kernel_height == 0 ||
            conv->kernel_width == 0 || conv->stride_height == 0 ||
            conv->stride_width == 0 || conv->dilation_height == 0 ||
            conv->dilation_width == 0 || conv->groups == 0 ||
            conv->groups > UINT16_MAX || input_channels % conv->groups != 0 ||
            conv->output_channels % conv->groups != 0 ||
            conv->input_layout != OPENNPUX_NPU_XGRAPH_LAYOUT_NHWC ||
            conv->weight_layout != OPENNPUX_NPU_XGRAPH_LAYOUT_OHWI ||
            conv->output_layout != OPENNPUX_NPU_XGRAPH_LAYOUT_NHWC ||
            conv->input_height > UINT16_MAX || conv->input_width > UINT16_MAX ||
            conv->output_height > UINT16_MAX ||
            conv->output_width > UINT16_MAX ||
            conv->kernel_height > UINT16_MAX ||
            conv->kernel_width > UINT16_MAX ||
            conv->stride_height > UINT8_MAX ||
            conv->stride_width > UINT8_MAX ||
            conv->padding_top > UINT8_MAX ||
            conv->padding_bottom > UINT8_MAX ||
            conv->padding_left > UINT8_MAX ||
            conv->padding_right > UINT8_MAX ||
            conv->dilation_height > UINT8_MAX ||
            conv->dilation_width > UINT8_MAX) {
            errno = EINVAL;
            return -1;
        }
        const uint64_t input_elements =
            (uint64_t)request->rows * conv->input_height *
            conv->input_width * input_channels;
        const uint64_t weight_elements =
            (uint64_t)conv->output_channels * conv->kernel_height *
            conv->kernel_width * (input_channels / conv->groups);
        const uint64_t output_elements =
            (uint64_t)request->rows * conv->output_height *
            conv->output_width * conv->output_channels;
        const uint64_t padded_height =
            (uint64_t)conv->input_height + conv->padding_top +
            conv->padding_bottom;
        const uint64_t padded_width =
            (uint64_t)conv->input_width + conv->padding_left +
            conv->padding_right;
        const uint64_t effective_kernel_height =
            (uint64_t)(conv->kernel_height - 1) * conv->dilation_height + 1;
        const uint64_t effective_kernel_width =
            (uint64_t)(conv->kernel_width - 1) * conv->dilation_width + 1;
        if (padded_height < effective_kernel_height ||
            padded_width < effective_kernel_width ||
            conv->output_height !=
                (padded_height - effective_kernel_height) /
                        conv->stride_height +
                    1 ||
            conv->output_width !=
                (padded_width - effective_kernel_width) /
                        conv->stride_width +
                    1 ||
            input_elements > UINT32_MAX / sizeof(float) ||
            weight_elements > UINT32_MAX / sizeof(float) ||
            output_elements > UINT32_MAX / sizeof(float) ||
            input->byte_size < input_elements * sizeof(float) ||
            weight->byte_size < weight_elements * sizeof(float) ||
            output->byte_size < output_elements * sizeof(float) ||
            (secondary != NULL &&
             secondary->byte_size <
                 (uint64_t)conv->output_channels * sizeof(float))) {
            errno = EINVAL;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TCONV;
        command->dim0 = request->rows;
        command->dim1 = conv->input_height;
        command->dim2 = conv->input_width;
        command->scalar0 = input_channels;
        command->flags = conv->output_channels | (conv->groups << 16);
        command->reserved[1] =
            conv->output_height | (conv->output_width << 16);
        command->reserved[2] =
            conv->kernel_height | (conv->kernel_width << 16);
        command->reserved[3] = conv->stride_height |
            (conv->stride_width << 8) | (conv->dilation_height << 16) |
            (conv->dilation_width << 24);
        command->reserved[4] = conv->padding_top |
            (conv->padding_bottom << 8) | (conv->padding_left << 16) |
            (conv->padding_right << 24);
        if (secondary != NULL &&
            operand_offset(secondary, extmem_base, extmem_size,
                           &command->reserved[0]) != 0) {
            return -1;
        }
        return set_operands(command, output, input, weight,
                            extmem_base, extmem_size);
    }
    case OPENNPUX_NPU_OP_ATTENTION: {
        const struct opennpux_npu_functional_operand *tertiary =
            find_operand(request, OPENNPUX_NPU_OPERAND_INPUT_TERTIARY);
        const uint64_t query_elements =
            (uint64_t)request->rows * request->heads * request->head_dim;
        const uint64_t state_elements =
            (uint64_t)2 * request->kv_length * request->kv_heads *
            request->head_dim;
        const uint64_t query_bytes = query_elements * sizeof(float);
        const uint64_t state_bytes = state_elements * sizeof(float);
        if (request->heads == 0 || request->kv_heads == 0 ||
            request->head_dim == 0 || request->kv_length == 0 ||
            request->rows > UINT16_MAX ||
            (uint64_t)request->heads * request->head_dim > UINT16_MAX ||
            request->rows > request->kv_length ||
            request->heads % request->kv_heads != 0 ||
            query_bytes > UINT32_MAX || state_bytes > UINT32_MAX ||
            input == NULL || secondary == NULL || output == NULL ||
            input->byte_size < query_bytes ||
            secondary->byte_size < state_bytes ||
            output->byte_size < query_bytes ||
            (tertiary != NULL && tertiary->byte_size < query_bytes)) {
            errno = EINVAL;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TATTENTION;
        command->dim0 = request->rows;
        command->dim1 = request->heads;
        command->dim2 = request->head_dim;
        command->scalar0 = request->kv_heads;
        command->flags = request->kv_length;
        if (tertiary != NULL) {
            command->reserved[1] = OPENNPUX_XGRAPH_TATTENTION_GATED;
            if (operand_offset(tertiary, extmem_base, extmem_size,
                               &command->reserved[0]) != 0) {
                return -1;
            }
        }
        return set_operands(command, output, input, secondary,
                            extmem_base, extmem_size);
    }
    default:
        errno = ENOTSUP;
        return -1;
    }
}

int
opennpux_npu_xgraph_lower_sequence(
    const struct opennpux_npu_functional_request *requests,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_xgraph_lowering_options *options,
    uint32_t command_count, uint32_t extmem_base, uint32_t extmem_size,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    struct opennpux_npu_xgraph_lowering_failure *failure)
{
    if (failure != NULL) {
        memset(failure, 0, sizeof(*failure));
        failure->command_index = UINT32_MAX;
    }
    if (requests == NULL || parameters == NULL || commands == NULL ||
        command_count == 0 || command_count > OPENNPUX_XGRAPH_MAX_COMMANDS ||
        command_count > command_capacity) {
        errno = command_count > command_capacity ? ENOSPC : EINVAL;
        return -1;
    }
    memset(commands, 0, command_count * sizeof(*commands));
    for (uint32_t index = 0; index < command_count; ++index) {
        const struct opennpux_npu_xgraph_lowering_options *command_options =
            options == NULL ? NULL : &options[index];
        if (requests[index].command_id != index ||
            opennpux_npu_xgraph_lower_primitive(
                &requests[index], &parameters[index], command_options,
                extmem_base, extmem_size, &commands[index]) != 0) {
            const int error = requests[index].command_id != index ?
                EINVAL : (errno == 0 ? EIO : errno);
            if (failure != NULL) {
                failure->command_index = index;
                failure->command_id = requests[index].command_id;
                failure->opcode = requests[index].opcode;
                failure->error_code = error;
            }
            errno = error;
            return -1;
        }
    }
    return 0;
}

int
opennpux_npu_xgraph_lower_gptq_matmul(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    struct opennpux_npu_gptq_tile_plan plan;
    if (commands == NULL || command_count == NULL ||
        opennpux_npu_gptq_plan_tiles(
            request, parameters, extmem_base, extmem_size, scratch_address,
            scratch_size, &plan) != 0) {
        if (errno == 0) {
            errno = EINVAL;
        }
        return -1;
    }
    const uint64_t commands_per_output_tile =
        (uint64_t)plan.input_tile_count * (plan.rows + 1u) +
        (uint64_t)(plan.input_tile_count - 1u) * plan.rows;
    const uint64_t required_commands =
        (uint64_t)plan.tile_count * commands_per_output_tile;
    if (required_commands > command_capacity || required_commands > UINT32_MAX ||
        first_command_id > UINT32_MAX - (uint32_t)required_commands) {
        errno = ENOSPC;
        return -1;
    }
    const uint32_t scale_data_type =
        xopennpux_scale_data_type(parameters->scale_data_type);
    if (scale_data_type == UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    if (input == NULL) {
        errno = EINVAL;
        return -1;
    }

    uint32_t emitted = 0;
    for (uint32_t tile_index = 0; tile_index < plan.tile_count; ++tile_index) {
        for (uint32_t input_tile_index = 0;
             input_tile_index < plan.input_tile_count; ++input_tile_index) {
            struct opennpux_npu_gptq_tile tile;
            if (opennpux_npu_gptq_get_tile_2d(
                    request, &plan, tile_index, input_tile_index, &tile) != 0) {
                return -1;
            }
            struct opennpux_xgraph_command *dequant = &commands[emitted++];
            memset(dequant, 0, sizeof(*dequant));
            dequant->opcode = OPENNPUX_XGRAPH_OP_TDEQUANT;
            dequant->dim0 = 1;
            dequant->dim1 = tile.column_count;
            dequant->dim2 = tile.input_count;
            dequant->flags = (tile.group_count << 16) | tile.group_base;
            dequant->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
            dequant->command_id = first_command_id + emitted - 1;
            dequant->scalar0 =
                (parameters->quantization_group_size & UINT32_C(0xffff)) |
                ((parameters->quantized_zero_bias & UINT32_C(0xf)) << 16) |
                ((scale_data_type & UINT32_C(0xf)) << 20) |
                (plan.has_g_idx != 0 ? UINT32_C(1) << 24 : 0);
            if (address_offset(tile.dequantized_address,
                               tile.dequantized_bytes, extmem_base,
                               extmem_size,
                               &dequant->destination_offset) != 0 ||
                address_offset(tile.qweight.address, tile.qweight.row_bytes,
                               extmem_base, extmem_size,
                               &dequant->source0_offset) != 0 ||
                address_offset(tile.qzeros.address, tile.qzeros.row_bytes,
                               extmem_base, extmem_size,
                               &dequant->source1_offset) != 0 ||
                address_offset(tile.scales.address, tile.scales.row_bytes,
                               extmem_base, extmem_size,
                               &dequant->reserved[0]) != 0 ||
                (plan.has_g_idx != 0 &&
                 address_offset(tile.g_idx.address, tile.g_idx.row_bytes,
                                extmem_base, extmem_size,
                                &dequant->reserved[1]) != 0)) {
                return -1;
            }
            dequant->reserved[2] = tile.qweight.row_stride_bytes;
            dequant->reserved[3] = tile.qzeros.row_stride_bytes;
            dequant->reserved[4] = tile.scales.row_stride_bytes;

            for (uint32_t row = 0; row < plan.rows; ++row) {
                struct opennpux_xgraph_command *mma = &commands[emitted++];
                memset(mma, 0, sizeof(*mma));
                mma->opcode = OPENNPUX_XGRAPH_OP_TMMA;
                mma->dim0 = 1;
                mma->dim1 = tile.column_count;
                mma->dim2 = tile.input_count;
                mma->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
                mma->command_id = first_command_id + emitted - 1;
                const uint64_t input_address =
                    (uint64_t)input->address +
                    ((uint64_t)row * plan.input_columns + tile.input_base) *
                        sizeof(float);
                const uint64_t output_address =
                    input_tile_index == 0
                        ? (uint64_t)tile.output.address +
                              (uint64_t)row * tile.output.row_stride_bytes
                        : tile.partial_address;
                if (input_address > UINT32_MAX ||
                    output_address > UINT32_MAX ||
                    address_offset((uint32_t)input_address,
                                   tile.input_count * sizeof(float),
                                   extmem_base, extmem_size,
                                   &mma->source0_offset) != 0 ||
                    address_offset(tile.dequantized_address,
                                   tile.dequantized_bytes, extmem_base,
                                   extmem_size, &mma->source1_offset) != 0 ||
                    address_offset((uint32_t)output_address,
                                   tile.column_count * sizeof(float),
                                   extmem_base, extmem_size,
                                   &mma->destination_offset) != 0) {
                    return -1;
                }

                if (input_tile_index != 0) {
                    struct opennpux_xgraph_command *add =
                        &commands[emitted++];
                    memset(add, 0, sizeof(*add));
                    add->opcode = OPENNPUX_XGRAPH_OP_TADD;
                    add->dim0 = 1;
                    add->dim1 = tile.column_count;
                    add->dim2 = 1;
                    add->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
                    add->command_id = first_command_id + emitted - 1;
                    if (address_offset(
                            (uint32_t)((uint64_t)tile.output.address +
                                       (uint64_t)row *
                                           tile.output.row_stride_bytes),
                            tile.column_count * sizeof(float), extmem_base,
                            extmem_size, &add->destination_offset) != 0) {
                        return -1;
                    }
                    add->source0_offset = add->destination_offset;
                    if (address_offset(tile.partial_address,
                                       tile.column_count * sizeof(float),
                                       extmem_base, extmem_size,
                                       &add->source1_offset) != 0) {
                        return -1;
                    }
                }
            }
        }
    }
    *command_count = emitted;
    return 0;
}

static int
append_projection_operand(
    struct opennpux_npu_functional_request *projection,
    const struct opennpux_npu_functional_request *expert,
    uint32_t source_role, uint32_t projection_role, int required)
{
    const struct opennpux_npu_functional_operand *source =
        find_operand(expert, source_role);
    if (source == NULL) {
        if (!required) {
            return 0;
        }
        errno = EINVAL;
        return -1;
    }
    if (projection->operand_count >= OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS) {
        errno = EOVERFLOW;
        return -1;
    }
    struct opennpux_npu_functional_operand *destination =
        &projection->operands[projection->operand_count++];
    *destination = *source;
    destination->role = projection_role;
    return 0;
}

static int
build_expert_projection(
    const struct opennpux_npu_functional_request *expert,
    const struct opennpux_npu_operator_parameters *expert_parameters,
    uint32_t input_role, uint32_t output_role, uint32_t qweight_role,
    uint32_t qzeros_role, uint32_t scales_role, uint32_t g_idx_role,
    uint32_t input_features, uint32_t output_features,
    struct opennpux_npu_functional_request *projection,
    struct opennpux_npu_operator_parameters *projection_parameters)
{
    *projection = *expert;
    projection->opcode = OPENNPUX_NPU_OP_MATMUL;
    projection->features = input_features;
    projection->operand_count = 0;
    memset(projection->operands, 0, sizeof(projection->operands));
    *projection_parameters = *expert_parameters;
    projection_parameters->opcode = OPENNPUX_NPU_OP_MATMUL;
    projection_parameters->input_features = input_features;
    projection_parameters->output_features = output_features;

    return append_projection_operand(
               projection, expert, input_role, OPENNPUX_NPU_OPERAND_INPUT,
               1) == 0 &&
           append_projection_operand(
               projection, expert, output_role, OPENNPUX_NPU_OPERAND_OUTPUT,
               1) == 0 &&
           append_projection_operand(
               projection, expert, qweight_role,
               OPENNPUX_NPU_OPERAND_QWEIGHT, 1) == 0 &&
           append_projection_operand(
               projection, expert, qzeros_role,
               OPENNPUX_NPU_OPERAND_QZEROS, 1) == 0 &&
           append_projection_operand(
               projection, expert, scales_role,
               OPENNPUX_NPU_OPERAND_SCALES, 1) == 0 &&
           append_projection_operand(
               projection, expert, g_idx_role,
               OPENNPUX_NPU_OPERAND_G_IDX, 0) == 0
        ? 0
        : -1;
}

int
opennpux_npu_xgraph_lower_gptq_expert(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (command_count != NULL) {
        *command_count = 0;
    }
    if (request == NULL || parameters == NULL || commands == NULL ||
        command_count == NULL ||
        validate_request(request, parameters, extmem_size) != 0 ||
        request->opcode != OPENNPUX_NPU_OP_EXPERT ||
        (parameters->flags & OPENNPUX_NPU_PARAMETER_GPTQ) == 0 ||
        parameters->input_features == 0 ||
        parameters->intermediate_features == 0 ||
        parameters->output_features != parameters->input_features) {
        errno = EINVAL;
        return -1;
    }

    struct opennpux_npu_functional_request projection;
    struct opennpux_npu_operator_parameters projection_parameters;
    uint32_t emitted = 0;
    uint32_t produced = 0;
#define LOWER_EXPERT_PROJECTION(input_role, output_role, prefix, input_size,   \
                                output_size)                                  \
    do {                                                                       \
        if (build_expert_projection(                                           \
                request, parameters, input_role, output_role,                  \
                OPENNPUX_NPU_OPERAND_##prefix##_QWEIGHT,                       \
                OPENNPUX_NPU_OPERAND_##prefix##_QZEROS,                        \
                OPENNPUX_NPU_OPERAND_##prefix##_SCALES,                        \
                OPENNPUX_NPU_OPERAND_##prefix##_G_IDX, input_size,             \
                output_size, &projection, &projection_parameters) != 0 ||      \
            opennpux_npu_xgraph_lower_gptq_matmul(                             \
                &projection, &projection_parameters, extmem_base, extmem_size, \
                scratch_address, scratch_size, first_command_id + emitted,     \
                commands + emitted, command_capacity - emitted,                \
                &produced) != 0) {                                             \
            return -1;                                                         \
        }                                                                      \
        emitted += produced;                                                   \
    } while (0)

    LOWER_EXPERT_PROJECTION(
        OPENNPUX_NPU_OPERAND_INPUT, OPENNPUX_NPU_OPERAND_GATE_OUTPUT, GATE,
        parameters->input_features, parameters->intermediate_features);
    LOWER_EXPERT_PROJECTION(
        OPENNPUX_NPU_OPERAND_INPUT, OPENNPUX_NPU_OPERAND_UP_OUTPUT, UP,
        parameters->input_features, parameters->intermediate_features);

    if (command_capacity - emitted < 2) {
        errno = ENOSPC;
        return -1;
    }
    const struct opennpux_npu_functional_operand *gate =
        find_operand(request, OPENNPUX_NPU_OPERAND_GATE_OUTPUT);
    const struct opennpux_npu_functional_operand *up =
        find_operand(request, OPENNPUX_NPU_OPERAND_UP_OUTPUT);
    const struct opennpux_npu_functional_operand *activated =
        find_operand(request, OPENNPUX_NPU_OPERAND_ACTIVATED);
    const uint64_t intermediate_elements =
        (uint64_t)request->rows * parameters->intermediate_features;
    const uint64_t intermediate_bytes = intermediate_elements * sizeof(float);
    if (intermediate_elements > UINT32_MAX || intermediate_bytes > UINT32_MAX ||
        gate == NULL || up == NULL || activated == NULL ||
        gate->byte_size < intermediate_bytes || up->byte_size < intermediate_bytes ||
        activated->byte_size < intermediate_bytes) {
        errno = EINVAL;
        return -1;
    }

    struct opennpux_xgraph_command *silu = &commands[emitted++];
    memset(silu, 0, sizeof(*silu));
    silu->opcode = OPENNPUX_XGRAPH_OP_TSILU;
    silu->command_id = first_command_id + emitted - 1;
    silu->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    silu->dim0 = request->rows;
    silu->dim1 = parameters->intermediate_features;
    silu->dim2 = 1;
    if (set_operands(silu, activated, gate, NULL, extmem_base, extmem_size) != 0) {
        return -1;
    }

    struct opennpux_xgraph_command *multiply = &commands[emitted++];
    memset(multiply, 0, sizeof(*multiply));
    multiply->opcode = OPENNPUX_XGRAPH_OP_TMUL;
    multiply->command_id = first_command_id + emitted - 1;
    multiply->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    multiply->dim0 = request->rows;
    multiply->dim1 = parameters->intermediate_features;
    multiply->dim2 = 1;
    if (set_operands(multiply, activated, activated, up, extmem_base,
                     extmem_size) != 0) {
        return -1;
    }

    LOWER_EXPERT_PROJECTION(
        OPENNPUX_NPU_OPERAND_ACTIVATED, OPENNPUX_NPU_OPERAND_OUTPUT, DOWN,
        parameters->intermediate_features, parameters->output_features);
#undef LOWER_EXPERT_PROJECTION

    *command_count = emitted;
    return 0;
}

int
opennpux_npu_xgraph_lower_routed_expert(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (command_count != NULL) {
        *command_count = 0;
    }
    if (validate_request(request, parameters, extmem_size) != 0 ||
        request->opcode != OPENNPUX_NPU_OP_EXPERT || commands == NULL ||
        command_count == NULL || command_capacity == 0 ||
        parameters->input_features == 0 ||
        parameters->output_features == 0 ||
        parameters->intermediate_features == 0 ||
        parameters->quantization_bits == 0 ||
        parameters->quantization_bits > UINT8_MAX ||
        parameters->quantization_group_size == 0 ||
        parameters->quantization_group_size > UINT32_C(0x00ffffff)) {
        errno = EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *expert_ids =
        find_operand(request, OPENNPUX_NPU_OPERAND_SECONDARY);
    const struct opennpux_npu_functional_operand *route_weights =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT_TERTIARY);
    const struct opennpux_npu_functional_operand *output =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const uint64_t input_bytes = (uint64_t)request->rows *
        parameters->input_features * sizeof(float);
    const uint64_t output_bytes = (uint64_t)request->rows *
        parameters->output_features * sizeof(float);
    if (input == NULL || expert_ids == NULL || route_weights == NULL ||
        output == NULL || input_bytes > input->byte_size ||
        output_bytes > output->byte_size ||
        expert_ids->byte_size == 0 ||
        expert_ids->byte_size % (request->rows * sizeof(uint32_t)) != 0) {
        errno = EINVAL;
        return -1;
    }
    const uint32_t active_experts =
        expert_ids->byte_size / (request->rows * sizeof(uint32_t));
    const uint64_t route_bytes = (uint64_t)request->rows * active_experts *
        sizeof(float);
    if (active_experts == 0 || route_bytes > route_weights->byte_size) {
        errno = EINVAL;
        return -1;
    }

    struct opennpux_xgraph_command command;
    memset(&command, 0, sizeof(command));
    command.opcode = OPENNPUX_XGRAPH_OP_TROUTED_EXPERT;
    command.flags = OPENNPUX_XGRAPH_TROUTED_EXPERT_WEIGHT_PLAN;
    command.dim0 = request->rows;
    command.dim1 = parameters->input_features;
    command.dim2 = parameters->intermediate_features;
    command.scalar0 = active_experts;
    command.data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    command.command_id = first_command_id;
    if (set_operands(&command, output, input, expert_ids, extmem_base,
                     extmem_size) != 0 ||
        operand_offset(route_weights, extmem_base, extmem_size,
                       &command.reserved[0]) != 0) {
        return -1;
    }
    command.reserved[1] = request->command_id;
    command.reserved[2] = parameters->output_features;
    command.reserved[3] = parameters->quantization_bits |
        (parameters->quantization_group_size << 8);
    command.reserved[4] = parameters->scale_data_type;
    commands[0] = command;
    *command_count = 1;
    return 0;
}

int
opennpux_npu_xgraph_lower_dma(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (commands == NULL || command_count == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (command_capacity < 2) {
        errno = ENOSPC;
        return -1;
    }
    if (validate_request(request, parameters, extmem_size) != 0) {
        return -1;
    }
    if (request->opcode != OPENNPUX_NPU_OP_DMA || request->kv_heads == 0 ||
        request->head_dim == 0 || request->kv_length < request->rows ||
        first_command_id > UINT32_MAX - 1) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_functional_operand *key =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *value =
        find_operand(request, OPENNPUX_NPU_OPERAND_SECONDARY);
    const struct opennpux_npu_functional_operand *state =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const uint64_t row_elements =
        (uint64_t)request->kv_heads * request->head_dim;
    const uint64_t copy_elements = (uint64_t)request->rows * row_elements;
    const uint64_t plane_elements =
        (uint64_t)request->kv_length * row_elements;
    const uint64_t copy_bytes = copy_elements * sizeof(float);
    const uint64_t plane_bytes = plane_elements * sizeof(float);
    const uint64_t tail_bytes =
        (uint64_t)(request->kv_length - request->rows) * row_elements *
        sizeof(float);
    if (key == NULL || value == NULL || state == NULL ||
        row_elements > UINT32_MAX || copy_bytes > UINT32_MAX ||
        plane_bytes > UINT32_MAX || key->byte_size < copy_bytes ||
        value->byte_size < copy_bytes || state->byte_size < plane_bytes * 2 ||
        (uint64_t)state->address + plane_bytes * 2 > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand *sources[2] = {key, value};
    for (uint32_t plane = 0; plane < 2; ++plane) {
        const uint64_t destination_address =
            (uint64_t)state->address + plane * plane_bytes + tail_bytes;
        const struct opennpux_npu_functional_operand destination = {
            OPENNPUX_NPU_OPERAND_OUTPUT,
            (uint32_t)destination_address,
            (uint32_t)copy_bytes,
            0,
        };
        struct opennpux_xgraph_command *command = &commands[plane];
        memset(command, 0, sizeof(*command));
        command->opcode = OPENNPUX_XGRAPH_OP_TDMA;
        command->command_id = first_command_id + plane;
        command->data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
        command->dim0 = request->rows;
        command->dim1 = (uint32_t)row_elements;
        command->dim2 = 1;
        if (set_operands(command, &destination, sources[plane], NULL,
                         extmem_base, extmem_size) != 0) {
            return -1;
        }
    }
    *command_count = 2;
    return 0;
}

int
opennpux_npu_xgraph_lower_recurrent_update(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (commands == NULL || command_count == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (validate_request(request, parameters, extmem_size) != 0) {
        return -1;
    }
    if (request->opcode != OPENNPUX_NPU_OP_RECURRENT_UPDATE) {
        errno = EINVAL;
        return -1;
    }

    if ((parameters->flags & OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET) != 0) {
        const struct opennpux_npu_functional_operand *qkv =
            find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
        const struct opennpux_npu_functional_operand *alpha =
            find_operand(request, OPENNPUX_NPU_OPERAND_SECONDARY);
        const struct opennpux_npu_functional_operand *beta =
            find_operand(request, OPENNPUX_NPU_OPERAND_INPUT_TERTIARY);
        const struct opennpux_npu_functional_operand *output =
            find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
        const struct opennpux_npu_functional_operand *state =
            find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY);
        const struct opennpux_npu_functional_operand *a_log =
            find_operand(request, OPENNPUX_NPU_OPERAND_LINEAR_A_LOG_WEIGHT);
        const struct opennpux_npu_functional_operand *dt_bias =
            find_operand(request, OPENNPUX_NPU_OPERAND_LINEAR_DT_BIAS_WEIGHT);
        const uint32_t key_heads = request->heads;
        const uint32_t value_heads = request->kv_heads;
        const uint32_t key_dim = request->head_dim;
        const uint32_t value_dim = value_heads == 0
                                       ? 0
                                       : parameters->output_features /
                                             value_heads;
        const uint64_t key_features = (uint64_t)key_heads * key_dim;
        const uint64_t value_features = (uint64_t)value_heads * value_dim;
        const uint64_t qkv_features = key_features * 2 + value_features;
        const uint64_t qkv_bytes =
            (uint64_t)request->rows * qkv_features * sizeof(float);
        const uint64_t gate_bytes =
            (uint64_t)request->rows * value_heads * sizeof(float);
        const uint64_t output_bytes =
            (uint64_t)request->rows * value_features * sizeof(float);
        const uint64_t state_bytes =
            (uint64_t)value_heads * key_dim * value_dim * sizeof(float);
        const uint64_t vector_bytes = (uint64_t)value_heads * sizeof(float);
        if (command_capacity < 1) {
            errno = ENOSPC;
            return -1;
        }
        if (qkv == NULL || alpha == NULL || beta == NULL || output == NULL ||
            state == NULL || a_log == NULL || dt_bias == NULL ||
            key_heads == 0 || value_heads == 0 || key_dim == 0 ||
            value_dim == 0 || value_heads % key_heads != 0 ||
            parameters->output_features % value_heads != 0 ||
            key_features > UINT32_MAX || qkv_features > UINT32_MAX ||
            value_heads > UINT16_MAX || value_dim > UINT16_MAX ||
            qkv_bytes > UINT32_MAX || gate_bytes > UINT32_MAX ||
            output_bytes > UINT32_MAX || state_bytes > UINT32_MAX ||
            qkv->byte_size < qkv_bytes || alpha->byte_size < gate_bytes ||
            beta->byte_size < gate_bytes || output->byte_size < output_bytes ||
            state->byte_size < state_bytes || a_log->byte_size < vector_bytes ||
            dt_bias->byte_size < vector_bytes) {
            errno = EINVAL;
            return -1;
        }

        memset(commands, 0, sizeof(*commands));
        commands[0].opcode = OPENNPUX_XGRAPH_OP_TRECURRENT;
        commands[0].command_id = first_command_id;
        commands[0].data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
        commands[0].dim0 = request->rows;
        commands[0].dim1 = key_heads;
        commands[0].dim2 = key_dim;
        commands[0].scalar0 = value_heads | (value_dim << 16);
        if (set_operands(&commands[0], output, qkv, alpha,
                         extmem_base, extmem_size) != 0 ||
            operand_offset(beta, extmem_base, extmem_size,
                           &commands[0].reserved[0]) != 0 ||
            operand_offset(state, extmem_base, extmem_size,
                           &commands[0].reserved[1]) != 0 ||
            operand_offset(a_log, extmem_base, extmem_size,
                           &commands[0].reserved[2]) != 0 ||
            operand_offset(dt_bias, extmem_base, extmem_size,
                           &commands[0].reserved[3]) != 0) {
            return -1;
        }
        *command_count = 1;
        return 0;
    }

    if (command_capacity < 2 || first_command_id > UINT32_MAX - 1) {
        errno = command_capacity < 2 ? ENOSPC : EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *output =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const struct opennpux_npu_functional_operand *state =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY);
    const uint64_t row_bytes = (uint64_t)request->features * sizeof(float);
    const uint64_t tensor_bytes = (uint64_t)request->rows * row_bytes;
    if (input == NULL || output == NULL || state == NULL ||
        row_bytes > UINT32_MAX || tensor_bytes > UINT32_MAX ||
        input->byte_size < tensor_bytes || output->byte_size < tensor_bytes ||
        state->byte_size < row_bytes ||
        (uint64_t)input->address + tensor_bytes > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand final_row = {
        OPENNPUX_NPU_OPERAND_INPUT,
        input->address + (uint32_t)(tensor_bytes - row_bytes),
        (uint32_t)row_bytes,
        0,
    };
    memset(commands, 0, 2 * sizeof(*commands));
    commands[0].opcode = OPENNPUX_XGRAPH_OP_TDMA;
    commands[0].command_id = first_command_id;
    commands[0].data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    commands[0].dim0 = request->rows;
    commands[0].dim1 = request->features;
    commands[0].dim2 = 1;
    if (set_operands(&commands[0], output, input, NULL,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }

    commands[1].opcode = OPENNPUX_XGRAPH_OP_TDMA;
    commands[1].command_id = first_command_id + 1;
    commands[1].data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    commands[1].dim0 = 1;
    commands[1].dim1 = request->features;
    commands[1].dim2 = 1;
    if (set_operands(&commands[1], state, &final_row, NULL,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }
    *command_count = 2;
    return 0;
}

int
opennpux_npu_xgraph_lower_router(
    const struct opennpux_npu_functional_request *request,
    const struct opennpux_npu_operator_parameters *parameters,
    uint32_t extmem_base, uint32_t extmem_size, uint32_t scratch_address,
    uint32_t scratch_size, uint32_t first_command_id,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (commands == NULL || command_count == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (command_capacity < 5) {
        errno = ENOSPC;
        return -1;
    }
    if (validate_request(request, parameters, extmem_size) != 0) {
        return -1;
    }
    if (request->opcode != OPENNPUX_NPU_OP_ROUTER || request->top_k == 0 ||
        parameters->input_features == 0 || parameters->output_features == 0 ||
        request->top_k > parameters->output_features ||
        first_command_id > UINT32_MAX - 4) {
        errno = EINVAL;
        return -1;
    }

    const struct opennpux_npu_functional_operand *input =
        find_operand(request, OPENNPUX_NPU_OPERAND_INPUT);
    const struct opennpux_npu_functional_operand *weight =
        find_operand(request, OPENNPUX_NPU_OPERAND_WEIGHT);
    if (weight == NULL) {
        weight = find_operand(request,
                              OPENNPUX_NPU_OPERAND_SHARED_ROUTER_WEIGHT);
    }
    const struct opennpux_npu_functional_operand *indices =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT_INDICES);
    const struct opennpux_npu_functional_operand *weights =
        find_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
    const uint64_t input_elements =
        (uint64_t)request->rows * parameters->input_features;
    const uint64_t weight_elements =
        (uint64_t)parameters->input_features * parameters->output_features;
    const uint64_t logit_elements =
        (uint64_t)request->rows * parameters->output_features;
    const uint64_t selected_elements =
        (uint64_t)request->rows * request->top_k;
    const uint64_t logits_bytes = logit_elements * sizeof(float);
    const uint64_t selected_bytes = selected_elements * sizeof(float);
    const uint64_t required_scratch = logits_bytes + selected_bytes * 2;
    const int gptq = has_gptq_operands(request);
    if (input == NULL || (!gptq && weight == NULL) || indices == NULL ||
        weights == NULL ||
        input_elements > UINT32_MAX || weight_elements > UINT32_MAX ||
        logit_elements > UINT32_MAX || selected_elements > UINT32_MAX ||
        logits_bytes > UINT32_MAX || selected_bytes > UINT32_MAX ||
        required_scratch > scratch_size ||
        (uint64_t)scratch_address + required_scratch > UINT32_MAX ||
        input->byte_size < input_elements * sizeof(float) ||
        (!gptq && weight->byte_size < weight_elements * sizeof(float)) ||
        indices->byte_size < selected_bytes || weights->byte_size < selected_bytes) {
        errno = EINVAL;
        return -1;
    }

    const uint32_t packed_address = scratch_address + (uint32_t)logits_bytes;
    const struct opennpux_npu_functional_operand logits = {
        OPENNPUX_NPU_OPERAND_OUTPUT, scratch_address, (uint32_t)logits_bytes, 0};
    const struct opennpux_npu_functional_operand packed = {
        OPENNPUX_NPU_OPERAND_OUTPUT, packed_address,
        (uint32_t)(selected_bytes * 2), 0};
    const struct opennpux_npu_functional_operand packed_indices = {
        OPENNPUX_NPU_OPERAND_INPUT, packed_address + (uint32_t)selected_bytes,
        (uint32_t)selected_bytes, 0};

    uint32_t projection_commands = 1;
    if (gptq) {
        const uint64_t projection_scratch64 =
            ((uint64_t)scratch_address + required_scratch + 63u) & ~UINT64_C(63);
        const uint64_t scratch_end = (uint64_t)scratch_address + scratch_size;
        if (projection_scratch64 >= scratch_end ||
            projection_scratch64 > UINT32_MAX) {
            errno = ENOSPC;
            return -1;
        }
        struct opennpux_npu_functional_request projection = *request;
        projection.opcode = OPENNPUX_NPU_OP_MATMUL;
        int output_replaced = 0;
        for (uint32_t index = 0; index < projection.operand_count; ++index) {
            if (projection.operands[index].role ==
                OPENNPUX_NPU_OPERAND_OUTPUT) {
                projection.operands[index] = logits;
                output_replaced = 1;
                break;
            }
        }
        struct opennpux_npu_operator_parameters projection_parameters =
            *parameters;
        projection_parameters.opcode = OPENNPUX_NPU_OP_MATMUL;
        if (!output_replaced ||
            opennpux_npu_xgraph_lower_gptq_matmul(
                &projection, &projection_parameters, extmem_base,
                extmem_size, (uint32_t)projection_scratch64,
                (uint32_t)(scratch_end - projection_scratch64),
                first_command_id, commands, command_capacity - 4,
                &projection_commands) != 0 || projection_commands == 0) {
            return -1;
        }
    } else {
        if (lower_dense_matmul_operands(
                input, weight, &logits, request->rows,
                parameters->input_features, parameters->output_features,
                extmem_base, extmem_size, first_command_id, commands,
                command_capacity - 4, &projection_commands) != 0 ||
            projection_commands == 0) {
            return -1;
        }
    }

    if (projection_commands > command_capacity - 4 ||
        first_command_id > UINT32_MAX - projection_commands - 3) {
        errno = ENOSPC;
        return -1;
    }
    struct opennpux_xgraph_command *tail = commands + projection_commands;
    memset(tail, 0, 4 * sizeof(*tail));

    tail[0].opcode = OPENNPUX_XGRAPH_OP_TTOPK;
    tail[0].dim0 = request->rows;
    tail[0].dim1 = parameters->output_features;
    tail[0].dim2 = 1;
    tail[0].scalar0 = request->top_k;
    tail[0].data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    if (set_operands(&tail[0], &packed, &logits, NULL,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }

    tail[1].opcode = OPENNPUX_XGRAPH_OP_TSOFTMAX;
    tail[1].dim0 = request->rows;
    tail[1].dim1 = request->top_k;
    tail[1].dim2 = 1;
    tail[1].data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    if (set_operands(&tail[1], &packed, &packed, NULL,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }

    tail[2].opcode = OPENNPUX_XGRAPH_OP_TDMA;
    tail[2].dim0 = request->rows;
    tail[2].dim1 = request->top_k;
    tail[2].dim2 = 1;
    tail[2].data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    if (set_operands(&tail[2], weights, &packed, NULL,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }

    tail[3].opcode = OPENNPUX_XGRAPH_OP_TDMA;
    tail[3].dim0 = request->rows;
    tail[3].dim1 = request->top_k;
    tail[3].dim2 = 1;
    tail[3].data_type = OPENNPUX_XGRAPH_DTYPE_FP32;
    if (set_operands(&tail[3], indices, &packed_indices, NULL,
                     extmem_base, extmem_size) != 0) {
        return -1;
    }
    for (uint32_t index = 0; index < 4; ++index) {
        tail[index].command_id = first_command_id + projection_commands + index;
    }
    *command_count = projection_commands + 4;
    return 0;
}

int
opennpux_npu_xgraph_lower_batch(
    const struct opennpux_npu_functional_request *requests,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_xgraph_lowering_options *options,
    uint32_t request_count, uint32_t extmem_base, uint32_t extmem_size,
    uint32_t scratch_address, uint32_t scratch_size,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_origins, uint32_t *requests_consumed,
    uint32_t *commands_emitted,
    struct opennpux_npu_xgraph_lowering_failure *failure)
{
    if (requests_consumed != NULL) {
        *requests_consumed = 0;
    }
    if (commands_emitted != NULL) {
        *commands_emitted = 0;
    }
    if (failure != NULL) {
        memset(failure, 0, sizeof(*failure));
        failure->command_index = UINT32_MAX;
    }
    if (requests == NULL || parameters == NULL || commands == NULL ||
        requests_consumed == NULL || commands_emitted == NULL ||
        request_count == 0 || command_capacity == 0 ||
        command_capacity > OPENNPUX_XGRAPH_MAX_COMMANDS) {
        errno = EINVAL;
        return -1;
    }

    memset(commands, 0, command_capacity * sizeof(*commands));
    if (command_origins != NULL) {
        for (uint32_t index = 0; index < command_capacity; ++index) {
            command_origins[index] = UINT32_MAX;
        }
    }

    const uint32_t first_request_id = requests[0].command_id;
    uint32_t emitted = 0;
    for (uint32_t index = 0; index < request_count; ++index) {
        const uint64_t expected_id = (uint64_t)first_request_id + index;
        if (expected_id > UINT32_MAX ||
            requests[index].command_id != (uint32_t)expected_id) {
            errno = EINVAL;
            goto fail;
        }

        const uint32_t available = command_capacity - emitted;
        const int is_attention_projection =
            requests[index].opcode == OPENNPUX_NPU_OP_MATMUL &&
            has_attention_projection_operands(&requests[index]);
        const int is_gptq_matmul =
            requests[index].opcode == OPENNPUX_NPU_OP_MATMUL &&
            has_gptq_operands(&requests[index]) && !is_attention_projection;
        const int is_dense_multi_projection =
            requests[index].opcode == OPENNPUX_NPU_OP_MATMUL &&
            has_dense_multi_projection_operands(&requests[index]);
        const int is_dense_matmul =
            requests[index].opcode == OPENNPUX_NPU_OP_MATMUL &&
            !has_gptq_operands(&requests[index]) &&
            !is_dense_multi_projection && !is_attention_projection;
        const int is_gated_normalize =
            requests[index].opcode == OPENNPUX_NPU_OP_NORMALIZE &&
            (parameters[index].flags &
             OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET) != 0;
        const int is_shared_expert =
            requests[index].opcode == OPENNPUX_NPU_OP_EXPERT &&
            find_operand(&requests[index],
                         OPENNPUX_NPU_OPERAND_SHARED_GATE_WEIGHT) != NULL;
        const int is_gptq_expert =
            requests[index].opcode == OPENNPUX_NPU_OP_EXPERT &&
            !is_shared_expert &&
            has_gptq_expert_operands(&requests[index]);
        const int is_routed_expert =
            requests[index].opcode == OPENNPUX_NPU_OP_EXPERT &&
            !is_shared_expert && !is_gptq_expert &&
            find_operand(&requests[index], OPENNPUX_NPU_OPERAND_SECONDARY) !=
                NULL &&
            find_operand(&requests[index],
                         OPENNPUX_NPU_OPERAND_INPUT_TERTIARY) != NULL;
        const int is_dma = requests[index].opcode == OPENNPUX_NPU_OP_DMA;
        const int is_router = requests[index].opcode == OPENNPUX_NPU_OP_ROUTER;
        const int is_recurrent =
            requests[index].opcode == OPENNPUX_NPU_OP_RECURRENT_UPDATE;
        uint32_t produced = 0;
        if (is_attention_projection) {
            if (opennpux_npu_xgraph_lower_attention_projection(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, emitted, commands + emitted, available,
                    &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_gptq_matmul) {
            if (opennpux_npu_xgraph_lower_gptq_matmul(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, scratch_address, scratch_size, emitted,
                    commands + emitted, available, &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_dense_multi_projection) {
            if (opennpux_npu_xgraph_lower_dense_multi_projection(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, emitted, commands + emitted, available,
                    &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_dense_matmul) {
            if (opennpux_npu_xgraph_lower_dense_matmul(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, emitted, commands + emitted, available,
                    &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_gated_normalize) {
            if (opennpux_npu_xgraph_lower_gated_normalize(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, scratch_address, scratch_size, emitted,
                    commands + emitted, available, &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_shared_expert) {
            if (opennpux_npu_xgraph_lower_shared_expert(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, scratch_address, scratch_size, emitted,
                    commands + emitted, available, &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_gptq_expert) {
            if (opennpux_npu_xgraph_lower_gptq_expert(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, scratch_address, scratch_size, emitted,
                    commands + emitted, available, &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_routed_expert) {
            if (opennpux_npu_xgraph_lower_routed_expert(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, emitted, commands + emitted, available,
                    &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_dma) {
            if (opennpux_npu_xgraph_lower_dma(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, emitted, commands + emitted, available,
                    &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_router) {
            if (opennpux_npu_xgraph_lower_router(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, scratch_address, scratch_size, emitted,
                    commands + emitted, available, &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else if (is_recurrent) {
            if (opennpux_npu_xgraph_lower_recurrent_update(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, emitted, commands + emitted, available,
                    &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else {
            if (available == 0) {
                break;
            }
            struct opennpux_npu_xgraph_lowering_options local_options;
            memset(&local_options, 0, sizeof(local_options));
            if (options != NULL) {
                local_options = options[index];
            }
            if (requests[index].opcode == OPENNPUX_NPU_OP_TOPK &&
                local_options.topk_packed_size == 0) {
                local_options.topk_packed_address = scratch_address;
                local_options.topk_packed_size = scratch_size;
            }
            struct opennpux_xgraph_command primitive;
            if (opennpux_npu_xgraph_lower_primitive(
                    &requests[index], &parameters[index], &local_options,
                    extmem_base, extmem_size, &primitive) != 0) {
                goto fail;
            }
            primitive.command_id = emitted;
            commands[emitted] = primitive;
            produced = 1;
        }

        if (command_origins != NULL) {
            for (uint32_t output = 0; output < produced; ++output) {
                command_origins[emitted + output] = requests[index].command_id;
            }
        }
        emitted += produced;
        *requests_consumed = index + 1;
        *commands_emitted = emitted;
    }
    return 0;

fail:
    if (failure != NULL) {
        const uint32_t index = *requests_consumed;
        failure->command_index = index;
        failure->command_id = requests[index].command_id;
        failure->opcode = requests[index].opcode;
        failure->error_code = errno == 0 ? EIO : errno;
    }
    if (emitted < command_capacity) {
        memset(commands + emitted, 0,
               (command_capacity - emitted) * sizeof(*commands));
    }
    *commands_emitted = emitted;
    return -1;
}
