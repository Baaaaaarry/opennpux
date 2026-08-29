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
        if ((parameters->flags & OPENNPUX_NPU_PARAMETER_GPTQ) != 0) {
            errno = ENOTSUP;
            return -1;
        }
        command->opcode = OPENNPUX_XGRAPH_OP_TMMA;
        command->dim1 = parameters->output_features;
        command->dim2 = parameters->input_features;
        return set_operands(command, output, input, weight,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_ADD:
        command->opcode = OPENNPUX_XGRAPH_OP_TADD;
        return set_operands(command, output, input, secondary,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_MUL:
        command->opcode = OPENNPUX_XGRAPH_OP_TMUL;
        return set_operands(command, output, input, secondary,
                            extmem_base, extmem_size);
    case OPENNPUX_NPU_OP_NORMALIZE:
        command->opcode = OPENNPUX_XGRAPH_OP_TRMSNORM;
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
        if (options == NULL || options->topk_packed_size == 0 ||
            request->top_k == 0) {
            errno = EINVAL;
            return -1;
        }
        const uint64_t required =
            (uint64_t)request->rows * request->top_k * 2 * sizeof(float);
        if (required > options->topk_packed_size) {
            errno = ENOSPC;
            return -1;
        }
        const struct opennpux_npu_functional_operand packed = {
            OPENNPUX_NPU_OPERAND_OUTPUT,
            options->topk_packed_address,
            options->topk_packed_size,
            0,
        };
        command->opcode = OPENNPUX_XGRAPH_OP_TTOPK;
        command->scalar0 = request->top_k;
        return set_operands(command, &packed, input, NULL,
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
        const int is_gptq_matmul =
            requests[index].opcode == OPENNPUX_NPU_OP_MATMUL &&
            (parameters[index].flags & OPENNPUX_NPU_PARAMETER_GPTQ) != 0;
        uint32_t produced = 0;
        if (is_gptq_matmul) {
            if (opennpux_npu_xgraph_lower_gptq_matmul(
                    &requests[index], &parameters[index], extmem_base,
                    extmem_size, scratch_address, scratch_size, emitted,
                    commands + emitted, available, &produced) != 0) {
                if (errno == ENOSPC && emitted != 0) {
                    break;
                }
                goto fail;
            }
        } else {
            if (available == 0) {
                break;
            }
            const struct opennpux_npu_xgraph_lowering_options *command_options =
                options == NULL ? NULL : &options[index];
            struct opennpux_xgraph_command primitive;
            if (opennpux_npu_xgraph_lower_primitive(
                    &requests[index], &parameters[index], command_options,
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
