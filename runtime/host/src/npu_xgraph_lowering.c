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
