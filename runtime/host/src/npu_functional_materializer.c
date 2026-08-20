#include "opennpux/npu_functional_materializer.h"

#include <errno.h>
#include <limits.h>
#include <string.h>

static int
add_view(struct opennpux_npu_functional_request *request, uint32_t role,
         const struct opennpux_npu_tensor_view *view)
{
    return opennpux_npu_functional_request_add_operand(
        request, role, view->address, view->size);
}

int
opennpux_npu_functional_request_add_operand(
    struct opennpux_npu_functional_request *request, uint32_t role,
    uint64_t address, uint64_t byte_size)
{
    if (request == NULL || role == 0 || address > UINT32_MAX ||
        byte_size == 0 || byte_size > UINT32_MAX ||
        request->operand_count >= OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS) {
        errno = address > UINT32_MAX || byte_size > UINT32_MAX ? EOVERFLOW :
            EINVAL;
        return -1;
    }
    for (uint32_t index = 0; index < request->operand_count; ++index) {
        if (request->operands[index].role == role) {
            errno = EEXIST;
            return -1;
        }
    }
    struct opennpux_npu_functional_operand *operand =
        &request->operands[request->operand_count++];
    operand->role = role;
    operand->address = (uint32_t)address;
    operand->byte_size = (uint32_t)byte_size;
    operand->reserved = 0;
    return 0;
}

static int
derive_shape(const struct opennpux_npu_tensor_view *view,
             uint32_t *rows, uint32_t *features)
{
    if (view->rank == 0 || view->rank > OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK) {
        errno = EINVAL;
        return -1;
    }
    uint64_t row_count = 1;
    for (uint32_t index = 0; index + 1 < view->rank; ++index) {
        if (view->dimensions[index] == 0 ||
            row_count > UINT32_MAX / view->dimensions[index]) {
            errno = EOVERFLOW;
            return -1;
        }
        row_count *= view->dimensions[index];
    }
    if (view->dimensions[view->rank - 1] == 0) {
        errno = EINVAL;
        return -1;
    }
    *rows = (uint32_t)row_count;
    *features = view->dimensions[view->rank - 1];
    return 0;
}

int
opennpux_npu_functional_request_materialize(
    const struct opennpux_npu_command *command,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_command_tensor_views *views,
    uint64_t parameter_address,
    const struct opennpux_npu_functional_operand *extra_operands,
    uint32_t extra_operand_count,
    struct opennpux_npu_functional_request *request)
{
    static const uint32_t input_roles[] = {
        OPENNPUX_NPU_OPERAND_INPUT,
        OPENNPUX_NPU_OPERAND_SECONDARY,
        OPENNPUX_NPU_OPERAND_INPUT_TERTIARY,
        OPENNPUX_NPU_OPERAND_INPUT_QUATERNARY,
    };
    static const uint32_t output_roles[] = {
        OPENNPUX_NPU_OPERAND_OUTPUT,
        OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY,
        OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY,
    };
    if (command == NULL || parameters == NULL || views == NULL ||
        request == NULL || command->command_id != views->command_id ||
        command->opcode != parameters->opcode ||
        parameters->magic != OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC ||
        parameters->version != OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION ||
        parameters->struct_size != sizeof(*parameters) ||
        views->input_count == 0 || views->output_count == 0 ||
        views->input_count > OPENNPUX_NPU_TENSOR_PLAN_MAX_INPUTS ||
        views->output_count > OPENNPUX_NPU_TENSOR_PLAN_MAX_OUTPUTS ||
        (extra_operand_count != 0 && extra_operands == NULL) ||
        parameter_address > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }

    memset(request, 0, sizeof(*request));
    request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
    request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
    request->struct_size = sizeof(*request);
    request->opcode = command->opcode;
    request->command_id = command->command_id;
    request->parameter_address = (uint32_t)parameter_address;
    request->parameter_size = sizeof(*parameters);
    request->heads = parameters->head_count;
    request->head_dim = parameters->head_dim;
    request->top_k = parameters->output_features;
    request->vocabulary_size = parameters->input_features;
    request->epsilon = 1.0e-5f;
    request->rope_theta = 10000.0f;
    if (derive_shape(&views->outputs[0], &request->rows,
                     &request->features) != 0) {
        return -1;
    }

    for (uint32_t index = 0; index < views->input_count; ++index) {
        uint32_t role = input_roles[index];
        if (command->opcode == OPENNPUX_NPU_OP_EMBED && index == 0) {
            role = OPENNPUX_NPU_OPERAND_INPUT_INDICES;
        }
        if (add_view(request, role, &views->inputs[index]) != 0) {
            return -1;
        }
    }
    for (uint32_t index = 0; index < views->output_count; ++index) {
        uint32_t role = output_roles[index];
        if ((command->opcode == OPENNPUX_NPU_OP_TOPK ||
             command->opcode == OPENNPUX_NPU_OP_ROUTER) && index == 0 &&
            views->output_count > 1) {
            role = OPENNPUX_NPU_OPERAND_OUTPUT_INDICES;
        } else if ((command->opcode == OPENNPUX_NPU_OP_TOPK ||
                    command->opcode == OPENNPUX_NPU_OP_ROUTER) &&
                   index == 1) {
            role = OPENNPUX_NPU_OPERAND_OUTPUT;
        }
        if (add_view(request, role, &views->outputs[index]) != 0) {
            return -1;
        }
    }
    for (uint32_t index = 0; index < extra_operand_count; ++index) {
        if (opennpux_npu_functional_request_add_operand(
                request, extra_operands[index].role,
                extra_operands[index].address,
                extra_operands[index].byte_size) != 0) {
            return -1;
        }
    }
    return 0;
}

int
opennpux_npu_functional_program_init(
    struct opennpux_npu_functional_program *program,
    const void *submission, size_t submission_size,
    uint64_t submission_address,
    const struct opennpux_npu_tensor_plan *tensor_plan,
    const struct opennpux_npu_tensor_plan_memory *memory)
{
    if (program == NULL || submission == NULL || tensor_plan == NULL ||
        tensor_plan->header == NULL || memory == NULL ||
        submission_address > UINT32_MAX ||
        opennpux_npu_submission_validate(submission, submission_size) != 0) {
        if (errno == 0) {
            errno = EINVAL;
        }
        return -1;
    }
    const struct opennpux_npu_invocation_header *header = submission;
    if (header->command_count > tensor_plan->header->command_count) {
        errno = EINVAL;
        return -1;
    }
    memset(program, 0, sizeof(*program));
    program->submission = submission;
    program->submission_size = submission_size;
    program->submission_address = submission_address;
    program->header = header;
    program->commands = (const struct opennpux_npu_command *)(
        program->submission + header->command_offset);
    program->tensor_plan = tensor_plan;
    program->runtime.batch_size =
        program->commands[0].runtime_shape & OPENNPUX_NPU_RUNTIME_FIELD_MASK;
    program->runtime.sequence_length =
        (program->commands[0].runtime_shape >>
         OPENNPUX_NPU_RUNTIME_SEQUENCE_SHIFT) &
        OPENNPUX_NPU_RUNTIME_FIELD_MASK;
    program->runtime.kv_length =
        (program->commands[0].runtime_shape >>
         OPENNPUX_NPU_RUNTIME_KV_SHIFT) &
        OPENNPUX_NPU_RUNTIME_FIELD_MASK;
    program->runtime.active_experts =
        (program->commands[0].runtime_shape >>
         OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT) &
        OPENNPUX_NPU_RUNTIME_FIELD_MASK;
    program->memory = *memory;
    return 0;
}

int
opennpux_npu_functional_program_materialize(
    const struct opennpux_npu_functional_program *program,
    uint32_t command_index,
    const struct opennpux_npu_functional_operand *extra_operands,
    uint32_t extra_operand_count,
    struct opennpux_npu_functional_request *request)
{
    if (program == NULL || program->header == NULL ||
        command_index >= program->header->command_count) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_command *command =
        &program->commands[command_index];
    if (command->command_id >= program->tensor_plan->header->command_count ||
        command->parameter_size !=
            sizeof(struct opennpux_npu_operator_parameters)) {
        errno = EINVAL;
        return -1;
    }
    const uint64_t parameter_offset = program->header->parameter_offset +
        command->parameter_offset;
    if (parameter_offset > program->submission_size ||
        command->parameter_size > program->submission_size - parameter_offset ||
        program->submission_address > UINT32_MAX - parameter_offset) {
        errno = EOVERFLOW;
        return -1;
    }
    const struct opennpux_npu_operator_parameters *parameters =
        (const struct opennpux_npu_operator_parameters *)(
            program->submission + parameter_offset);
    struct opennpux_npu_command_tensor_views views;
    if (opennpux_npu_tensor_plan_resolve_command(
            program->tensor_plan, command->command_id, &program->runtime,
            &program->memory, &views) != 0) {
        return -1;
    }
    return opennpux_npu_functional_request_materialize(
        command, parameters, &views,
        program->submission_address + parameter_offset, extra_operands,
        extra_operand_count, request);
}
