#include "opennpux/npu_functional_materializer.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int
validate_program_commands(
    const struct opennpux_npu_invocation_header *header,
    const struct opennpux_npu_command *commands,
    const struct opennpux_npu_tensor_plan *tensor_plan)
{
    if (header == NULL || commands == NULL || tensor_plan == NULL ||
        tensor_plan->header == NULL || header->command_count == 0 ||
        header->command_count > tensor_plan->header->command_count) {
        errno = EINVAL;
        return -1;
    }

    uint8_t *seen = calloc(tensor_plan->header->command_count, sizeof(*seen));
    if (seen == NULL) {
        return -1;
    }
    const uint64_t runtime_shape = commands[0].runtime_shape;
    int result = 0;
    for (uint32_t index = 0; index < header->command_count; ++index) {
        const uint32_t command_id = commands[index].command_id;
        if (command_id >= tensor_plan->header->command_count ||
            tensor_plan->commands[command_id].command_id != command_id) {
            errno = ERANGE;
            result = -1;
            break;
        }
        if (seen[command_id] != 0) {
            errno = EEXIST;
            result = -1;
            break;
        }
        if (commands[index].runtime_shape != runtime_shape) {
            errno = EINVAL;
            result = -1;
            break;
        }
        seen[command_id] = 1;
    }
    free(seen);
    return result;
}

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
        OPENNPUX_NPU_OPERAND_OUTPUT_QUATERNARY,
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
    request->kv_heads = parameters->kv_head_count;
    request->vocabulary_size = parameters->input_features;
    request->epsilon = 1.0e-5f;
    if (parameters->opcode == OPENNPUX_NPU_OP_NORMALIZE &&
        parameters->quantized_zero_bias != 0) {
        uint32_t epsilon_bits = parameters->quantized_zero_bias;
        memcpy(&request->epsilon, &epsilon_bits, sizeof(request->epsilon));
        if (!isfinite(request->epsilon) || !(request->epsilon > 0.0f)) {
            errno = EINVAL;
            return -1;
        }
    }
    request->rope_theta =
        parameters->opcode == OPENNPUX_NPU_OP_ROPE &&
                parameters->quantization_group_size != 0
            ? (float)parameters->quantization_group_size
            : 10000.0f;
    const struct opennpux_npu_tensor_view *shape_view =
        command->opcode == OPENNPUX_NPU_OP_DMA ||
                command->opcode == OPENNPUX_NPU_OP_TOPK ?
            &views->inputs[0] : &views->outputs[0];
    if (derive_shape(shape_view, &request->rows,
                     &request->features) != 0) {
        return -1;
    }
    if (command->opcode == OPENNPUX_NPU_OP_TOPK ||
        command->opcode == OPENNPUX_NPU_OP_ROUTER) {
        request->top_k = views->outputs[0].dimensions[
            views->outputs[0].rank - 1];
    }
    if (command->opcode == OPENNPUX_NPU_OP_DMA) {
        if (views->outputs[0].rank != 5 ||
            views->outputs[0].dimensions[0] != 2) {
            errno = EINVAL;
            return -1;
        }
        request->kv_length = views->outputs[0].dimensions[2];
        request->kv_heads = views->outputs[0].dimensions[3];
    } else if (command->opcode == OPENNPUX_NPU_OP_ATTENTION &&
               views->inputs[1].rank == 5) {
        request->kv_length = views->inputs[1].dimensions[2];
        request->kv_heads = views->inputs[1].dimensions[3];
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
        if (command->opcode == OPENNPUX_NPU_OP_TOPK && index == 0) {
            role = OPENNPUX_NPU_OPERAND_OUTPUT_INDICES;
        } else if (command->opcode == OPENNPUX_NPU_OP_ROUTER && index == 0 &&
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
    const struct opennpux_npu_command *commands =
        (const struct opennpux_npu_command *)(
            (const uint8_t *)submission + header->command_offset);
    if (validate_program_commands(header, commands, tensor_plan) != 0) {
        return -1;
    }
    memset(program, 0, sizeof(*program));
    program->submission = submission;
    program->submission_size = submission_size;
    program->submission_address = submission_address;
    program->header = header;
    program->commands = commands;
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

int
opennpux_npu_functional_gptq_operands(
    uint32_t slot_id,
    const struct opennpux_npu_functional_gptq_views *views,
    struct opennpux_npu_functional_operand *operands,
    uint32_t operand_capacity, uint32_t *operand_count)
{
    static const uint32_t default_roles[] = {
        OPENNPUX_NPU_OPERAND_QWEIGHT, OPENNPUX_NPU_OPERAND_QZEROS,
        OPENNPUX_NPU_OPERAND_SCALES, OPENNPUX_NPU_OPERAND_G_IDX,
    };
    static const uint32_t q_roles[] = {
        OPENNPUX_NPU_OPERAND_Q_QWEIGHT, OPENNPUX_NPU_OPERAND_Q_QZEROS,
        OPENNPUX_NPU_OPERAND_Q_SCALES, OPENNPUX_NPU_OPERAND_Q_G_IDX,
    };
    static const uint32_t k_roles[] = {
        OPENNPUX_NPU_OPERAND_K_QWEIGHT, OPENNPUX_NPU_OPERAND_K_QZEROS,
        OPENNPUX_NPU_OPERAND_K_SCALES, OPENNPUX_NPU_OPERAND_K_G_IDX,
    };
    static const uint32_t v_roles[] = {
        OPENNPUX_NPU_OPERAND_V_QWEIGHT, OPENNPUX_NPU_OPERAND_V_QZEROS,
        OPENNPUX_NPU_OPERAND_V_SCALES, OPENNPUX_NPU_OPERAND_V_G_IDX,
    };
    static const uint32_t gate_roles[] = {
        OPENNPUX_NPU_OPERAND_GATE_QWEIGHT,
        OPENNPUX_NPU_OPERAND_GATE_QZEROS,
        OPENNPUX_NPU_OPERAND_GATE_SCALES,
        OPENNPUX_NPU_OPERAND_GATE_G_IDX,
    };
    static const uint32_t up_roles[] = {
        OPENNPUX_NPU_OPERAND_UP_QWEIGHT,
        OPENNPUX_NPU_OPERAND_UP_QZEROS,
        OPENNPUX_NPU_OPERAND_UP_SCALES,
        OPENNPUX_NPU_OPERAND_UP_G_IDX,
    };
    static const uint32_t down_roles[] = {
        OPENNPUX_NPU_OPERAND_DOWN_QWEIGHT,
        OPENNPUX_NPU_OPERAND_DOWN_QZEROS,
        OPENNPUX_NPU_OPERAND_DOWN_SCALES,
        OPENNPUX_NPU_OPERAND_DOWN_G_IDX,
    };
    if (views == NULL || operands == NULL || operand_count == NULL ||
        operand_capacity < 3) {
        errno = EINVAL;
        return -1;
    }
    const uint32_t *roles = default_roles;
    switch (slot_id) {
      case OPENNPUX_NPU_WEIGHT_SLOT_Q_PROJ: roles = q_roles; break;
      case OPENNPUX_NPU_WEIGHT_SLOT_K_PROJ: roles = k_roles; break;
      case OPENNPUX_NPU_WEIGHT_SLOT_V_PROJ: roles = v_roles; break;
      case OPENNPUX_NPU_WEIGHT_SLOT_GATE_PROJ: roles = gate_roles; break;
      case OPENNPUX_NPU_WEIGHT_SLOT_UP_PROJ: roles = up_roles; break;
      case OPENNPUX_NPU_WEIGHT_SLOT_DOWN_PROJ: roles = down_roles; break;
      case OPENNPUX_NPU_WEIGHT_SLOT_DEFAULT:
      case OPENNPUX_NPU_WEIGHT_SLOT_O_PROJ:
      case OPENNPUX_NPU_WEIGHT_SLOT_QKV_PROJ:
        break;
      default:
        errno = EINVAL;
        return -1;
    }
    const uint64_t addresses[] = {
        views->qweight_address, views->qzeros_address,
        views->scales_address, views->g_idx_address,
    };
    const uint64_t sizes[] = {
        views->qweight_size, views->qzeros_size,
        views->scales_size, views->g_idx_size,
    };
    const uint32_t count = views->g_idx_address == 0 ? 3 : 4;
    if (operand_capacity < count) {
        errno = ENOSPC;
        return -1;
    }
    for (uint32_t index = 0; index < count; ++index) {
        if (addresses[index] > UINT32_MAX || sizes[index] == 0 ||
            sizes[index] > UINT32_MAX) {
            errno = addresses[index] > UINT32_MAX || sizes[index] > UINT32_MAX ?
                EOVERFLOW : EINVAL;
            return -1;
        }
        operands[index] = (struct opennpux_npu_functional_operand){
            .role = roles[index],
            .address = (uint32_t)addresses[index],
            .byte_size = (uint32_t)sizes[index],
            .reserved = 0,
        };
    }
    *operand_count = count;
    return 0;
}
