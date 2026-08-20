#ifndef OPENNPUX_NPU_FUNCTIONAL_MATERIALIZER_H
#define OPENNPUX_NPU_FUNCTIONAL_MATERIALIZER_H

#include "opennpux/npu_functional_request.h"
#include "opennpux/npu_submission.h"
#include "opennpux/npu_tensor_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

struct opennpux_npu_functional_program {
    const uint8_t *submission;
    size_t submission_size;
    uint64_t submission_address;
    const struct opennpux_npu_invocation_header *header;
    const struct opennpux_npu_command *commands;
    const struct opennpux_npu_tensor_plan *tensor_plan;
    struct opennpux_npu_tensor_plan_runtime runtime;
    struct opennpux_npu_tensor_plan_memory memory;
};

int opennpux_npu_functional_request_add_operand(
    struct opennpux_npu_functional_request *request, uint32_t role,
    uint64_t address, uint64_t byte_size);
int opennpux_npu_functional_request_materialize(
    const struct opennpux_npu_command *command,
    const struct opennpux_npu_operator_parameters *parameters,
    const struct opennpux_npu_command_tensor_views *views,
    uint64_t parameter_address,
    const struct opennpux_npu_functional_operand *extra_operands,
    uint32_t extra_operand_count,
    struct opennpux_npu_functional_request *request);
int opennpux_npu_functional_program_init(
    struct opennpux_npu_functional_program *program,
    const void *submission, size_t submission_size,
    uint64_t submission_address,
    const struct opennpux_npu_tensor_plan *tensor_plan,
    const struct opennpux_npu_tensor_plan_memory *memory);
int opennpux_npu_functional_program_materialize(
    const struct opennpux_npu_functional_program *program,
    uint32_t command_index,
    const struct opennpux_npu_functional_operand *extra_operands,
    uint32_t extra_operand_count,
    struct opennpux_npu_functional_request *request);

#ifdef __cplusplus
}
#endif

#endif
