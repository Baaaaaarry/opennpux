#include "opennpux/npu_xgraph_codegen_ffi.h"

#include "opennpux/npu_functional_request.h"
#include "opennpux/npu_submission.h"
#include "opennpux/npu_xgraph_lowering.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>

static int
add_operand(struct opennpux_npu_functional_request *request, uint32_t role,
            uint32_t base, uint32_t offset, uint32_t byte_size)
{
    if (request->operand_count >= OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS ||
        offset > UINT32_MAX - base || byte_size == 0) {
        errno = EOVERFLOW;
        return -1;
    }
    request->operands[request->operand_count++] =
        (struct opennpux_npu_functional_operand){
            .role = role,
            .address = base + offset,
            .byte_size = byte_size,
            .reserved = 0,
        };
    return 0;
}

int
opennpux_xgraph_codegen_dense_matmul(
    const struct opennpux_xgraph_dense_spec *spec,
    struct opennpux_xgraph_command *commands, uint32_t command_capacity,
    uint32_t *command_count)
{
    if (command_count != NULL) {
        *command_count = 0;
    }
    if (spec == NULL || commands == NULL || command_count == NULL ||
        spec->struct_size != sizeof(*spec) || spec->extmem_size == 0 ||
        spec->rows == 0 || spec->input_features == 0 ||
        spec->output_features == 0 || spec->transpose_rhs > 1) {
        errno = EINVAL;
        return -1;
    }
    struct opennpux_npu_functional_request request;
    memset(&request, 0, sizeof(request));
    request.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
    request.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
    request.struct_size = sizeof(request);
    request.opcode = OPENNPUX_NPU_OP_MATMUL;
    request.command_id = spec->first_command_id;
    request.rows = spec->rows;
    request.features = spec->input_features;
    if (add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT,
                    spec->extmem_base, spec->lhs_offset,
                    spec->lhs_bytes) != 0 ||
        add_operand(&request, OPENNPUX_NPU_OPERAND_WEIGHT,
                    spec->extmem_base, spec->rhs_offset,
                    spec->rhs_bytes) != 0 ||
        add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT,
                    spec->extmem_base, spec->output_offset,
                    spec->output_bytes) != 0) {
        return -1;
    }
    struct opennpux_npu_operator_parameters parameters;
    memset(&parameters, 0, sizeof(parameters));
    parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
    parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
    parameters.struct_size = sizeof(parameters);
    parameters.opcode = OPENNPUX_NPU_OP_MATMUL;
    parameters.input_features = spec->input_features;
    parameters.output_features = spec->output_features;
    return opennpux_npu_xgraph_lower_dense_matmul_layout(
        &request, &parameters, spec->transpose_rhs, spec->extmem_base,
        spec->extmem_size, spec->first_command_id, commands,
        command_capacity, command_count);
}
