#include "opennpux/npu_functional_materializer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}

static struct opennpux_npu_tensor_view
view(uint32_t id, uint64_t address, uint64_t size, uint32_t rows,
     uint32_t features)
{
    struct opennpux_npu_tensor_view result = {
        .tensor_id = id,
        .storage = OPENNPUX_NPU_TENSOR_SCRATCH,
        .data_type = OPENNPUX_NPU_DTYPE_FLOAT32,
        .rank = 2,
        .dimensions = {rows, features},
        .address = address,
        .size = size,
    };
    return result;
}

int
main(void)
{
    struct opennpux_npu_command command = {
        .command_id = 7,
        .opcode = OPENNPUX_NPU_OP_MATMUL,
    };
    struct opennpux_npu_operator_parameters parameters = {
        .magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC,
        .version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION,
        .struct_size = sizeof(parameters),
        .opcode = OPENNPUX_NPU_OP_MATMUL,
        .input_features = 8,
        .output_features = 12,
    };
    struct opennpux_npu_command_tensor_views views = {
        .command_id = 7,
        .input_count = 1,
        .output_count = 3,
    };
    views.inputs[0] = view(1, 0x20001000, 128, 4, 8);
    views.outputs[0] = view(2, 0x20002000, 192, 4, 12);
    views.outputs[1] = view(3, 0x20003000, 192, 4, 12);
    views.outputs[2] = view(4, 0x20004000, 192, 4, 12);
    const struct opennpux_npu_functional_operand weights[] = {
        {OPENNPUX_NPU_OPERAND_QWEIGHT, 0x20300000, 48, 0},
        {OPENNPUX_NPU_OPERAND_QZEROS, 0x20300100, 8, 0},
        {OPENNPUX_NPU_OPERAND_SCALES, 0x20300200, 24, 0},
    };
    struct opennpux_npu_functional_request request;
    check(opennpux_npu_functional_request_materialize(
              &command, &parameters, &views, 0x20000100, weights, 3,
              &request) == 0,
          "materialize qkv request");
    check(request.rows == 4 && request.features == 12,
          "derive output shape");
    check(request.operand_count == 7 &&
              request.operands[1].role == OPENNPUX_NPU_OPERAND_OUTPUT &&
              request.operands[2].role ==
                  OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY &&
              request.operands[3].role ==
                  OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY,
          "preserve all qkv outputs");

    command.opcode = OPENNPUX_NPU_OP_ROUTER;
    parameters.opcode = OPENNPUX_NPU_OP_ROUTER;
    parameters.output_features = 8;
    views.output_count = 2;
    check(opennpux_npu_functional_request_materialize(
              &command, &parameters, &views, 0x20000100, NULL, 0,
              &request) == 0,
          "materialize router request");
    check(request.operands[1].role == OPENNPUX_NPU_OPERAND_OUTPUT_INDICES &&
              request.operands[2].role == OPENNPUX_NPU_OPERAND_OUTPUT,
          "map router indices and weights");
    puts("npu_functional_materializer=PASS");
    return 0;
}
