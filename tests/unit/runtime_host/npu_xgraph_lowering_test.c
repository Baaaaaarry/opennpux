#include "opennpux/npu_xgraph_lowering.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#define EXTMEM_BASE UINT32_C(0x20000000)
#define EXTMEM_SIZE UINT32_C(0x00800000)

static void
initialize(struct opennpux_npu_functional_request *request,
           struct opennpux_npu_operator_parameters *parameters,
           uint32_t opcode)
{
    memset(request, 0, sizeof(*request));
    request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
    request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
    request->struct_size = sizeof(*request);
    request->opcode = opcode;
    request->command_id = opcode;
    request->rows = 2;
    request->features = 4;
    request->epsilon = 1.0e-5f;

    memset(parameters, 0, sizeof(*parameters));
    parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
    parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
    parameters->struct_size = sizeof(*parameters);
    parameters->opcode = opcode;
    parameters->input_features = 4;
    parameters->output_features = 3;
}

static void
add_operand(struct opennpux_npu_functional_request *request,
            uint32_t role, uint32_t offset, uint32_t bytes)
{
    assert(request->operand_count < OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS);
    struct opennpux_npu_functional_operand *operand =
        &request->operands[request->operand_count++];
    operand->role = role;
    operand->address = EXTMEM_BASE + offset;
    operand->byte_size = bytes;
}

static void
test_direct_primitives(void)
{
    const struct {
        uint32_t generic_opcode;
        uint32_t xgraph_opcode;
        uint32_t source_count;
    } cases[] = {
        {OPENNPUX_NPU_OP_MATMUL, OPENNPUX_XGRAPH_OP_TMMA, 2},
        {OPENNPUX_NPU_OP_ADD, OPENNPUX_XGRAPH_OP_TADD, 2},
        {OPENNPUX_NPU_OP_MUL, OPENNPUX_XGRAPH_OP_TMUL, 2},
        {OPENNPUX_NPU_OP_NORMALIZE, OPENNPUX_XGRAPH_OP_TRMSNORM, 2},
        {OPENNPUX_NPU_OP_SOFTMAX, OPENNPUX_XGRAPH_OP_TSOFTMAX, 1},
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        struct opennpux_npu_functional_request request;
        struct opennpux_npu_operator_parameters parameters;
        struct opennpux_xgraph_command command;
        initialize(&request, &parameters, cases[index].generic_opcode);
        add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT, 0x1000, 32);
        if (cases[index].source_count == 2) {
            add_operand(&request,
                        cases[index].generic_opcode == OPENNPUX_NPU_OP_MATMUL ||
                                cases[index].generic_opcode ==
                                    OPENNPUX_NPU_OP_NORMALIZE
                            ? OPENNPUX_NPU_OPERAND_WEIGHT
                            : OPENNPUX_NPU_OPERAND_SECONDARY,
                        0x2000, 64);
        }
        add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x3000, 32);
        assert(opennpux_npu_xgraph_lower_primitive(
                   &request, &parameters, NULL, EXTMEM_BASE, EXTMEM_SIZE,
                   &command) == 0);
        assert(command.opcode == cases[index].xgraph_opcode);
        assert(command.source0_offset == 0x1000);
        assert(command.destination_offset == 0x3000);
    }
}

static void
test_semantic_options(void)
{
    struct opennpux_npu_functional_request request;
    struct opennpux_npu_operator_parameters parameters;
    struct opennpux_xgraph_command command;
    struct opennpux_npu_xgraph_lowering_options options = {
        OPENNPUX_NPU_XGRAPH_ROPE_HALF_SPLIT,
        OPENNPUX_NPU_XGRAPH_ACTIVATION_SILU,
        EXTMEM_BASE + 0x5000,
        64,
    };

    initialize(&request, &parameters, OPENNPUX_NPU_OP_ROPE);
    add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT, 0x1000, 32);
    add_operand(&request, OPENNPUX_NPU_OPERAND_SECONDARY, 0x2000, 64);
    add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x3000, 32);
    assert(opennpux_npu_xgraph_lower_primitive(
               &request, &parameters, &options, EXTMEM_BASE, EXTMEM_SIZE,
               &command) == 0);
    assert(command.opcode == OPENNPUX_XGRAPH_OP_TROPE);
    assert(command.scalar0 == OPENNPUX_NPU_XGRAPH_ROPE_HALF_SPLIT);

    initialize(&request, &parameters, OPENNPUX_NPU_OP_ACTIVATION);
    add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT, 0x1000, 32);
    add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x3000, 32);
    assert(opennpux_npu_xgraph_lower_primitive(
               &request, &parameters, &options, EXTMEM_BASE, EXTMEM_SIZE,
               &command) == 0);
    assert(command.opcode == OPENNPUX_XGRAPH_OP_TSILU);

    initialize(&request, &parameters, OPENNPUX_NPU_OP_TOPK);
    request.top_k = 2;
    add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT, 0x1000, 32);
    add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT_INDICES, 0x3000, 16);
    assert(opennpux_npu_xgraph_lower_primitive(
               &request, &parameters, &options, EXTMEM_BASE, EXTMEM_SIZE,
               &command) == 0);
    assert(command.opcode == OPENNPUX_XGRAPH_OP_TTOPK);
    assert(command.destination_offset == 0x5000);
}

static void
test_embed_and_rejections(void)
{
    struct opennpux_npu_functional_request request;
    struct opennpux_npu_operator_parameters parameters;
    struct opennpux_xgraph_command command;

    initialize(&request, &parameters, OPENNPUX_NPU_OP_EMBED);
    parameters.input_features = 16;
    add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT_INDICES, 0x1000, 8);
    add_operand(&request, OPENNPUX_NPU_OPERAND_WEIGHT, 0x2000, 256);
    add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x3000, 32);
    assert(opennpux_npu_xgraph_lower_primitive(
               &request, &parameters, NULL, EXTMEM_BASE, EXTMEM_SIZE,
               &command) == 0);
    assert(command.opcode == OPENNPUX_XGRAPH_OP_TGATHER);
    assert(command.scalar0 == 16);

    initialize(&request, &parameters, OPENNPUX_NPU_OP_MATMUL);
    parameters.flags = OPENNPUX_NPU_PARAMETER_GPTQ;
    add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT, 0x1000, 32);
    add_operand(&request, OPENNPUX_NPU_OPERAND_WEIGHT, 0x2000, 64);
    add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x3000, 32);
    errno = 0;
    assert(opennpux_npu_xgraph_lower_primitive(
               &request, &parameters, NULL, EXTMEM_BASE, EXTMEM_SIZE,
               &command) == -1);
    assert(errno == ENOTSUP);

    initialize(&request, &parameters, OPENNPUX_NPU_OP_ATTENTION);
    errno = 0;
    assert(opennpux_npu_xgraph_lower_primitive(
               &request, &parameters, NULL, EXTMEM_BASE, EXTMEM_SIZE,
               &command) == -1);
    assert(errno == ENOTSUP);
}

static void
test_sequence_lowering(void)
{
    enum { count = 3 };
    struct opennpux_npu_functional_request requests[count];
    struct opennpux_npu_operator_parameters parameters[count];
    struct opennpux_npu_xgraph_lowering_options options[count];
    struct opennpux_xgraph_command commands[count];
    struct opennpux_npu_xgraph_lowering_failure failure;
    const uint32_t opcodes[count] = {
        OPENNPUX_NPU_OP_ADD,
        OPENNPUX_NPU_OP_MUL,
        OPENNPUX_NPU_OP_SOFTMAX,
    };
    memset(options, 0, sizeof(options));
    for (uint32_t index = 0; index < count; ++index) {
        initialize(&requests[index], &parameters[index], opcodes[index]);
        requests[index].command_id = index;
        add_operand(&requests[index], OPENNPUX_NPU_OPERAND_INPUT,
                    0x1000 + index * 0x100, 32);
        if (opcodes[index] != OPENNPUX_NPU_OP_SOFTMAX) {
            add_operand(&requests[index], OPENNPUX_NPU_OPERAND_SECONDARY,
                        0x2000 + index * 0x100, 32);
        }
        add_operand(&requests[index], OPENNPUX_NPU_OPERAND_OUTPUT,
                    0x3000 + index * 0x100, 32);
    }
    assert(opennpux_npu_xgraph_lower_sequence(
               requests, parameters, options, count, EXTMEM_BASE, EXTMEM_SIZE,
               commands, count, &failure) == 0);
    assert(commands[0].opcode == OPENNPUX_XGRAPH_OP_TADD);
    assert(commands[1].opcode == OPENNPUX_XGRAPH_OP_TMUL);
    assert(commands[2].opcode == OPENNPUX_XGRAPH_OP_TSOFTMAX);
    assert(failure.command_index == UINT32_MAX);

    requests[1].opcode = OPENNPUX_NPU_OP_ATTENTION;
    parameters[1].opcode = OPENNPUX_NPU_OP_ATTENTION;
    errno = 0;
    assert(opennpux_npu_xgraph_lower_sequence(
               requests, parameters, options, count, EXTMEM_BASE, EXTMEM_SIZE,
               commands, count, &failure) == -1);
    assert(errno == ENOTSUP);
    assert(failure.command_index == 1);
    assert(failure.command_id == 1);
    assert(failure.opcode == OPENNPUX_NPU_OP_ATTENTION);
    assert(failure.error_code == ENOTSUP);
    assert(commands[2].opcode == 0);

    requests[1].opcode = OPENNPUX_NPU_OP_MUL;
    parameters[1].opcode = OPENNPUX_NPU_OP_MUL;
    requests[1].command_id = 7;
    errno = 0;
    assert(opennpux_npu_xgraph_lower_sequence(
               requests, parameters, options, count, EXTMEM_BASE, EXTMEM_SIZE,
               commands, count, &failure) == -1);
    assert(errno == EINVAL);
    assert(failure.command_index == 1);
    assert(failure.command_id == 7);
}

int
main(void)
{
    test_direct_primitives();
    test_semantic_options();
    test_embed_and_rejections();
    test_sequence_lowering();
    puts("NPU XGraph primitive lowering test: PASS");
    return 0;
}
