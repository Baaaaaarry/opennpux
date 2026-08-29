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

static void
test_gptq_tiled_lowering(void)
{
    struct opennpux_npu_functional_request request;
    struct opennpux_npu_operator_parameters parameters;
    struct opennpux_xgraph_command commands[12];
    uint32_t command_count = 0;
    initialize(&request, &parameters, OPENNPUX_NPU_OP_MATMUL);
    request.command_id = 0;
    request.rows = 3;
    request.features = 32;
    parameters.flags = OPENNPUX_NPU_PARAMETER_GPTQ;
    parameters.input_features = 32;
    parameters.output_features = 18;
    parameters.quantization_bits = 4;
    parameters.quantization_group_size = 16;
    parameters.quantized_zero_bias = 1;
    parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT16;
    add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT, 0x1000,
                3 * 32 * 4);
    add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x2000,
                3 * 18 * 4);
    add_operand(&request, OPENNPUX_NPU_OPERAND_QWEIGHT, 0x3000,
                4 * 18 * 4);
    add_operand(&request, OPENNPUX_NPU_OPERAND_QZEROS, 0x5000,
                2 * 3 * 4);
    add_operand(&request, OPENNPUX_NPU_OPERAND_SCALES, 0x6000,
                2 * 18 * 2);
    add_operand(&request, OPENNPUX_NPU_OPERAND_G_IDX, 0x7000, 32 * 4);

    assert(opennpux_npu_xgraph_lower_gptq_matmul(
               &request, &parameters, EXTMEM_BASE, EXTMEM_SIZE,
               EXTMEM_BASE + 0x10000, 32 * 8 * 4, 100, commands, 12,
               &command_count) == 0);
    assert(command_count == 12);
    assert(commands[0].opcode == OPENNPUX_XGRAPH_OP_TDEQUANT);
    assert(commands[0].command_id == 100);
    assert(commands[0].destination_offset == 0x10000);
    assert(commands[0].source0_offset == 0x3000);
    assert(commands[0].source1_offset == 0x5000);
    assert(commands[0].dim0 == 1 && commands[0].dim1 == 8 &&
           commands[0].dim2 == 32);
    assert(commands[0].scalar0 == UINT32_C(0x01010010));
    assert(commands[0].reserved[0] == 0x6000);
    assert(commands[0].reserved[1] == 0x7000);
    assert(commands[0].reserved[2] == 18 * 4);
    assert(commands[0].reserved[3] == 3 * 4);
    assert(commands[0].reserved[4] == 18 * 2);
    assert(commands[1].opcode == OPENNPUX_XGRAPH_OP_TMMA);
    assert(commands[1].source0_offset == 0x1000);
    assert(commands[1].source1_offset == 0x10000);
    assert(commands[1].destination_offset == 0x2000);
    assert(commands[3].source0_offset == 0x1000 + 2 * 32 * 4);
    assert(commands[3].destination_offset == 0x2000 + 2 * 18 * 4);

    assert(commands[4].opcode == OPENNPUX_XGRAPH_OP_TDEQUANT);
    assert(commands[4].source0_offset == 0x3000 + 8 * 4);
    assert(commands[4].source1_offset == 0x5000 + 4);
    assert(commands[4].reserved[0] == 0x6000 + 8 * 2);
    assert(commands[8].opcode == OPENNPUX_XGRAPH_OP_TDEQUANT);
    assert(commands[8].dim1 == 2);
    assert(commands[8].source0_offset == 0x3000 + 16 * 4);
    assert(commands[9].destination_offset == 0x2000 + 16 * 4);
    assert(commands[11].destination_offset ==
           0x2000 + 2 * 18 * 4 + 16 * 4);

    errno = 0;
    assert(opennpux_npu_xgraph_lower_gptq_matmul(
               &request, &parameters, EXTMEM_BASE, EXTMEM_SIZE,
               EXTMEM_BASE + 0x10000, 32 * 8 * 4, 0, commands, 11,
               &command_count) == -1);
    assert(errno == ENOSPC);
}

static void
test_gptq_k_tiled_accumulation(void)
{
    struct opennpux_npu_functional_request request;
    struct opennpux_npu_operator_parameters parameters;
    struct opennpux_xgraph_command commands[39];
    uint32_t command_count = 0;
    initialize(&request, &parameters, OPENNPUX_NPU_OP_MATMUL);
    request.command_id = 0;
    request.rows = 2;
    request.features = 2048;
    parameters.flags = OPENNPUX_NPU_PARAMETER_GPTQ;
    parameters.input_features = 2048;
    parameters.output_features = 18;
    parameters.quantization_bits = 4;
    parameters.quantization_group_size = 128;
    parameters.quantized_zero_bias = 1;
    parameters.scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT16;
    add_operand(&request, OPENNPUX_NPU_OPERAND_INPUT, 0x100000,
                2 * 2048 * 4);
    add_operand(&request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x110000,
                2 * 18 * 4);
    add_operand(&request, OPENNPUX_NPU_OPERAND_QWEIGHT, 0x120000,
                256 * 18 * 4);
    add_operand(&request, OPENNPUX_NPU_OPERAND_QZEROS, 0x130000,
                16 * 3 * 4);
    add_operand(&request, OPENNPUX_NPU_OPERAND_SCALES, 0x140000,
                16 * 18 * 2);
    add_operand(&request, OPENNPUX_NPU_OPERAND_G_IDX, 0x150000,
                2048 * 4);

    assert(opennpux_npu_xgraph_lower_gptq_matmul(
               &request, &parameters, EXTMEM_BASE, EXTMEM_SIZE,
               EXTMEM_BASE + 0x160000, (896 + 1) * 8 * 4, 200, commands,
               39, &command_count) == 0);
    assert(command_count == 39);
    assert(commands[0].opcode == OPENNPUX_XGRAPH_OP_TDEQUANT);
    assert(commands[0].dim2 == 896);
    assert(commands[0].flags == UINT32_C(0x00100000));
    assert(commands[3].opcode == OPENNPUX_XGRAPH_OP_TDEQUANT);
    assert(commands[3].flags == UINT32_C(0x00100007));
    assert(commands[3].source0_offset == 0x120000 + (896 / 8) * 18 * 4);
    assert(commands[3].reserved[1] == 0x150000 + 896 * 4);
    assert(commands[4].opcode == OPENNPUX_XGRAPH_OP_TMMA);
    assert(commands[4].source0_offset == 0x100000 + 896 * 4);
    assert(commands[4].destination_offset == 0x160000 + 896 * 8 * 4);
    assert(commands[5].opcode == OPENNPUX_XGRAPH_OP_TADD);
    assert(commands[5].destination_offset == 0x110000);
    assert(commands[5].source0_offset == 0x110000);
    assert(commands[5].source1_offset == 0x160000 + 896 * 8 * 4);
    assert(commands[6].source0_offset ==
           0x100000 + (2048 + 896) * 4);
    assert(commands[8].opcode == OPENNPUX_XGRAPH_OP_TDEQUANT);
    assert(commands[8].dim2 == 256);
    assert(commands[8].flags == UINT32_C(0x0010000e));
    assert(commands[38].opcode == OPENNPUX_XGRAPH_OP_TADD);
    assert(commands[38].destination_offset ==
           0x110000 + 18 * 4 + 16 * 4);
}

static void
initialize_large_gptq(struct opennpux_npu_functional_request *request,
                      struct opennpux_npu_operator_parameters *parameters,
                      uint32_t command_id)
{
    initialize(request, parameters, OPENNPUX_NPU_OP_MATMUL);
    request->command_id = command_id;
    request->rows = 2;
    request->features = 2048;
    parameters->flags = OPENNPUX_NPU_PARAMETER_GPTQ;
    parameters->input_features = 2048;
    parameters->output_features = 18;
    parameters->quantization_bits = 4;
    parameters->quantization_group_size = 128;
    parameters->quantized_zero_bias = 1;
    parameters->scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT16;
    add_operand(request, OPENNPUX_NPU_OPERAND_INPUT, 0x100000,
                2 * 2048 * 4);
    add_operand(request, OPENNPUX_NPU_OPERAND_OUTPUT, 0x110000,
                2 * 18 * 4);
    add_operand(request, OPENNPUX_NPU_OPERAND_QWEIGHT, 0x120000,
                256 * 18 * 4);
    add_operand(request, OPENNPUX_NPU_OPERAND_QZEROS, 0x130000,
                16 * 3 * 4);
    add_operand(request, OPENNPUX_NPU_OPERAND_SCALES, 0x140000,
                16 * 18 * 2);
    add_operand(request, OPENNPUX_NPU_OPERAND_G_IDX, 0x150000, 2048 * 4);
}

static void
test_bounded_mixed_batch_lowering(void)
{
    enum { request_count = 3, first_batch_capacity = 40 };
    struct opennpux_npu_functional_request requests[request_count];
    struct opennpux_npu_operator_parameters parameters[request_count];
    struct opennpux_npu_xgraph_lowering_options options[request_count];
    struct opennpux_xgraph_command commands[first_batch_capacity + 1];
    uint32_t origins[first_batch_capacity + 1];
    uint32_t requests_consumed = 0;
    uint32_t commands_emitted = 0;
    struct opennpux_npu_xgraph_lowering_failure failure;
    memset(options, 0, sizeof(options));

    initialize(&requests[0], &parameters[0], OPENNPUX_NPU_OP_ADD);
    requests[0].command_id = 10;
    add_operand(&requests[0], OPENNPUX_NPU_OPERAND_INPUT, 0x1000, 32);
    add_operand(&requests[0], OPENNPUX_NPU_OPERAND_SECONDARY, 0x2000, 32);
    add_operand(&requests[0], OPENNPUX_NPU_OPERAND_OUTPUT, 0x3000, 32);
    initialize_large_gptq(&requests[1], &parameters[1], 11);
    initialize(&requests[2], &parameters[2], OPENNPUX_NPU_OP_SOFTMAX);
    requests[2].command_id = 12;
    add_operand(&requests[2], OPENNPUX_NPU_OPERAND_INPUT, 0x4000, 32);
    add_operand(&requests[2], OPENNPUX_NPU_OPERAND_OUTPUT, 0x5000, 32);

    assert(opennpux_npu_xgraph_lower_batch(
               requests, parameters, options, request_count, EXTMEM_BASE,
               EXTMEM_SIZE, EXTMEM_BASE + 0x160000, (896 + 1) * 8 * 4,
               commands, first_batch_capacity, origins, &requests_consumed,
               &commands_emitted, &failure) == 0);
    assert(requests_consumed == 2);
    assert(commands_emitted == first_batch_capacity);
    assert(commands[0].opcode == OPENNPUX_XGRAPH_OP_TADD);
    assert(commands[0].command_id == 0);
    assert(origins[0] == 10);
    assert(commands[1].opcode == OPENNPUX_XGRAPH_OP_TDEQUANT);
    assert(commands[39].opcode == OPENNPUX_XGRAPH_OP_TADD);
    for (uint32_t index = 1; index < first_batch_capacity; ++index) {
        assert(commands[index].command_id == index);
        assert(origins[index] == 11);
    }

    struct opennpux_xgraph_command tail[1];
    uint32_t tail_origin[1];
    assert(opennpux_npu_xgraph_lower_batch(
               &requests[2], &parameters[2], &options[2], 1, EXTMEM_BASE,
               EXTMEM_SIZE, EXTMEM_BASE + 0x160000, (896 + 1) * 8 * 4,
               tail, 1, tail_origin, &requests_consumed, &commands_emitted,
               &failure) == 0);
    assert(requests_consumed == 1 && commands_emitted == 1);
    assert(tail[0].opcode == OPENNPUX_XGRAPH_OP_TSOFTMAX);
    assert(tail[0].command_id == 0);
    assert(tail_origin[0] == 12);

    struct opennpux_xgraph_command too_small[38];
    errno = 0;
    assert(opennpux_npu_xgraph_lower_batch(
               &requests[1], &parameters[1], &options[1], 1, EXTMEM_BASE,
               EXTMEM_SIZE, EXTMEM_BASE + 0x160000, (896 + 1) * 8 * 4,
               too_small, 38, NULL, &requests_consumed, &commands_emitted,
               &failure) == -1);
    assert(errno == ENOSPC);
    assert(requests_consumed == 0 && commands_emitted == 0);
    assert(failure.command_index == 0);
    assert(failure.command_id == 11);
    assert(failure.opcode == OPENNPUX_NPU_OP_MATMUL);
    assert(failure.error_code == ENOSPC);

    requests[2].command_id = 13;
    errno = 0;
    assert(opennpux_npu_xgraph_lower_batch(
               requests, parameters, options, request_count, EXTMEM_BASE,
               EXTMEM_SIZE, EXTMEM_BASE + 0x160000, (896 + 1) * 8 * 4,
               commands, first_batch_capacity + 1, origins,
               &requests_consumed, &commands_emitted, &failure) == -1);
    assert(errno == EINVAL);
    assert(requests_consumed == 2);
    assert(commands_emitted == first_batch_capacity);
    assert(failure.command_index == 2);
    assert(failure.command_id == 13);
    assert(failure.opcode == OPENNPUX_NPU_OP_SOFTMAX);
    assert(failure.error_code == EINVAL);
    assert(commands[first_batch_capacity].opcode == 0);
    assert(origins[first_batch_capacity] == UINT32_MAX);
}

int
main(void)
{
    test_direct_primitives();
    test_semantic_options();
    test_embed_and_rejections();
    test_sequence_lowering();
    test_gptq_tiled_lowering();
    test_gptq_k_tiled_accumulation();
    test_bounded_mixed_batch_lowering();
    puts("NPU XGraph primitive lowering test: PASS");
    return 0;
}
