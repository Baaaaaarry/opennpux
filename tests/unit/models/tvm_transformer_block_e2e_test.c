#include "opennpux/xopennpux_graph.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARENA_BYTES UINT32_C(0x00100000)
#define ROWS UINT32_C(2)
#define FEATURES UINT32_C(64)

static float *
tensor(float *arena, uint32_t offset)
{
    assert(offset < ARENA_BYTES);
    return (float *)((uint8_t *)arena + offset);
}

static void
execute(const struct opennpux_xgraph_command *command, float *arena)
{
    float *output = tensor(arena, command->destination_offset);
    const float *input = tensor(arena, command->source0_offset);
    const float *rhs = command->source1_offset == 0
                           ? NULL
                           : tensor(arena, command->source1_offset);
    if (command->opcode == OPENNPUX_XGRAPH_OP_TRMSNORM) {
        float epsilon;
        const uint32_t epsilon_bits = command->scalar0;
        memcpy(&epsilon, &epsilon_bits, sizeof(epsilon));
        for (uint32_t row = 0; row < command->dim0; ++row) {
            float square_sum = 0.0f;
            for (uint32_t column = 0; column < command->dim1; ++column) {
                const float value = input[row * command->dim1 + column];
                square_sum += value * value;
            }
            const float inverse =
                1.0f / sqrtf(square_sum / command->dim1 + epsilon);
            for (uint32_t column = 0; column < command->dim1; ++column) {
                output[row * command->dim1 + column] =
                    input[row * command->dim1 + column] * inverse * rhs[column];
            }
        }
        return;
    }
    if (command->opcode == OPENNPUX_XGRAPH_OP_TMMA) {
        for (uint32_t row = 0; row < command->dim0; ++row) {
            for (uint32_t column = 0; column < command->dim1; ++column) {
                float sum = 0.0f;
                for (uint32_t inner = 0; inner < command->dim2; ++inner) {
                    sum += input[row * command->dim2 + inner] *
                           rhs[inner * command->dim1 + column];
                }
                output[row * command->dim1 + column] = sum;
            }
        }
        return;
    }
    for (uint32_t index = 0; index < command->dim0 * command->dim1; ++index) {
        if (command->opcode == OPENNPUX_XGRAPH_OP_TADD) {
            output[index] = input[index] + rhs[index];
        } else {
            assert(command->opcode == OPENNPUX_XGRAPH_OP_TSILU);
            output[index] = input[index] / (1.0f + expf(-input[index]));
        }
    }
}

int
main(int argc, char **argv)
{
    assert(argc == 3);
    FILE *graph = fopen(argv[1], "rb");
    assert(graph != NULL);
    struct opennpux_xgraph_header header;
    assert(fread(&header, sizeof(header), 1, graph) == 1);
    assert(header.magic == OPENNPUX_XGRAPH_MAGIC);
    assert(header.command_count == 4);
    struct opennpux_xgraph_command commands[4];
    assert(fread(commands, sizeof(commands), 1, graph) == 1);
    assert(fclose(graph) == 0);
    const uint32_t expected_opcodes[] = {
        OPENNPUX_XGRAPH_OP_TRMSNORM,
        OPENNPUX_XGRAPH_OP_TMMA,
        OPENNPUX_XGRAPH_OP_TADD,
        OPENNPUX_XGRAPH_OP_TSILU,
    };
    for (uint32_t index = 0; index < header.command_count; ++index) {
        assert(commands[index].opcode == expected_opcodes[index]);
    }

    FILE *arena_file = fopen(argv[2], "rb");
    assert(arena_file != NULL);
    float *arena = calloc(1, ARENA_BYTES);
    assert(arena != NULL);
    assert(fread(arena, 1, ARENA_BYTES, arena_file) > 0);
    assert(fclose(arena_file) == 0);

    const float *hidden = tensor(arena, commands[0].source0_offset);
    const float *norm_weight = tensor(arena, commands[0].source1_offset);
    const float *projection_weight = tensor(arena, commands[1].source1_offset);
    const float *residual = tensor(arena, commands[2].source1_offset);
    for (uint32_t index = 0; index < header.command_count; ++index) {
        execute(&commands[index], arena);
    }

    float expected[ROWS * FEATURES];
    for (uint32_t row = 0; row < ROWS; ++row) {
        float square_sum = 0.0f;
        for (uint32_t inner = 0; inner < FEATURES; ++inner) {
            const float value = hidden[row * FEATURES + inner];
            square_sum += value * value;
        }
        const float inverse = 1.0f / sqrtf(square_sum / FEATURES + 1.0e-5f);
        for (uint32_t column = 0; column < FEATURES; ++column) {
            float projected = 0.0f;
            for (uint32_t inner = 0; inner < FEATURES; ++inner) {
                projected += hidden[row * FEATURES + inner] * inverse *
                             norm_weight[inner] *
                             projection_weight[inner * FEATURES + column];
            }
            const float sum = projected + residual[row * FEATURES + column];
            expected[row * FEATURES + column] =
                sum / (1.0f + expf(-sum));
        }
    }
    const float *actual = tensor(arena, header.output_offset);
    float maximum = 0.0f;
    for (uint32_t index = 0; index < ROWS * FEATURES; ++index) {
        maximum = fmaxf(maximum, fabsf(actual[index] - expected[index]));
    }
    assert(maximum < 5.0e-5f);
    FILE *output = fopen(argv[2], "r+b");
    assert(output != NULL);
    assert(fseek(output, (long)header.output_offset, SEEK_SET) == 0);
    assert(fwrite(expected, sizeof(expected), 1, output) == 1);
    assert(fclose(output) == 0);
    printf("transformer_block_max_abs_error=%.9g\n", maximum);
    puts("TVM Transformer block -> BYOC -> XGraph -> C execution: PASS");
    free(arena);
    return 0;
}
