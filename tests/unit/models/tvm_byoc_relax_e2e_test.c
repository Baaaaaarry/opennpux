#include "opennpux/xopennpux_graph.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ARENA_BYTES UINT32_C(0x00100000)

static float *
tensor(float *arena, uint32_t offset)
{
    assert(offset < ARENA_BYTES);
    return (float *)((uint8_t *)arena + offset);
}

static void
execute(const struct opennpux_xgraph_command *command, float *arena)
{
    float *destination = tensor(arena, command->destination_offset);
    const float *source0 = tensor(arena, command->source0_offset);
    const float *source1 = command->source1_offset == 0
                               ? NULL
                               : tensor(arena, command->source1_offset);
    uint32_t row;
    uint32_t column;

    switch (command->opcode) {
      case OPENNPUX_XGRAPH_OP_TMMA:
      {
        const uint32_t lhs_stride = command->reserved[0] == 0
            ? command->dim2 : command->reserved[0] / sizeof(float);
        const uint32_t rhs_stride = command->reserved[1] == 0
            ? (command->flags & OPENNPUX_XGRAPH_TMMA_TRANSPOSE_RHS
                   ? command->dim2 : command->dim1)
            : command->reserved[1] / sizeof(float);
        const uint32_t output_stride = command->reserved[2] == 0
            ? command->dim1 : command->reserved[2] / sizeof(float);
        for (row = 0; row < command->dim0; ++row) {
            for (column = 0; column < command->dim1; ++column) {
                float sum =
                    command->flags & OPENNPUX_XGRAPH_TMMA_ACCUMULATE
                        ? destination[row * output_stride + column]
                        : 0.0f;
                for (uint32_t inner = 0; inner < command->dim2; ++inner) {
                    const float rhs =
                        command->flags & OPENNPUX_XGRAPH_TMMA_TRANSPOSE_RHS
                            ? source1[column * rhs_stride + inner]
                            : source1[inner * rhs_stride + column];
                    sum += source0[row * lhs_stride + inner] * rhs;
                }
                destination[row * output_stride + column] = sum;
            }
        }
        break;
      }
      case OPENNPUX_XGRAPH_OP_TADD:
        for (row = 0; row < command->dim0 * command->dim1; ++row) {
            destination[row] = source0[row] + source1[row];
        }
        break;
      case OPENNPUX_XGRAPH_OP_TSILU:
        for (row = 0; row < command->dim0 * command->dim1; ++row) {
            destination[row] = source0[row] / (1.0f + expf(-source0[row]));
        }
        break;
      case OPENNPUX_XGRAPH_OP_TSOFTMAX:
        for (row = 0; row < command->dim0; ++row) {
            float maximum = source0[row * command->dim1];
            float denominator = 0.0f;
            for (column = 1; column < command->dim1; ++column) {
                maximum = fmaxf(maximum, source0[row * command->dim1 + column]);
            }
            for (column = 0; column < command->dim1; ++column) {
                const float value =
                    expf(source0[row * command->dim1 + column] - maximum);
                destination[row * command->dim1 + column] = value;
                denominator += value;
            }
            for (column = 0; column < command->dim1; ++column) {
                destination[row * command->dim1 + column] /= denominator;
            }
        }
        break;
      default:
        assert(!"unexpected XGraph opcode");
    }
}

int
main(int argc, char **argv)
{
    assert(argc == 3);
    FILE *source = fopen(argv[1], "rb");
    assert(source != NULL);
    struct opennpux_xgraph_header header;
    assert(fread(&header, sizeof(header), 1, source) == 1);
    assert(header.magic == OPENNPUX_XGRAPH_MAGIC);
    assert(header.version == OPENNPUX_XGRAPH_VERSION);
    assert(header.command_count == 6);
    struct opennpux_xgraph_command *commands =
        calloc(header.command_count, sizeof(*commands));
    assert(commands != NULL);
    assert(fread(commands, sizeof(*commands), header.command_count, source) ==
           header.command_count);
    assert(fclose(source) == 0);

    FILE *arena_source = fopen(argv[2], "rb");
    assert(arena_source != NULL);
    float *arena = calloc(1, ARENA_BYTES);
    assert(arena != NULL);
    assert(fread(arena, 1, ARENA_BYTES, arena_source) > 0);
    assert(fclose(arena_source) == 0);
    const float *input = tensor(arena, commands[0].source0_offset);
    const float *weight = tensor(arena, commands[0].source1_offset);
    const float *bias = tensor(arena, commands[3].source1_offset);

    const uint32_t expected_opcodes[] = {
        OPENNPUX_XGRAPH_OP_TMMA,
        OPENNPUX_XGRAPH_OP_TMMA,
        OPENNPUX_XGRAPH_OP_TMMA,
        OPENNPUX_XGRAPH_OP_TADD,
        OPENNPUX_XGRAPH_OP_TSILU,
        OPENNPUX_XGRAPH_OP_TSOFTMAX,
    };
    for (uint32_t index = 0; index < header.command_count; ++index) {
        assert(commands[index].opcode == expected_opcodes[index]);
        execute(&commands[index], arena);
    }

    float expected[16];
    for (uint32_t row = 0; row < 2; ++row) {
        float denominator = 0.0f;
        for (uint32_t column = 0; column < 8; ++column) {
            float sum = bias[row * 8 + column];
            for (uint32_t inner = 0; inner < 2048; ++inner) {
                sum += input[row * 2048 + inner] *
                       weight[inner * 8 + column];
            }
            expected[row * 8 + column] = sum / (1.0f + expf(-sum));
        }
        float maximum = expected[row * 8];
        for (uint32_t column = 1; column < 8; ++column) {
            maximum = fmaxf(maximum, expected[row * 8 + column]);
        }
        for (uint32_t column = 0; column < 8; ++column) {
            expected[row * 8 + column] =
                expf(expected[row * 8 + column] - maximum);
            denominator += expected[row * 8 + column];
        }
        for (uint32_t column = 0; column < 8; ++column) {
            expected[row * 8 + column] /= denominator;
        }
    }
    const float *actual = tensor(arena, header.output_offset);
    float max_error = 0.0f;
    for (uint32_t index = 0; index < 16; ++index) {
        const float error = fabsf(actual[index] - expected[index]);
        max_error = fmaxf(max_error, error);
    }
    assert(max_error < 5.0e-5f);
    const uint8_t *output_bytes = (const uint8_t *)(const void *)actual;
    uint32_t checksum = UINT32_C(2166136261);
    for (uint32_t index = 0; index < header.output_bytes; ++index) {
        checksum ^= output_bytes[index];
        checksum *= UINT32_C(16777619);
    }
    printf("xgraph_output_checksum=0x%08x\n", checksum);
    printf("xgraph_max_abs_error=%.9g\n", max_error);
    free(commands);
    free(arena);
    puts("TVM Relax -> BYOC -> XGraph -> C execution: PASS");
    return 0;
}
