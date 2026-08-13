#include "opennpux/npu_submission.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s (errno=%d)\n", message, errno);
        _Exit(1);
    }
}

int
main(void)
{
    _Alignas(64) uint8_t buffer[4096];
    struct opennpux_npu_submission_builder builder;
    check(opennpux_npu_submission_begin(
              &builder, buffer, sizeof(buffer), 7, 0x1001, 0x2002,
              OPENNPUX_NPU_ENTRY_DECODE, 3, 2) == 0,
          "submission begin failed");

    for (uint32_t index = 0; index < 3; ++index) {
        struct opennpux_npu_tensor_binding *binding =
            opennpux_npu_submission_binding(&builder, index);
        check(binding != NULL, "binding lookup failed");
        binding->tensor_id = index + 1;
        binding->flags = index == 2 ? OPENNPUX_NPU_BIND_WRITE :
                                      OPENNPUX_NPU_BIND_READ;
        binding->data_type = OPENNPUX_NPU_DTYPE_BFLOAT16;
        binding->rank = 2;
        binding->device_address = UINT64_C(0x8ff00000) + index * 0x1000;
        binding->byte_size = 4096;
        binding->dimensions[0] = 1;
        binding->dimensions[1] = 2048;
    }
    builder.bindings[0].flags |= OPENNPUX_NPU_BIND_WEIGHT;
    builder.bindings[1].flags |= OPENNPUX_NPU_BIND_PERSISTENT;

    struct opennpux_npu_command *matmul =
        opennpux_npu_submission_command(&builder, 0);
    struct opennpux_npu_command *add =
        opennpux_npu_submission_command(&builder, 1);
    check(matmul != NULL && add != NULL, "command lookup failed");
    matmul->command_id = 10;
    matmul->opcode = OPENNPUX_NPU_OP_MATMUL;
    matmul->first_binding = 0;
    matmul->binding_count = 3;
    matmul->completion_token = 1;
    matmul->estimated_operations = 8192;
    matmul->parameter_symbol = 0x1234;
    matmul->runtime_shape = opennpux_npu_pack_runtime_shape(1, 1, 1, 8);
    matmul->resource_bindings =
        opennpux_npu_pack_resource_bindings(0, 1, 2);
    add->command_id = 11;
    add->opcode = OPENNPUX_NPU_OP_ADD;
    add->first_binding = 0;
    add->binding_count = 3;
    add->dependency_token = 1;
    add->completion_token = 2;
    add->parameter_symbol = 0x5678;
    add->runtime_shape = matmul->runtime_shape;
    add->resource_bindings = matmul->resource_bindings;

    builder.header->flags = OPENNPUX_NPU_INVOKE_PROFILE;
    builder.header->persistent_state_handle = 0x3003;
    builder.header->dependency_fence = 0x4004;
    builder.header->completion_address = 0x8ff0f000;
    check(opennpux_npu_submission_finalize(&builder) == 0,
          "submission finalize failed");
    check(opennpux_npu_submission_validate(
              buffer, builder.header->total_size) == 0,
          "valid submission rejected");

    const uint32_t checksum = builder.header->checksum;
    buffer[builder.header->command_offset +
           offsetof(struct opennpux_npu_command, opcode)] ^= 1;
    check(opennpux_npu_submission_validate(
              buffer, builder.header->total_size) != 0,
          "corrupted submission accepted");
    buffer[builder.header->command_offset +
           offsetof(struct opennpux_npu_command, opcode)] ^= 1;
    check(builder.header->checksum == checksum,
          "test did not preserve checksum field");
    check(opennpux_npu_submission_validate(
              buffer, builder.header->total_size) == 0,
          "restored submission rejected");

    struct opennpux_npu_completion completion = {
        .magic = OPENNPUX_NPU_COMPLETION_MAGIC,
        .version = OPENNPUX_NPU_COMPLETION_VERSION,
        .struct_size = sizeof(completion),
        .state = OPENNPUX_NPU_COMPLETION_SUCCESS,
        .sequence = 7,
        .completed_commands = 2,
    };
    check(completion.sequence == builder.header->sequence,
          "completion sequence mismatch");
    puts("PASS: generic NPU submission ABI tests");
    return 0;
}
