#include "opennpux/npu_executable.h"
#include "opennpux/npu_functional_materializer.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s (errno=%d)\n", message, errno);
        exit(1);
    }
}

int
main(int argc, char **argv)
{
    check(argc == 3,
          "usage: npu_functional_program_test <model.npxc> <model.npxtb>");
    struct opennpux_npu_executable executable;
    struct opennpux_npu_tensor_plan tensor_plan;
    check(opennpux_npu_executable_load(argv[1], &executable) == 0,
          "load executable");
    check(opennpux_npu_tensor_plan_load(argv[2], &tensor_plan) == 0,
          "load tensor plan");

    struct opennpux_npu_tensor_binding bindings[5] = {0};
    for (uint32_t index = 0; index < 5; ++index) {
        bindings[index].tensor_id = index;
        bindings[index].flags = OPENNPUX_NPU_BIND_READ;
        bindings[index].data_type = OPENNPUX_NPU_DTYPE_BFLOAT16;
        bindings[index].rank = 2;
        bindings[index].device_address = 0x20000000 + index * 0x01000000;
        bindings[index].byte_size = 0x01000000;
        bindings[index].dimensions[0] = 1;
        bindings[index].dimensions[1] = 2048;
        bindings[index].memory_object = index + 1;
    }
    bindings[1].flags = OPENNPUX_NPU_BIND_WRITE;
    bindings[2].flags |= OPENNPUX_NPU_BIND_WEIGHT;
    bindings[3].flags |= OPENNPUX_NPU_BIND_PERSISTENT |
                         OPENNPUX_NPU_BIND_WRITE;
    bindings[4].flags |= OPENNPUX_NPU_BIND_WRITE;

    const size_t capacity = 1024 * 1024;
    void *submission = aligned_alloc(OPENNPUX_NPU_RECORD_ALIGNMENT, capacity);
    check(submission != NULL, "allocate submission");
    const struct opennpux_npu_invocation_parameters invocation = {
        .batch_size = 1,
        .sequence_length = 1,
        .kv_length = 17,
        .active_experts = 2,
    };
    size_t submission_size = 0;
    check(opennpux_npu_executable_instantiate_with_parameters(
              &executable, OPENNPUX_NPU_ENTRY_DECODE, 1, 1, &invocation,
              bindings, 5, submission, capacity, &submission_size) == 0,
          "instantiate executable");

    const struct opennpux_npu_tensor_plan_memory memory = {
        .input_address = 0x20000000,
        .input_size = 0x01000000,
        .output_address = 0x21000000,
        .output_size = 0x01000000,
        .persistent_address = 0x22000000,
        .persistent_size = 0x01000000,
        .scratch_address = 0x23000000,
        .scratch_size = 0x01000000,
    };
    struct opennpux_npu_functional_program program;
    check(opennpux_npu_functional_program_init(
              &program, submission, submission_size, 0x24000000,
              &tensor_plan, &memory) == 0,
          "initialize functional program");
    uint32_t multi_output_commands = 0;
    for (uint32_t index = 0; index < program.header->command_count; ++index) {
        struct opennpux_npu_functional_request request;
        check(opennpux_npu_functional_program_materialize(
                  &program, index, NULL, 0, &request) == 0,
              "materialize program command");
        check(request.command_id == program.commands[index].command_id &&
                  request.opcode == program.commands[index].opcode,
              "request identity");
        if (request.operand_count > 2) {
            ++multi_output_commands;
        }
    }
    check(multi_output_commands != 0, "multi-operand commands absent");

    check(program.header->command_count > 1,
          "functional program needs multiple commands for validation tests");
    struct opennpux_npu_invocation_header *mutable_header = submission;
    struct opennpux_npu_command *mutable_commands =
        (struct opennpux_npu_command *)((uint8_t *)submission +
                                        mutable_header->command_offset);
    const uint32_t saved_command_id = mutable_commands[1].command_id;
    mutable_commands[1].command_id = mutable_commands[0].command_id;
    mutable_header->checksum = opennpux_npu_submission_checksum(
        submission, mutable_header->total_size);
    errno = 0;
    check(opennpux_npu_functional_program_init(
              &program, submission, submission_size, 0x24000000,
              &tensor_plan, &memory) != 0 && errno == EEXIST,
          "reject duplicate command id");
    mutable_commands[1].command_id = saved_command_id;

    const uint64_t saved_runtime_shape = mutable_commands[1].runtime_shape;
    mutable_commands[1].runtime_shape = opennpux_npu_pack_runtime_shape(
        1, 2, invocation.kv_length, invocation.active_experts);
    mutable_header->checksum = opennpux_npu_submission_checksum(
        submission, mutable_header->total_size);
    errno = 0;
    check(opennpux_npu_functional_program_init(
              &program, submission, submission_size, 0x24000000,
              &tensor_plan, &memory) != 0 && errno == EINVAL,
          "reject inconsistent runtime shape");
    mutable_commands[1].runtime_shape = saved_runtime_shape;
    mutable_header->checksum = opennpux_npu_submission_checksum(
        submission, mutable_header->total_size);
    check(opennpux_npu_functional_program_init(
              &program, submission, submission_size, 0x24000000,
              &tensor_plan, &memory) == 0,
          "restore valid functional program");

    printf("functional_program_commands=%u\n", program.header->command_count);
    puts("npu_functional_program=PASS");
    free(submission);
    opennpux_npu_tensor_plan_unload(&tensor_plan);
    opennpux_npu_executable_unload(&executable);
    return 0;
}
