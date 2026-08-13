#include "opennpux/npu_executable.h"

#include <errno.h>
#include <stdint.h>
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
    check(argc == 2, "usage: npu_executable_test <model.npxc>");
    struct opennpux_npu_executable executable;
    check(opennpux_npu_executable_load(argv[1], &executable) == 0,
          "executable load failed");
    check(executable.header->entry_count == 2, "entry count mismatch");
    check(executable.header->command_count != 0, "command list is empty");

    struct opennpux_npu_tensor_binding bindings[5];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t index = 0; index < 5; ++index) {
        bindings[index].tensor_id = index;
        bindings[index].flags = index == 1 ? OPENNPUX_NPU_BIND_WRITE :
                                             OPENNPUX_NPU_BIND_READ;
        bindings[index].data_type = OPENNPUX_NPU_DTYPE_BFLOAT16;
        bindings[index].rank = 2;
        bindings[index].device_address = UINT64_C(0x80000000) + index * 0x10000;
        bindings[index].byte_size = 0x10000;
        bindings[index].dimensions[0] = 1;
        bindings[index].dimensions[1] = 2048;
        bindings[index].memory_object = index + 1;
    }
    bindings[2].flags |= OPENNPUX_NPU_BIND_WEIGHT;
    bindings[3].flags |= OPENNPUX_NPU_BIND_PERSISTENT |
                         OPENNPUX_NPU_BIND_WRITE;
    bindings[4].flags |= OPENNPUX_NPU_BIND_WRITE;

    const size_t submission_capacity =
        sizeof(struct opennpux_npu_invocation_header) +
        5 * sizeof(struct opennpux_npu_tensor_binding) +
        executable.header->command_count * sizeof(struct opennpux_npu_command) +
        3 * OPENNPUX_NPU_RECORD_ALIGNMENT;
    void *submission = aligned_alloc(
        OPENNPUX_NPU_RECORD_ALIGNMENT,
        (submission_capacity + OPENNPUX_NPU_RECORD_ALIGNMENT - 1) &
            ~(size_t)(OPENNPUX_NPU_RECORD_ALIGNMENT - 1));
    check(submission != NULL, "submission allocation failed");
    size_t submission_size = 0;
    const struct opennpux_npu_invocation_parameters parameters = {
        .batch_size = 1,
        .sequence_length = 1,
        .kv_length = 17,
        .active_experts = 2,
    };
    check(opennpux_npu_executable_instantiate_with_parameters(
              &executable, OPENNPUX_NPU_ENTRY_DECODE, 11, 22, &parameters,
              bindings, 5, submission, submission_capacity,
              &submission_size) == 0,
          "decode invocation instantiate failed");
    check(opennpux_npu_submission_validate(submission, submission_size) == 0,
          "instantiated invocation rejected");
    const struct opennpux_npu_invocation_header *header =
        (const struct opennpux_npu_invocation_header *)submission;
    check(header->entry_point == OPENNPUX_NPU_ENTRY_DECODE,
          "decode entry mismatch");
    check(header->sequence == 11 && header->context_id == 22,
          "live invocation identity mismatch");
    check(header->persistent_state_handle == bindings[3].memory_object,
          "persistent state was not bound at runtime");
    check(header->command_count == executable.header->command_count,
          "command template was not instantiated");
    const struct opennpux_npu_command *commands =
        (const struct opennpux_npu_command *)((const uint8_t *)submission +
                                               header->command_offset);
    check(commands[0].parameter_symbol != 0,
          "command parameter symbol was not relocated");
    check((commands[0].runtime_shape & OPENNPUX_NPU_RUNTIME_FIELD_MASK) == 1,
          "runtime batch was not instantiated");
    check(((commands[0].runtime_shape >> OPENNPUX_NPU_RUNTIME_KV_SHIFT) &
           OPENNPUX_NPU_RUNTIME_FIELD_MASK) == parameters.kv_length,
          "KV length was not instantiated");
    check(((commands[0].runtime_shape >> OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT) &
           OPENNPUX_NPU_RUNTIME_FIELD_MASK) == parameters.active_experts,
          "active expert count was not instantiated");
    check((commands[0].resource_bindings & OPENNPUX_NPU_RUNTIME_FIELD_MASK) == 2,
          "weight binding was not relocated");

    free(submission);
    opennpux_npu_executable_unload(&executable);
    puts("PASS: generic NPU executable and invocation tests");
    return 0;
}
