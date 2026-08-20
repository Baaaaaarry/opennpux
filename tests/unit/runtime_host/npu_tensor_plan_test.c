#include "opennpux/npu_tensor_plan.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
    check(argc == 2, "usage: npu_tensor_plan_test <model.npxtb>");
    struct opennpux_npu_tensor_plan plan;
    check(opennpux_npu_tensor_plan_load(argv[1], &plan) == 0,
          "tensor plan load failed");
    check(plan.header->command_count == 30, "command count mismatch");
    check(plan.header->tensor_count == 37, "tensor count mismatch");
    check(plan.header->slot_count == 6, "slot count mismatch");

    uint64_t scratch_size = 0;
    check(opennpux_npu_tensor_plan_scratch_size(&plan, 2, 4, &scratch_size) == 0,
          "scratch size resolution failed");
    check(scratch_size == plan.header->scratch_bytes_per_runtime_row * 8,
          "scratch size scaling mismatch");

    const struct opennpux_npu_tensor_plan_tensor *scratch = NULL;
    for (uint32_t index = 0; index < plan.header->tensor_count; ++index) {
        if (plan.tensors[index].storage == OPENNPUX_NPU_TENSOR_SCRATCH) {
            scratch = &plan.tensors[index];
            break;
        }
    }
    check(scratch != NULL, "scratch tensor missing");
    uint64_t address = 0;
    uint64_t size = 0;
    check(opennpux_npu_tensor_plan_resolve_scratch(
              &plan, scratch->tensor_id, 2, 4, UINT64_C(0x90000000),
              scratch_size, &address, &size) == 0,
          "scratch tensor resolution failed");
    check(address >= UINT64_C(0x90000000) &&
              address + size <= UINT64_C(0x90000000) + scratch_size,
          "resolved scratch tensor outside arena");
    check(opennpux_npu_tensor_plan_resolve_scratch(
              &plan, scratch->tensor_id, 2, 4, UINT64_C(0x90000000),
              scratch_size - 1, &address, &size) != 0 && errno == ENOSPC,
          "undersized scratch arena accepted");

    const struct opennpux_npu_tensor_plan_runtime runtime = {
        .batch_size = 2,
        .sequence_length = 4,
        .kv_length = 7,
        .active_experts = 2,
    };
    uint64_t persistent_size = 0;
    check(opennpux_npu_tensor_plan_persistent_size(
              &plan, &runtime, &persistent_size) == 0 && persistent_size != 0,
          "persistent size resolution failed");
    const struct opennpux_npu_tensor_plan_memory memory = {
        .input_address = UINT64_C(0x80000000),
        .input_size = UINT64_C(0x10000),
        .output_address = UINT64_C(0x81000000),
        .output_size = UINT64_C(0x10000),
        .persistent_address = UINT64_C(0x82000000),
        .persistent_size = persistent_size,
        .scratch_address = UINT64_C(0x90000000),
        .scratch_size = scratch_size,
    };
    check(opennpux_npu_tensor_plan_resolve(
              &plan, 0, &runtime, &memory, &address, &size) == 0 &&
              address == memory.input_address && size == 32,
          "input tensor resolution failed");
    check(opennpux_npu_tensor_plan_resolve(
              &plan, 1, &runtime, &memory, &address, &size) == 0 &&
              address == memory.output_address && size == 8,
          "output tensor resolution failed");
    const struct opennpux_npu_tensor_plan_tensor *persistent = NULL;
    for (uint32_t index = 0; index < plan.header->tensor_count; ++index) {
        if (plan.tensors[index].storage == OPENNPUX_NPU_TENSOR_PERSISTENT) {
            persistent = &plan.tensors[index];
            break;
        }
    }
    check(persistent != NULL && opennpux_npu_tensor_plan_resolve(
              &plan, persistent->tensor_id, &runtime, &memory, &address, &size) == 0 &&
              address >= memory.persistent_address &&
              address + size <= memory.persistent_address + memory.persistent_size,
          "persistent tensor resolution failed");

    opennpux_npu_tensor_plan_unload(&plan);
    puts("PASS: NPU tensor plan loader and scratch resolver tests");
    return 0;
}
