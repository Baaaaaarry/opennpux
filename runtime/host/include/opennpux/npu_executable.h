#ifndef OPENNPUX_NPU_EXECUTABLE_H
#define OPENNPUX_NPU_EXECUTABLE_H

#include "opennpux/npu_submission.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_NPU_EXECUTABLE_MAGIC UINT32_C(0x4558504e)
#define OPENNPUX_NPU_EXECUTABLE_VERSION UINT32_C(1)
#define OPENNPUX_NPU_MAX_ENTRY_POINTS UINT32_C(32)
#define OPENNPUX_NPU_EXECUTABLE_HEADER_SIZE UINT32_C(88)
#define OPENNPUX_NPU_EXECUTABLE_ENTRY_SIZE UINT32_C(48)
#define OPENNPUX_NPU_COMMAND_TEMPLATE_SIZE UINT32_C(80)

struct opennpux_npu_executable_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t total_size;
    uint32_t entry_count;
    uint32_t command_count;
    uint32_t entry_record_size;
    uint32_t command_record_size;
    uint64_t entry_offset;
    uint64_t command_offset;
    uint64_t executable_id;
    uint32_t checksum;
    uint32_t default_active_experts;
    uint64_t reserved[3];
};

struct opennpux_npu_executable_entry {
    uint32_t entry_point;
    uint32_t first_command;
    uint32_t command_count;
    uint32_t flags;
    uint64_t scratch_size;
    uint64_t persistent_state_size;
    uint64_t reserved[2];
};

struct opennpux_npu_command_template {
    uint32_t command_id;
    uint32_t opcode;
    uint32_t flags;
    uint32_t capability_id;
    uint32_t first_binding;
    uint32_t binding_count;
    uint32_t dependency_token;
    uint32_t completion_token;
    uint64_t parameter_symbol;
    uint64_t estimated_operations;
    uint64_t estimated_bytes;
    uint64_t profiling_tag;
    uint64_t reserved0;
    uint64_t resource_bindings;
};

struct opennpux_npu_invocation_parameters {
    uint32_t batch_size;
    uint32_t sequence_length;
    uint32_t kv_length;
    uint32_t active_experts;
};

struct opennpux_npu_executable {
    void *storage;
    size_t storage_size;
    const struct opennpux_npu_executable_header *header;
    const struct opennpux_npu_executable_entry *entries;
    const struct opennpux_npu_command_template *commands;
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_executable_header) ==
              OPENNPUX_NPU_EXECUTABLE_HEADER_SIZE);
static_assert(sizeof(struct opennpux_npu_executable_entry) ==
              OPENNPUX_NPU_EXECUTABLE_ENTRY_SIZE);
static_assert(sizeof(struct opennpux_npu_command_template) ==
              OPENNPUX_NPU_COMMAND_TEMPLATE_SIZE);
#else
_Static_assert(sizeof(struct opennpux_npu_executable_header) ==
               OPENNPUX_NPU_EXECUTABLE_HEADER_SIZE,
               "NPU executable header ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_executable_entry) ==
               OPENNPUX_NPU_EXECUTABLE_ENTRY_SIZE,
               "NPU executable entry ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_command_template) ==
               OPENNPUX_NPU_COMMAND_TEMPLATE_SIZE,
               "NPU command template ABI size changed");
#endif

int opennpux_npu_executable_load(
    const char *path, struct opennpux_npu_executable *executable);
void opennpux_npu_executable_unload(
    struct opennpux_npu_executable *executable);
const struct opennpux_npu_executable_entry *opennpux_npu_executable_find_entry(
    const struct opennpux_npu_executable *executable, uint32_t entry_point);
int opennpux_npu_executable_instantiate(
    const struct opennpux_npu_executable *executable, uint32_t entry_point,
    uint64_t sequence, uint64_t context_id,
    const struct opennpux_npu_tensor_binding *bindings, uint32_t binding_count,
    void *submission, size_t submission_capacity, size_t *submission_size);
int opennpux_npu_executable_instantiate_with_parameters(
    const struct opennpux_npu_executable *executable, uint32_t entry_point,
    uint64_t sequence, uint64_t context_id,
    const struct opennpux_npu_invocation_parameters *parameters,
    const struct opennpux_npu_tensor_binding *bindings, uint32_t binding_count,
    void *submission, size_t submission_capacity, size_t *submission_size);

#ifdef __cplusplus
}
#endif

#endif
