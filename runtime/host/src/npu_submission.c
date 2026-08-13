#include "opennpux/npu_submission.h"

#include <errno.h>
#include <string.h>

static size_t
align_size(size_t value)
{
    const size_t alignment = OPENNPUX_NPU_RECORD_ALIGNMENT;
    if (value > SIZE_MAX - (alignment - 1)) {
        return 0;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}

static int
array_size(uint32_t count, size_t element_size, size_t *result)
{
    if (count != 0 && element_size > SIZE_MAX / count) {
        errno = EOVERFLOW;
        return -1;
    }
    *result = (size_t)count * element_size;
    return 0;
}

uint64_t
opennpux_npu_pack_runtime_shape(
    uint32_t batch_size, uint32_t sequence_length, uint32_t kv_length,
    uint32_t active_experts)
{
    if (batch_size > OPENNPUX_NPU_RUNTIME_FIELD_MASK ||
        sequence_length > OPENNPUX_NPU_RUNTIME_FIELD_MASK ||
        kv_length > OPENNPUX_NPU_RUNTIME_FIELD_MASK ||
        active_experts > OPENNPUX_NPU_RUNTIME_FIELD_MASK) {
        errno = ERANGE;
        return 0;
    }
    return (uint64_t)batch_size |
        ((uint64_t)sequence_length << OPENNPUX_NPU_RUNTIME_SEQUENCE_SHIFT) |
        ((uint64_t)kv_length << OPENNPUX_NPU_RUNTIME_KV_SHIFT) |
        ((uint64_t)active_experts << OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT);
}

uint64_t
opennpux_npu_pack_resource_bindings(
    uint32_t weight_binding, uint32_t state_binding, uint32_t scratch_binding)
{
    if (weight_binding > UINT16_MAX || state_binding > UINT16_MAX ||
        scratch_binding > UINT16_MAX) {
        errno = ERANGE;
        return UINT64_MAX;
    }
    return (uint64_t)weight_binding |
        ((uint64_t)state_binding << OPENNPUX_NPU_RESOURCE_STATE_SHIFT) |
        ((uint64_t)scratch_binding << OPENNPUX_NPU_RESOURCE_SCRATCH_SHIFT);
}

uint32_t
opennpux_npu_submission_checksum(const void *buffer, size_t size)
{
    if (buffer == NULL || size == 0) {
        return 0;
    }
    const uint8_t *bytes = buffer;
    uint32_t checksum = UINT32_C(2166136261);
    const size_t checksum_offset =
        offsetof(struct opennpux_npu_invocation_header, checksum);
    for (size_t index = 0; index < size; ++index) {
        const uint8_t value =
            index >= checksum_offset && index < checksum_offset + sizeof(uint32_t) ?
            0 : bytes[index];
        checksum ^= value;
        checksum *= UINT32_C(16777619);
    }
    return checksum;
}

int
opennpux_npu_submission_begin(
    struct opennpux_npu_submission_builder *builder, void *buffer,
    size_t capacity, uint64_t sequence, uint64_t executable_id,
    uint64_t context_id, uint32_t entry_point, uint32_t binding_count,
    uint32_t command_count)
{
    if (builder == NULL || buffer == NULL || sequence == 0 ||
        executable_id == 0 || binding_count > OPENNPUX_NPU_MAX_BINDINGS ||
        command_count == 0 || command_count > OPENNPUX_NPU_MAX_COMMANDS) {
        errno = EINVAL;
        return -1;
    }
    size_t binding_bytes = 0;
    size_t command_bytes = 0;
    if (array_size(binding_count, sizeof(struct opennpux_npu_tensor_binding),
                   &binding_bytes) != 0 ||
        array_size(command_count, sizeof(struct opennpux_npu_command),
                   &command_bytes) != 0) {
        return -1;
    }
    const size_t binding_offset = align_size(sizeof(struct opennpux_npu_invocation_header));
    const size_t command_offset = align_size(binding_offset + binding_bytes);
    const size_t total_size = align_size(command_offset + command_bytes);
    if (binding_offset == 0 || command_offset == 0 || total_size == 0 ||
        total_size > capacity || total_size > UINT32_MAX) {
        errno = ENOSPC;
        return -1;
    }
    memset(buffer, 0, total_size);
    memset(builder, 0, sizeof(*builder));
    builder->buffer = buffer;
    builder->capacity = capacity;
    builder->header = buffer;
    builder->bindings = (struct opennpux_npu_tensor_binding *)(
        builder->buffer + binding_offset);
    builder->commands = (struct opennpux_npu_command *)(
        builder->buffer + command_offset);

    struct opennpux_npu_invocation_header *header = builder->header;
    header->magic = OPENNPUX_NPU_INVOCATION_MAGIC;
    header->version = OPENNPUX_NPU_INVOCATION_VERSION;
    header->header_size = sizeof(*header);
    header->total_size = (uint32_t)total_size;
    header->sequence = sequence;
    header->executable_id = executable_id;
    header->context_id = context_id;
    header->entry_point = entry_point;
    header->binding_count = binding_count;
    header->command_count = command_count;
    header->binding_offset = binding_offset;
    header->command_offset = command_offset;
    return 0;
}

struct opennpux_npu_tensor_binding *
opennpux_npu_submission_binding(
    struct opennpux_npu_submission_builder *builder, uint32_t index)
{
    if (builder == NULL || builder->header == NULL ||
        index >= builder->header->binding_count) {
        errno = EINVAL;
        return NULL;
    }
    return &builder->bindings[index];
}

struct opennpux_npu_command *
opennpux_npu_submission_command(
    struct opennpux_npu_submission_builder *builder, uint32_t index)
{
    if (builder == NULL || builder->header == NULL ||
        index >= builder->header->command_count) {
        errno = EINVAL;
        return NULL;
    }
    return &builder->commands[index];
}

int
opennpux_npu_submission_validate(const void *buffer, size_t size)
{
    if (buffer == NULL || size < sizeof(struct opennpux_npu_invocation_header)) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_invocation_header *header = buffer;
    size_t binding_bytes = 0;
    size_t command_bytes = 0;
    if (header->magic != OPENNPUX_NPU_INVOCATION_MAGIC ||
        header->version != OPENNPUX_NPU_INVOCATION_VERSION ||
        header->header_size != sizeof(*header) || header->total_size > size ||
        header->total_size < sizeof(*header) || header->sequence == 0 ||
        header->executable_id == 0 ||
        header->binding_count > OPENNPUX_NPU_MAX_BINDINGS ||
        header->command_count == 0 ||
        header->command_count > OPENNPUX_NPU_MAX_COMMANDS ||
        header->binding_offset % OPENNPUX_NPU_RECORD_ALIGNMENT != 0 ||
        header->command_offset % OPENNPUX_NPU_RECORD_ALIGNMENT != 0 ||
        array_size(header->binding_count,
                   sizeof(struct opennpux_npu_tensor_binding),
                   &binding_bytes) != 0 ||
        array_size(header->command_count, sizeof(struct opennpux_npu_command),
                   &command_bytes) != 0 ||
        header->binding_offset > header->total_size ||
        binding_bytes > header->total_size - header->binding_offset ||
        header->command_offset > header->total_size ||
        command_bytes > header->total_size - header->command_offset ||
        header->checksum !=
            opennpux_npu_submission_checksum(buffer, header->total_size)) {
        errno = EINVAL;
        return -1;
    }
    const uint8_t *bytes = buffer;
    const struct opennpux_npu_tensor_binding *bindings =
        (const struct opennpux_npu_tensor_binding *)(bytes + header->binding_offset);
    for (uint32_t index = 0; index < header->binding_count; ++index) {
        if (bindings[index].rank > OPENNPUX_NPU_MAX_RANK ||
            bindings[index].data_type == OPENNPUX_NPU_DTYPE_INVALID ||
            bindings[index].byte_size == 0 ||
            (bindings[index].device_address == 0 &&
             bindings[index].memory_object == 0)) {
            errno = EINVAL;
            return -1;
        }
    }
    const struct opennpux_npu_command *commands =
        (const struct opennpux_npu_command *)(bytes + header->command_offset);
    for (uint32_t index = 0; index < header->command_count; ++index) {
        const uint32_t weight_binding =
            commands[index].resource_bindings & OPENNPUX_NPU_RUNTIME_FIELD_MASK;
        const uint32_t state_binding =
            (commands[index].resource_bindings >>
             OPENNPUX_NPU_RESOURCE_STATE_SHIFT) &
            OPENNPUX_NPU_RUNTIME_FIELD_MASK;
        const uint32_t scratch_binding =
            (commands[index].resource_bindings >>
             OPENNPUX_NPU_RESOURCE_SCRATCH_SHIFT) &
            OPENNPUX_NPU_RUNTIME_FIELD_MASK;
        if (commands[index].opcode == OPENNPUX_NPU_OP_INVALID ||
            commands[index].first_binding > header->binding_count ||
            commands[index].binding_count >
                header->binding_count - commands[index].first_binding ||
            commands[index].parameter_symbol == 0 ||
            (commands[index].runtime_shape &
             OPENNPUX_NPU_RUNTIME_FIELD_MASK) == 0 ||
            weight_binding >= header->binding_count ||
            state_binding >= header->binding_count ||
            scratch_binding >= header->binding_count ||
            (bindings[weight_binding].flags &
             (OPENNPUX_NPU_BIND_READ | OPENNPUX_NPU_BIND_WEIGHT)) !=
                (OPENNPUX_NPU_BIND_READ | OPENNPUX_NPU_BIND_WEIGHT) ||
            (bindings[state_binding].flags &
             OPENNPUX_NPU_BIND_PERSISTENT) == 0 ||
            (bindings[scratch_binding].flags &
             OPENNPUX_NPU_BIND_WRITE) == 0 ||
            (header->parameter_size != 0 &&
             commands[index].parameter_offset > header->parameter_size) ||
            (header->parameter_size != 0 && commands[index].parameter_size >
                header->parameter_size - commands[index].parameter_offset)) {
            errno = EINVAL;
            return -1;
        }
    }
    return 0;
}

int
opennpux_npu_submission_finalize(
    struct opennpux_npu_submission_builder *builder)
{
    if (builder == NULL || builder->header == NULL) {
        errno = EINVAL;
        return -1;
    }
    builder->header->checksum = 0;
    builder->header->checksum = opennpux_npu_submission_checksum(
        builder->buffer, builder->header->total_size);
    return opennpux_npu_submission_validate(
        builder->buffer, builder->header->total_size);
}
