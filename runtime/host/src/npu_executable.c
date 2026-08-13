#include "opennpux/npu_executable.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t
executable_checksum(const void *buffer, size_t size)
{
    const uint8_t *bytes = buffer;
    const size_t offset =
        offsetof(struct opennpux_npu_executable_header, checksum);
    uint32_t checksum = UINT32_C(2166136261);
    for (size_t index = 0; index < size; ++index) {
        const uint8_t value =
            index >= offset && index < offset + sizeof(uint32_t) ? 0 : bytes[index];
        checksum ^= value;
        checksum *= UINT32_C(16777619);
    }
    return checksum;
}

static int
range_valid(uint64_t offset, uint64_t count, uint64_t record_size,
            uint64_t total_size)
{
    return offset <= total_size &&
        (count == 0 || record_size <= UINT64_MAX / count) &&
        count * record_size <= total_size - offset;
}

int
opennpux_npu_executable_load(
    const char *path, struct opennpux_npu_executable *executable)
{
    if (path == NULL || executable == NULL) {
        errno = EINVAL;
        return -1;
    }
    memset(executable, 0, sizeof(*executable));
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    const long length = ftell(file);
    if (length < (long)sizeof(struct opennpux_npu_executable_header) ||
        (unsigned long)length > UINT32_MAX ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        errno = EINVAL;
        return -1;
    }
    void *storage = malloc((size_t)length);
    if (storage == NULL) {
        fclose(file);
        errno = ENOMEM;
        return -1;
    }
    if (fread(storage, 1, (size_t)length, file) != (size_t)length) {
        free(storage);
        fclose(file);
        errno = EIO;
        return -1;
    }
    fclose(file);
    const struct opennpux_npu_executable_header *header = storage;
    if (header->magic != OPENNPUX_NPU_EXECUTABLE_MAGIC ||
        header->version != OPENNPUX_NPU_EXECUTABLE_VERSION ||
        header->header_size != sizeof(*header) ||
        header->total_size != (uint32_t)length || header->executable_id == 0 ||
        header->entry_count == 0 ||
        header->entry_count > OPENNPUX_NPU_MAX_ENTRY_POINTS ||
        header->command_count == 0 ||
        header->command_count > OPENNPUX_NPU_MAX_COMMANDS ||
        header->entry_record_size != sizeof(struct opennpux_npu_executable_entry) ||
        header->command_record_size != sizeof(struct opennpux_npu_command_template) ||
        !range_valid(header->entry_offset, header->entry_count,
                     header->entry_record_size, header->total_size) ||
        !range_valid(header->command_offset, header->command_count,
                     header->command_record_size, header->total_size) ||
        header->checksum != executable_checksum(storage, (size_t)length)) {
        free(storage);
        errno = EINVAL;
        return -1;
    }
    executable->storage = storage;
    executable->storage_size = (size_t)length;
    executable->header = header;
    executable->entries = (const struct opennpux_npu_executable_entry *)(
        (const uint8_t *)storage + header->entry_offset);
    executable->commands = (const struct opennpux_npu_command_template *)(
        (const uint8_t *)storage + header->command_offset);
    for (uint32_t index = 0; index < header->entry_count; ++index) {
        const struct opennpux_npu_executable_entry *entry = &executable->entries[index];
        if (entry->command_count == 0 ||
            entry->first_command > header->command_count ||
            entry->command_count > header->command_count - entry->first_command) {
            opennpux_npu_executable_unload(executable);
            errno = EINVAL;
            return -1;
        }
        for (uint32_t previous = 0; previous < index; ++previous) {
            if (executable->entries[previous].entry_point == entry->entry_point) {
                opennpux_npu_executable_unload(executable);
                errno = EINVAL;
                return -1;
            }
        }
    }
    for (uint32_t index = 0; index < header->command_count; ++index) {
        const struct opennpux_npu_command_template *command =
            &executable->commands[index];
        if (command->opcode == OPENNPUX_NPU_OP_INVALID ||
            command->binding_count > OPENNPUX_NPU_MAX_BINDINGS ||
            command->first_binding > OPENNPUX_NPU_MAX_BINDINGS -
                command->binding_count) {
            opennpux_npu_executable_unload(executable);
            errno = EINVAL;
            return -1;
        }
    }
    return 0;
}

void
opennpux_npu_executable_unload(struct opennpux_npu_executable *executable)
{
    if (executable != NULL) {
        free(executable->storage);
        memset(executable, 0, sizeof(*executable));
    }
}

const struct opennpux_npu_executable_entry *
opennpux_npu_executable_find_entry(
    const struct opennpux_npu_executable *executable, uint32_t entry_point)
{
    if (executable == NULL || executable->header == NULL) {
        errno = EINVAL;
        return NULL;
    }
    for (uint32_t index = 0; index < executable->header->entry_count; ++index) {
        if (executable->entries[index].entry_point == entry_point) {
            return &executable->entries[index];
        }
    }
    errno = ENOENT;
    return NULL;
}

int
opennpux_npu_executable_instantiate(
    const struct opennpux_npu_executable *executable, uint32_t entry_point,
    uint64_t sequence, uint64_t context_id,
    const struct opennpux_npu_tensor_binding *bindings, uint32_t binding_count,
    void *submission, size_t submission_capacity, size_t *submission_size)
{
    if (executable == NULL || executable->header == NULL || bindings == NULL ||
        binding_count == 0 || submission == NULL || submission_size == NULL) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_executable_entry *entry =
        opennpux_npu_executable_find_entry(executable, entry_point);
    if (entry == NULL) {
        return -1;
    }
    struct opennpux_npu_submission_builder builder;
    if (opennpux_npu_submission_begin(
            &builder, submission, submission_capacity, sequence,
            executable->header->executable_id, context_id, entry_point,
            binding_count, entry->command_count) != 0) {
        return -1;
    }
    memcpy(builder.bindings, bindings,
           binding_count * sizeof(struct opennpux_npu_tensor_binding));
    for (uint32_t index = 0; index < entry->command_count; ++index) {
        const struct opennpux_npu_command_template *source =
            &executable->commands[entry->first_command + index];
        struct opennpux_npu_command *destination = &builder.commands[index];
        destination->command_id = source->command_id;
        destination->opcode = source->opcode;
        destination->flags = source->flags;
        destination->capability_id = source->capability_id;
        destination->first_binding = source->first_binding;
        destination->binding_count = source->binding_count;
        destination->dependency_token = source->dependency_token;
        destination->completion_token = source->completion_token;
        destination->estimated_operations = source->estimated_operations;
        destination->estimated_bytes = source->estimated_bytes;
        destination->profiling_tag = source->profiling_tag;
    }
    for (uint32_t index = 0; index < binding_count; ++index) {
        if ((bindings[index].flags & OPENNPUX_NPU_BIND_PERSISTENT) != 0) {
            builder.header->persistent_state_handle =
                bindings[index].memory_object;
            break;
        }
    }
    builder.header->flags = OPENNPUX_NPU_INVOKE_PROFILE;
    if (opennpux_npu_submission_finalize(&builder) != 0) {
        return -1;
    }
    *submission_size = builder.header->total_size;
    return 0;
}
