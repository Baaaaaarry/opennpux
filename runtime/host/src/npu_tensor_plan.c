#include "opennpux/npu_tensor_plan.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
range_valid(uint64_t offset, uint64_t count, uint64_t record_size,
            uint64_t total_size)
{
    return offset <= total_size &&
        (count == 0 || record_size <= UINT64_MAX / count) &&
        count * record_size <= total_size - offset;
}

static uint32_t
plan_checksum(const void *buffer, size_t size)
{
    const uint8_t *bytes = buffer;
    const size_t offset =
        offsetof(struct opennpux_npu_tensor_plan_header, checksum);
    uint32_t value = UINT32_C(2166136261);
    for (size_t index = 0; index < size; ++index) {
        const uint8_t byte =
            index >= offset && index < offset + sizeof(uint32_t) ? 0 : bytes[index];
        value ^= byte;
        value *= UINT32_C(16777619);
    }
    return value;
}

static int
multiply_size(uint64_t value, uint64_t factor, uint64_t *result)
{
    if (factor != 0 && value > UINT64_MAX / factor) {
        errno = EOVERFLOW;
        return -1;
    }
    *result = value * factor;
    return 0;
}

static int
tensor_size(const struct opennpux_npu_tensor_plan_tensor *tensor,
            const struct opennpux_npu_tensor_plan_runtime *runtime,
            uint64_t *size)
{
    if (tensor == NULL || runtime == NULL || size == NULL ||
        runtime->batch_size == 0 || runtime->sequence_length == 0) {
        errno = EINVAL;
        return -1;
    }
    uint64_t elements = 1;
    for (uint32_t index = 0; index < tensor->rank; ++index) {
        const uint32_t symbol = (tensor->dimension_symbols >> (index * 4)) & 0xf;
        uint64_t dimension = tensor->dimensions[index];
        switch (symbol) {
        case OPENNPUX_NPU_DIMENSION_STATIC:
            break;
        case OPENNPUX_NPU_DIMENSION_BATCH:
            dimension = runtime->batch_size;
            break;
        case OPENNPUX_NPU_DIMENSION_SEQUENCE:
            dimension = runtime->sequence_length;
            break;
        case OPENNPUX_NPU_DIMENSION_KV:
            dimension = runtime->kv_length;
            break;
        case OPENNPUX_NPU_DIMENSION_ACTIVE_EXPERTS:
            dimension = runtime->active_experts;
            break;
        default:
            errno = EINVAL;
            return -1;
        }
        if (dimension == 0 || multiply_size(elements, dimension, &elements) != 0) {
            errno = EINVAL;
            return -1;
        }
    }
    uint64_t element_bytes = 0;
    switch (tensor->data_type) {
    case 2: /* int8 */
        element_bytes = 1;
        break;
    case 3: /* int32 */
    case 6: /* float32 */
        element_bytes = 4;
        break;
    case 4: /* float16 */
    case 5: /* bfloat16 */
        element_bytes = 2;
        break;
    default:
        errno = EINVAL;
        return -1;
    }
    return multiply_size(elements, element_bytes, size);
}

static int
align_u64(uint64_t value, uint64_t *result)
{
    const uint64_t alignment = 64;
    if (value > UINT64_MAX - (alignment - 1)) {
        errno = EOVERFLOW;
        return -1;
    }
    *result = (value + alignment - 1) & ~(alignment - 1);
    return 0;
}

static int
validate_plan(struct opennpux_npu_tensor_plan *plan)
{
    const struct opennpux_npu_tensor_plan_header *header = plan->header;
    if (header->magic != OPENNPUX_NPU_TENSOR_PLAN_MAGIC ||
        header->version != OPENNPUX_NPU_TENSOR_PLAN_VERSION ||
        header->header_size != sizeof(*header) ||
        header->total_size != plan->storage_size ||
        header->tensor_count == 0 || header->command_count == 0 ||
        header->tensor_record_size != sizeof(*plan->tensors) ||
        header->command_record_size != sizeof(*plan->commands) ||
        header->slot_record_size != sizeof(*plan->slots) ||
        !range_valid(header->tensor_offset, header->tensor_count,
                     sizeof(*plan->tensors), header->total_size) ||
        !range_valid(header->command_offset, header->command_count,
                     sizeof(*plan->commands), header->total_size) ||
        !range_valid(header->slot_offset, header->slot_count,
                     sizeof(*plan->slots), header->total_size) ||
        header->checksum != plan_checksum(plan->storage, plan->storage_size)) {
        errno = EINVAL;
        return -1;
    }
    for (uint32_t index = 0; index < header->slot_count; ++index) {
        if (plan->slots[index].slot_id != index ||
            plan->slots[index].bytes_per_runtime_row == 0) {
            errno = EINVAL;
            return -1;
        }
    }
    for (uint32_t index = 0; index < header->tensor_count; ++index) {
        const struct opennpux_npu_tensor_plan_tensor *tensor = &plan->tensors[index];
        if (tensor->tensor_id != index || tensor->rank > OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK ||
            tensor->storage < OPENNPUX_NPU_TENSOR_INPUT ||
            tensor->storage > OPENNPUX_NPU_TENSOR_PERSISTENT ||
            (tensor->producer_command != OPENNPUX_NPU_TENSOR_NONE &&
             tensor->producer_command >= header->command_count) ||
            (tensor->producer_command != OPENNPUX_NPU_TENSOR_NONE &&
             tensor->last_consumer_command < tensor->producer_command) ||
            (tensor->storage == OPENNPUX_NPU_TENSOR_SCRATCH &&
             (tensor->allocation_slot >= header->slot_count ||
              tensor->bytes_per_runtime_row == 0 ||
              tensor->bytes_per_runtime_row >
                  plan->slots[tensor->allocation_slot].bytes_per_runtime_row))) {
            errno = EINVAL;
            return -1;
        }
    }
    for (uint32_t index = 0; index < header->command_count; ++index) {
        const struct opennpux_npu_tensor_plan_command *command = &plan->commands[index];
        if (command->command_id != index || command->input_count == 0 ||
            command->input_count > OPENNPUX_NPU_TENSOR_PLAN_MAX_INPUTS ||
            command->output_count == 0 ||
            command->output_count > OPENNPUX_NPU_TENSOR_PLAN_MAX_OUTPUTS) {
            errno = EINVAL;
            return -1;
        }
        for (uint32_t input = 0; input < command->input_count; ++input) {
            const uint32_t tensor_id = command->input_tensor_ids[input];
            if (tensor_id >= header->tensor_count ||
                (plan->tensors[tensor_id].producer_command != OPENNPUX_NPU_TENSOR_NONE &&
                 plan->tensors[tensor_id].producer_command >= index)) {
                errno = EINVAL;
                return -1;
            }
        }
        for (uint32_t output = 0; output < command->output_count; ++output) {
            const uint32_t tensor_id = command->output_tensor_ids[output];
            if (tensor_id >= header->tensor_count ||
                plan->tensors[tensor_id].producer_command != index) {
                errno = EINVAL;
                return -1;
            }
        }
    }
    return 0;
}

int
opennpux_npu_tensor_plan_load(
    const char *path, struct opennpux_npu_tensor_plan *plan)
{
    if (path == NULL || plan == NULL) {
        errno = EINVAL;
        return -1;
    }
    memset(plan, 0, sizeof(*plan));
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    const long length = ftell(file);
    if (length < (long)sizeof(struct opennpux_npu_tensor_plan_header) ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        errno = EINVAL;
        return -1;
    }
    void *storage = malloc((size_t)length);
    if (storage == NULL) {
        fclose(file);
        return -1;
    }
    const size_t read_size = fread(storage, 1, (size_t)length, file);
    const int close_result = fclose(file);
    if (read_size != (size_t)length || close_result != 0) {
        free(storage);
        errno = EIO;
        return -1;
    }
    plan->storage = storage;
    plan->storage_size = (size_t)length;
    plan->header = storage;
    plan->tensors = (const struct opennpux_npu_tensor_plan_tensor *)(
        (const uint8_t *)storage + plan->header->tensor_offset);
    plan->commands = (const struct opennpux_npu_tensor_plan_command *)(
        (const uint8_t *)storage + plan->header->command_offset);
    plan->slots = (const struct opennpux_npu_tensor_plan_slot *)(
        (const uint8_t *)storage + plan->header->slot_offset);
    if (validate_plan(plan) != 0) {
        opennpux_npu_tensor_plan_unload(plan);
        return -1;
    }
    return 0;
}

void
opennpux_npu_tensor_plan_unload(struct opennpux_npu_tensor_plan *plan)
{
    if (plan != NULL) {
        free(plan->storage);
        memset(plan, 0, sizeof(*plan));
    }
}

int
opennpux_npu_tensor_plan_scratch_size(
    const struct opennpux_npu_tensor_plan *plan, uint32_t batch_size,
    uint32_t sequence_length, uint64_t *size)
{
    if (plan == NULL || plan->header == NULL || size == NULL || batch_size == 0 ||
        sequence_length == 0 || batch_size > UINT64_MAX / sequence_length) {
        errno = EINVAL;
        return -1;
    }
    const uint64_t rows = (uint64_t)batch_size * sequence_length;
    if (plan->header->scratch_bytes_per_runtime_row > UINT64_MAX / rows) {
        errno = EOVERFLOW;
        return -1;
    }
    *size = plan->header->scratch_bytes_per_runtime_row * rows;
    return 0;
}

int
opennpux_npu_tensor_plan_resolve_scratch(
    const struct opennpux_npu_tensor_plan *plan, uint32_t tensor_id,
    uint32_t batch_size, uint32_t sequence_length, uint64_t scratch_address,
    uint64_t scratch_size, uint64_t *tensor_address, uint64_t *tensor_size)
{
    if (plan == NULL || plan->header == NULL || tensor_id >= plan->header->tensor_count ||
        tensor_address == NULL || tensor_size == NULL) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_tensor_plan_tensor *tensor = &plan->tensors[tensor_id];
    if (tensor->storage != OPENNPUX_NPU_TENSOR_SCRATCH) {
        errno = EINVAL;
        return -1;
    }
    uint64_t required = 0;
    if (opennpux_npu_tensor_plan_scratch_size(
            plan, batch_size, sequence_length, &required) != 0 ||
        required > scratch_size) {
        errno = ENOSPC;
        return -1;
    }
    const uint64_t rows = (uint64_t)batch_size * sequence_length;
    uint64_t offset_per_row = 0;
    for (uint32_t index = 0; index < tensor->allocation_slot; ++index) {
        offset_per_row += plan->slots[index].bytes_per_runtime_row;
    }
    if (offset_per_row > UINT64_MAX / rows ||
        tensor->bytes_per_runtime_row > UINT64_MAX / rows) {
        errno = EOVERFLOW;
        return -1;
    }
    const uint64_t offset = offset_per_row * rows;
    *tensor_size = tensor->bytes_per_runtime_row * rows;
    if (offset > scratch_size || *tensor_size > scratch_size - offset ||
        scratch_address > UINT64_MAX - offset) {
        errno = ENOSPC;
        return -1;
    }
    *tensor_address = scratch_address + offset;
    return 0;
}

int
opennpux_npu_tensor_plan_persistent_size(
    const struct opennpux_npu_tensor_plan *plan,
    const struct opennpux_npu_tensor_plan_runtime *runtime, uint64_t *size)
{
    if (plan == NULL || plan->header == NULL || runtime == NULL || size == NULL) {
        errno = EINVAL;
        return -1;
    }
    uint64_t total = 0;
    for (uint32_t index = 0; index < plan->header->tensor_count; ++index) {
        if (plan->tensors[index].storage != OPENNPUX_NPU_TENSOR_PERSISTENT) {
            continue;
        }
        uint64_t bytes = 0;
        if (tensor_size(&plan->tensors[index], runtime, &bytes) != 0 ||
            total > UINT64_MAX - bytes || align_u64(total + bytes, &total) != 0) {
            return -1;
        }
    }
    *size = total;
    return 0;
}

int
opennpux_npu_tensor_plan_resolve(
    const struct opennpux_npu_tensor_plan *plan, uint32_t tensor_id,
    const struct opennpux_npu_tensor_plan_runtime *runtime,
    const struct opennpux_npu_tensor_plan_memory *memory,
    uint64_t *tensor_address, uint64_t *resolved_size)
{
    if (plan == NULL || plan->header == NULL || tensor_id >= plan->header->tensor_count ||
        runtime == NULL || memory == NULL || tensor_address == NULL ||
        resolved_size == NULL) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_npu_tensor_plan_tensor *tensor = &plan->tensors[tensor_id];
    uint64_t size = 0;
    if (tensor_size(tensor, runtime, &size) != 0) {
        return -1;
    }
    uint64_t address = 0;
    uint64_t capacity = 0;
    if (tensor->storage == OPENNPUX_NPU_TENSOR_INPUT) {
        address = memory->input_address;
        capacity = memory->input_size;
    } else if (tensor->storage == OPENNPUX_NPU_TENSOR_OUTPUT) {
        address = memory->output_address;
        capacity = memory->output_size;
    } else if (tensor->storage == OPENNPUX_NPU_TENSOR_SCRATCH) {
        return opennpux_npu_tensor_plan_resolve_scratch(
            plan, tensor_id, runtime->batch_size, runtime->sequence_length,
            memory->scratch_address, memory->scratch_size, tensor_address,
            resolved_size);
    } else if (tensor->storage == OPENNPUX_NPU_TENSOR_PERSISTENT) {
        uint64_t offset = 0;
        for (uint32_t index = 0; index < tensor_id; ++index) {
            if (plan->tensors[index].storage != OPENNPUX_NPU_TENSOR_PERSISTENT) {
                continue;
            }
            uint64_t previous_size = 0;
            if (tensor_size(&plan->tensors[index], runtime, &previous_size) != 0 ||
                offset > UINT64_MAX - previous_size ||
                align_u64(offset + previous_size, &offset) != 0) {
                return -1;
            }
        }
        if (offset > memory->persistent_size ||
            size > memory->persistent_size - offset ||
            memory->persistent_address > UINT64_MAX - offset) {
            errno = ENOSPC;
            return -1;
        }
        *tensor_address = memory->persistent_address + offset;
        *resolved_size = size;
        return 0;
    } else {
        errno = EINVAL;
        return -1;
    }
    if (address == 0 || size > capacity) {
        errno = ENOSPC;
        return -1;
    }
    *tensor_address = address;
    *resolved_size = size;
    return 0;
}
