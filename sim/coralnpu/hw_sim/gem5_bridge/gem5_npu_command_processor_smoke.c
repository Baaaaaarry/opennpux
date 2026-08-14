#include <stddef.h>
#include <stdint.h>

#include "hw_sim/gem5_bridge/npu_submission.h"
#include "hw_sim/gem5_bridge/npu_weight_queue.h"

#define EXTMEM_BASE UINT32_C(0x20000000)
#define COMMAND_BUFFER_SIZE UINT32_C(0x00800000)
#define ERROR_ABI UINT32_C(1)
#define ERROR_BOUNDS UINT32_C(2)
#define ERROR_CHECKSUM UINT32_C(3)
#define ERROR_COMMAND UINT32_C(4)
#define ERROR_RELOCATION UINT32_C(5)
#define ERROR_DEPENDENCY UINT32_C(6)
#define ERROR_PAGING UINT32_C(7)
#define MODELED_OPS_PER_CYCLE UINT64_C(256)
#define MODELED_BYTES_PER_CYCLE UINT64_C(32)
#define PAGING_TRANSFER_SIZE UINT32_C(65536)
#define EXPERT_NONE UINT64_MAX

static uint32_t
align_record(uint32_t value)
{
    return (value + OPENNPUX_NPU_RECORD_ALIGNMENT - 1) &
        ~(OPENNPUX_NPU_RECORD_ALIGNMENT - 1);
}

static uint32_t
checksum(const volatile uint8_t *bytes, uint32_t size)
{
    const uint32_t begin = (uint32_t)offsetof(
        struct opennpux_npu_invocation_header, checksum);
    uint32_t value = UINT32_C(2166136261);
    for (uint32_t index = 0; index < size; ++index) {
        uint8_t byte = index >= begin && index < begin + 4 ? 0 : bytes[index];
        value ^= byte;
        value *= UINT32_C(16777619);
    }
    return value;
}

static int
range_valid(uint64_t offset, uint32_t count, uint32_t record_size,
            uint32_t total_size)
{
    const uint64_t bytes = (uint64_t)count * record_size;
    return offset <= total_size && bytes <= total_size - offset;
}

static void
finish(volatile struct opennpux_npu_completion *completion, uint32_t state,
       uint32_t error, uint32_t fault)
{
    completion->error_code = error;
    completion->faulting_command = fault;
    __asm__ volatile("" ::: "memory");
    completion->state = state;
}

static void
memory_fence(void)
{
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static int
page_weight(volatile struct opennpux_npu_weight_queue_header *queue,
            volatile uint8_t *cache, uint32_t cache_slots,
            uint32_t command_id, uint32_t *word,
            volatile uint64_t *stall_cycles)
{
    if (queue->magic != OPENNPUX_NPU_WEIGHT_QUEUE_MAGIC ||
        queue->version != OPENNPUX_NPU_WEIGHT_QUEUE_VERSION ||
        queue->header_size != sizeof(*queue) ||
        queue->entry_size != sizeof(struct opennpux_npu_page_fault) ||
        queue->capacity == 0 || cache_slots == 0) {
        return -1;
    }
    uint32_t producer = queue->producer_index;
    while (producer - queue->retire_index >= queue->capacity) {
        ++queue->backpressure_count;
        ++*stall_cycles;
    }
    volatile struct opennpux_npu_page_fault *entries =
        (volatile struct opennpux_npu_page_fault *)(queue + 1);
    volatile struct opennpux_npu_page_fault *fault =
        &entries[producer % queue->capacity];
    while (fault->state != OPENNPUX_NPU_PAGE_FAULT_EMPTY) {
        ++*stall_cycles;
    }
    fault->magic = OPENNPUX_NPU_PAGE_FAULT_MAGIC;
    fault->version = OPENNPUX_NPU_PAGE_FAULT_VERSION;
    fault->struct_size = sizeof(*fault);
    fault->sequence = (uint64_t)command_id + 1;
    fault->command_id = command_id;
    fault->shard_index = 0;
    fault->file_offset = (uint64_t)command_id * PAGING_TRANSFER_SIZE;
    fault->expert_id = EXPERT_NONE;
    fault->cache_slot = 0;
    fault->error_code = 0;
    fault->page_size = PAGING_TRANSFER_SIZE;
    fault->reserved = 0;
    memory_fence();
    fault->state = OPENNPUX_NPU_PAGE_FAULT_PENDING;
    memory_fence();
    queue->producer_index = producer + 1;
    memory_fence();
    while (fault->state == OPENNPUX_NPU_PAGE_FAULT_PENDING) {
        ++*stall_cycles;
    }
    if (fault->state != OPENNPUX_NPU_PAGE_FAULT_READY ||
        fault->error_code != 0 || fault->cache_slot >= cache_slots) {
        return -1;
    }
    const uint32_t offset = command_id * sizeof(uint32_t);
    if (offset > PAGING_TRANSFER_SIZE - sizeof(uint32_t)) {
        return -1;
    }
    const volatile uint32_t *cache_word =
        (const volatile uint32_t *)(cache +
            fault->cache_slot * PAGING_TRANSFER_SIZE + offset);
    *word = *cache_word;
    fault->state = OPENNPUX_NPU_PAGE_FAULT_EMPTY;
    memory_fence();
    queue->retire_index = producer + 1;
    memory_fence();
    return 0;
}

int
main(void)
{
    volatile uint8_t *base = (volatile uint8_t *)EXTMEM_BASE;
    volatile struct opennpux_npu_invocation_header *header =
        (volatile struct opennpux_npu_invocation_header *)base;
    if (header->completion_address < EXTMEM_BASE ||
        header->completion_address >
            EXTMEM_BASE + COMMAND_BUFFER_SIZE - sizeof(struct opennpux_npu_completion)) {
        return 1;
    }
    volatile struct opennpux_npu_completion *completion =
        (volatile struct opennpux_npu_completion *)(uintptr_t)
            header->completion_address;
    completion->magic = OPENNPUX_NPU_COMPLETION_MAGIC;
    completion->version = OPENNPUX_NPU_COMPLETION_VERSION;
    completion->struct_size = sizeof(*completion);
    completion->sequence = header->sequence;
    completion->state = OPENNPUX_NPU_COMPLETION_RUNNING;

    const uint32_t trace_offset = align_record(
        (uint32_t)(header->completion_address - EXTMEM_BASE) +
        sizeof(struct opennpux_npu_completion));
    const uint32_t trace_size = sizeof(struct opennpux_npu_trace_header) +
        OPENNPUX_NPU_TRACE_MAX_OPCODE *
            sizeof(struct opennpux_npu_trace_record);
    if (trace_offset > COMMAND_BUFFER_SIZE ||
        trace_size > COMMAND_BUFFER_SIZE - trace_offset) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_BOUNDS, 0);
        return 1;
    }
    volatile struct opennpux_npu_trace_header *trace =
        (volatile struct opennpux_npu_trace_header *)(base + trace_offset);
    volatile struct opennpux_npu_trace_record *records =
        (volatile struct opennpux_npu_trace_record *)(base + trace_offset +
                                                      sizeof(*trace));
    for (uint32_t index = 0; index < trace_size; ++index) {
        base[trace_offset + index] = 0;
    }
    trace->magic = OPENNPUX_NPU_TRACE_MAGIC;
    trace->version = OPENNPUX_NPU_TRACE_VERSION;
    trace->struct_size = sizeof(*trace);
    trace->record_count = OPENNPUX_NPU_TRACE_MAX_OPCODE;
    completion->trace_address = EXTMEM_BASE + trace_offset;
    completion->trace_size = trace_size;

    if (header->magic != OPENNPUX_NPU_INVOCATION_MAGIC ||
        header->version != OPENNPUX_NPU_INVOCATION_VERSION ||
        header->header_size != sizeof(*header) || header->total_size < sizeof(*header) ||
        header->total_size > COMMAND_BUFFER_SIZE || header->command_count == 0) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_ABI, 0);
        return 1;
    }
    if (!range_valid(header->binding_offset, header->binding_count,
                     sizeof(struct opennpux_npu_tensor_binding), header->total_size) ||
        !range_valid(header->command_offset, header->command_count,
                     sizeof(struct opennpux_npu_command), header->total_size)) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_BOUNDS, 0);
        return 1;
    }
    if (checksum(base, header->total_size) != header->checksum) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_CHECKSUM, 0);
        return 1;
    }

    volatile struct opennpux_npu_command *commands =
        (volatile struct opennpux_npu_command *)(base + header->command_offset);
    volatile struct opennpux_npu_tensor_binding *bindings =
        (volatile struct opennpux_npu_tensor_binding *)(base +
                                                        header->binding_offset);
    volatile struct opennpux_npu_tensor_binding *queue_binding = NULL;
    volatile struct opennpux_npu_tensor_binding *cache_binding = NULL;
    for (uint32_t index = 0; index < header->binding_count; ++index) {
        if ((bindings[index].flags & OPENNPUX_NPU_BIND_PAGE_QUEUE) != 0) {
            if (queue_binding != NULL) {
                finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                       ERROR_PAGING, 0);
                return 1;
            }
            queue_binding = &bindings[index];
        }
        if ((bindings[index].flags & OPENNPUX_NPU_BIND_PAGE_CACHE) != 0) {
            if (cache_binding != NULL) {
                finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                       ERROR_PAGING, 0);
                return 1;
            }
            cache_binding = &bindings[index];
        }
    }
    if ((queue_binding == NULL) != (cache_binding == NULL)) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_PAGING, 0);
        return 1;
    }
    if (queue_binding != NULL &&
        (queue_binding->device_address < EXTMEM_BASE ||
         queue_binding->byte_size < sizeof(struct opennpux_npu_weight_queue_header) ||
         queue_binding->device_address > EXTMEM_BASE + COMMAND_BUFFER_SIZE ||
         queue_binding->byte_size > EXTMEM_BASE + COMMAND_BUFFER_SIZE -
             queue_binding->device_address ||
         cache_binding->device_address < EXTMEM_BASE ||
         cache_binding->byte_size < PAGING_TRANSFER_SIZE ||
         cache_binding->device_address > EXTMEM_BASE + COMMAND_BUFFER_SIZE ||
         cache_binding->byte_size > EXTMEM_BASE + COMMAND_BUFFER_SIZE -
             cache_binding->device_address)) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_PAGING, 0);
        return 1;
    }
    uint64_t cycles = 0;
    uint32_t relocated = 0;
    uint32_t parameter_checksum = UINT32_C(2166136261);
    uint32_t retired_token = 0;
    trace->weight_checksum = UINT32_C(2166136261);
    for (uint32_t index = 0; index < header->command_count; ++index) {
        if (commands[index].opcode == 0 ||
            commands[index].opcode > OPENNPUX_NPU_TRACE_MAX_OPCODE ||
            commands[index].capability_id == 0 ||
            commands[index].first_binding > header->binding_count ||
            commands[index].binding_count >
                header->binding_count - commands[index].first_binding) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_COMMAND, index);
            return 1;
        }
        const uint32_t batch_size =
            commands[index].runtime_shape & OPENNPUX_NPU_RUNTIME_FIELD_MASK;
        const uint32_t sequence_length =
            (commands[index].runtime_shape >>
             OPENNPUX_NPU_RUNTIME_SEQUENCE_SHIFT) &
            OPENNPUX_NPU_RUNTIME_FIELD_MASK;
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
        if (commands[index].parameter_symbol == 0 || batch_size == 0 ||
            sequence_length == 0 || weight_binding >= header->binding_count ||
            state_binding >= header->binding_count ||
            scratch_binding >= header->binding_count ||
            (bindings[weight_binding].flags &
             (OPENNPUX_NPU_BIND_READ | OPENNPUX_NPU_BIND_WEIGHT)) !=
                (OPENNPUX_NPU_BIND_READ | OPENNPUX_NPU_BIND_WEIGHT) ||
            (bindings[state_binding].flags & OPENNPUX_NPU_BIND_PERSISTENT) == 0 ||
            (bindings[scratch_binding].flags & OPENNPUX_NPU_BIND_WRITE) == 0) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_RELOCATION, index);
            return 1;
        }
        if ((commands[index].dependency_token != 0 &&
             commands[index].dependency_token > retired_token) ||
            commands[index].completion_token <= retired_token) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_DEPENDENCY, index);
            return 1;
        }
        if (commands[index].dependency_token != 0) {
            ++trace->dependency_edges;
        }
        retired_token = commands[index].completion_token;
        parameter_checksum ^= (uint32_t)commands[index].parameter_symbol;
        parameter_checksum *= UINT32_C(16777619);
        ++relocated;
        const uint64_t token_count =
            (uint64_t)batch_size * sequence_length;
        const uint64_t operations =
            commands[index].estimated_operations * token_count;
        const uint64_t command_bytes =
            commands[index].estimated_bytes * token_count;
        cycles += (operations + MODELED_OPS_PER_CYCLE - 1) /
                MODELED_OPS_PER_CYCLE +
            (command_bytes + MODELED_BYTES_PER_CYCLE - 1) /
                MODELED_BYTES_PER_CYCLE + 1;
        volatile struct opennpux_npu_trace_record *record =
            &records[commands[index].opcode - 1];
        record->opcode = commands[index].opcode;
        ++record->command_count;
        record->estimated_operations += operations;
        record->estimated_bytes += command_bytes;
        if ((commands[index].flags &
             OPENNPUX_NPU_COMMAND_USES_WEIGHT) != 0) {
            if (queue_binding != NULL) {
                uint32_t word = 0;
                volatile struct opennpux_npu_weight_queue_header *queue =
                    (volatile struct opennpux_npu_weight_queue_header *)(uintptr_t)
                        queue_binding->device_address;
                volatile uint8_t *cache =
                    (volatile uint8_t *)(uintptr_t)cache_binding->device_address;
                const uint32_t cache_slots =
                    (uint32_t)(cache_binding->byte_size / PAGING_TRANSFER_SIZE);
                if (page_weight(queue, cache, cache_slots,
                                commands[index].command_id, &word,
                                &completion->stall_cycles) != 0) {
                    finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                           ERROR_PAGING, index);
                    return 1;
                }
                trace->weight_checksum ^= word;
                trace->weight_checksum *= UINT32_C(16777619);
                ++trace->weight_page_requests;
                trace->weight_dma_bytes += PAGING_TRANSFER_SIZE;
                record->weight_dma_bytes += PAGING_TRANSFER_SIZE;
                goto weight_complete;
            }
            const uint64_t weight_address =
                bindings[weight_binding].device_address;
            const uint64_t weight_size = bindings[weight_binding].byte_size;
            if (weight_size < sizeof(uint32_t) ||
                weight_address < EXTMEM_BASE ||
                weight_address > EXTMEM_BASE + COMMAND_BUFFER_SIZE ||
                weight_size > EXTMEM_BASE + COMMAND_BUFFER_SIZE -
                    weight_address) {
                finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                       ERROR_RELOCATION, index);
                return 1;
            }
            const uint32_t word_count = weight_size / sizeof(uint32_t);
            if (commands[index].command_id >= word_count) {
                finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                       ERROR_RELOCATION, index);
                return 1;
            }
            const uint32_t word_index = commands[index].command_id;
            const volatile uint32_t *weight_word =
                (const volatile uint32_t *)(uintptr_t)(
                    weight_address + word_index * sizeof(uint32_t));
            trace->weight_checksum ^= *weight_word;
            trace->weight_checksum *= UINT32_C(16777619);
            ++trace->weight_page_requests;
            trace->weight_dma_bytes += sizeof(uint32_t);
            record->weight_dma_bytes += sizeof(uint32_t);
weight_complete:
            ;
        }
        trace->capability_mask |= UINT64_C(1) << commands[index].opcode;
        trace->estimated_operations += operations;
        trace->estimated_bytes += command_bytes;
        completion->completed_commands = index + 1;
    }
    trace->command_count = header->command_count;
    completion->npu_cycles = cycles;
    completion->dma_bytes_read = trace->weight_dma_bytes;
    completion->reserved[0] = relocated;
    completion->reserved[1] = parameter_checksum;
    finish(completion, OPENNPUX_NPU_COMPLETION_SUCCESS, 0, UINT32_MAX);
    return 0;
}
