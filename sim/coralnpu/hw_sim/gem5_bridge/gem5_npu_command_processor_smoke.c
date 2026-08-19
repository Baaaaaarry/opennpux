#include <stddef.h>
#include <stdint.h>

#include "hw_sim/gem5_bridge/coral_gptq_paged.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/npu_submission.h"
#include "hw_sim/gem5_bridge/npu_inference_io.h"
#include "hw_sim/gem5_bridge/npu_route_table.h"
#include "hw_sim/gem5_bridge/npu_weight_queue.h"
#include "hw_sim/gem5_bridge/npu_weight_residency.h"

#define EXTMEM_BASE UINT32_C(0x20000000)
#define COMMAND_BUFFER_SIZE UINT32_C(0x00800000)
#define ERROR_ABI UINT32_C(1)
#define ERROR_BOUNDS UINT32_C(2)
#define ERROR_CHECKSUM UINT32_C(3)
#define ERROR_COMMAND UINT32_C(4)
#define ERROR_RELOCATION UINT32_C(5)
#define ERROR_DEPENDENCY UINT32_C(6)
#define ERROR_PAGING UINT32_C(7)
#define ERROR_ROUTE UINT32_C(8)
#define MODELED_OPS_PER_CYCLE UINT64_C(256)
#define MODELED_BYTES_PER_CYCLE UINT64_C(32)
#define PAGING_TRANSFER_SIZE UINT32_C(65536)
#define EXPERT_NONE UINT64_MAX
#define OPERATOR_BASE UINT32_C(0x30000100)
#define OPERATOR_STATUS UINT32_C(0x30000108)

#define PAGE_PROGRESS_WAIT_READY UINT64_C(0x50470001)
#define PAGE_PROGRESS_READY UINT64_C(0x50470002)
#define PAGE_PROGRESS_RESIDENCY_HEADER UINT64_C(0x50470003)
#define PAGE_PROGRESS_RESIDENCY_RECORD UINT64_C(0x50470004)
#define PAGE_PROGRESS_CACHE_LOAD UINT64_C(0x50470005)
#define PAGE_PROGRESS_RETURN UINT64_C(0x50470006)
#define PAGE_PROGRESS_CALL_RETURNED UINT64_C(0x50470007)

static volatile struct opennpux_npu_completion *page_completion;

static void
page_progress(uint64_t marker)
{
    if (page_completion != NULL) {
        page_completion->reserved[0] = marker;
        __asm__ volatile("" ::: "memory");
    }
}

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

static uint32_t
route_checksum(const volatile uint8_t *bytes, uint32_t size)
{
    const uint32_t begin = (uint32_t)offsetof(
        struct opennpux_npu_route_table_header, checksum);
    uint32_t value = UINT32_C(2166136261);
    for (uint32_t index = 0; index < size; ++index) {
        uint8_t byte = index >= begin && index < begin + 4 ? 0 : bytes[index];
        value ^= byte;
        value *= UINT32_C(16777619);
    }
    return value;
}

static uint32_t
byte_checksum(const volatile uint8_t *bytes, uint32_t size)
{
    uint32_t value = UINT32_C(2166136261);
    for (uint32_t index = 0; index < size; ++index) {
        value ^= bytes[index];
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
            volatile const struct opennpux_npu_weight_residency_header *residency,
            uint32_t command_id, uint32_t *word,
            volatile uint64_t *stall_cycles, uint32_t *last,
            uint32_t *role_id, uint32_t *component_id,
            uint64_t *expert_id)
{
    uint32_t local_backpressure_count = 0;
    uint64_t local_stall_cycles = 0;
    if (queue->magic != OPENNPUX_NPU_WEIGHT_QUEUE_MAGIC ||
        queue->version != OPENNPUX_NPU_WEIGHT_QUEUE_VERSION ||
        queue->header_size != sizeof(*queue) ||
        queue->entry_size != sizeof(struct opennpux_npu_page_fault) ||
        queue->capacity == 0 || cache_slots == 0) {
        return -1;
    }
    uint32_t producer = queue->producer_index;
    while (producer - queue->retire_index >= queue->capacity) {
        ++local_backpressure_count;
        ++local_stall_cycles;
    }
    if (local_backpressure_count != 0) {
        queue->backpressure_count += local_backpressure_count;
    }
    volatile struct opennpux_npu_page_fault *entries =
        (volatile struct opennpux_npu_page_fault *)(queue + 1);
    volatile struct opennpux_npu_page_fault *fault =
        &entries[producer % queue->capacity];
    while (fault->state != OPENNPUX_NPU_PAGE_FAULT_EMPTY) {
        ++local_stall_cycles;
    }
    fault->magic = OPENNPUX_NPU_PAGE_FAULT_MAGIC;
    fault->version = OPENNPUX_NPU_PAGE_FAULT_VERSION;
    fault->struct_size = sizeof(*fault);
    fault->sequence = (uint64_t)command_id + 1;
    fault->command_id = command_id;
    fault->shard_index = 0;
    fault->file_offset = (uint64_t)command_id * PAGING_TRANSFER_SIZE;
    fault->expert_id = EXPERT_NONE;
    fault->role_id = 0;
    fault->component_id = 0;
    fault->range_file_offset = 0;
    fault->range_size = 0;
    fault->cache_slot = 0;
    fault->error_code = 0;
    fault->page_size = PAGING_TRANSFER_SIZE;
    fault->flags = 0;
    memory_fence();
    fault->state = OPENNPUX_NPU_PAGE_FAULT_PENDING;
    memory_fence();
    queue->producer_index = producer + 1;
    memory_fence();
    page_progress(PAGE_PROGRESS_WAIT_READY);
    while (fault->state == OPENNPUX_NPU_PAGE_FAULT_PENDING) {
        ++local_stall_cycles;
    }
    if (local_stall_cycles != 0) {
        *stall_cycles += local_stall_cycles;
    }
    page_progress(PAGE_PROGRESS_READY);
    if (fault->state != OPENNPUX_NPU_PAGE_FAULT_READY ||
        fault->error_code != 0 || fault->cache_slot >= cache_slots) {
        return -1;
    }
    if (residency == NULL ||
        residency->magic != OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC ||
        residency->version != OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION ||
        residency->header_size != sizeof(*residency) ||
        residency->record_size !=
            sizeof(struct opennpux_npu_weight_residency_record) ||
        fault->cache_slot >= residency->capacity) {
        return -1;
    }
    page_progress(PAGE_PROGRESS_RESIDENCY_HEADER);
    volatile const struct opennpux_npu_weight_residency_record *resident =
        (volatile const struct opennpux_npu_weight_residency_record *)(
            residency + 1) + fault->cache_slot;
    if ((resident->flags & OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID) == 0 ||
        resident->command_id != fault->command_id ||
        resident->role_id != fault->role_id ||
        resident->component_id != fault->component_id ||
        resident->page_file_offset != fault->file_offset ||
        resident->cache_slot != fault->cache_slot) {
        return -1;
    }
    page_progress(PAGE_PROGRESS_RESIDENCY_RECORD);
    const uint32_t offset = command_id * sizeof(uint32_t);
    if (offset > PAGING_TRANSFER_SIZE - sizeof(uint32_t)) {
        return -1;
    }
    const volatile uint32_t *cache_word =
        (const volatile uint32_t *)(cache +
            fault->cache_slot * PAGING_TRANSFER_SIZE + offset);
    *word = *cache_word;
    page_progress(PAGE_PROGRESS_CACHE_LOAD);
    *last = (fault->flags & OPENNPUX_NPU_PAGE_FAULT_LAST) != 0;
    *role_id = fault->role_id;
    *component_id = fault->component_id;
    *expert_id = fault->expert_id;
    fault->state = OPENNPUX_NPU_PAGE_FAULT_EMPTY;
    memory_fence();
    queue->retire_index = producer + 1;
    memory_fence();
    page_progress(PAGE_PROGRESS_RETURN);
    return 0;
}

static int
execute_paged_matmul(
    uint32_t command_id,
    volatile const struct opennpux_npu_operator_parameters *parameters,
    uint32_t rows, uint32_t role_id, uint64_t expert_id,
    volatile const struct opennpux_npu_tensor_binding *input,
    volatile const struct opennpux_npu_tensor_binding *output,
    volatile const struct opennpux_npu_tensor_binding *scratch,
    volatile const struct opennpux_npu_tensor_binding *residency,
    volatile const struct opennpux_npu_tensor_binding *cache)
{
    const uint32_t descriptor_size = align_record(
        sizeof(struct coral_operator_descriptor));
    const uint32_t required = descriptor_size +
        sizeof(struct coral_gptq_paged_request);
    if (parameters == NULL || input == NULL || output == NULL ||
        scratch == NULL || residency == NULL || cache == NULL ||
        scratch->device_address < EXTMEM_BASE ||
        scratch->device_address > UINT32_MAX || scratch->byte_size < required ||
        input->device_address > UINT32_MAX || input->byte_size > UINT32_MAX ||
        output->device_address > UINT32_MAX || output->byte_size > UINT32_MAX ||
        residency->device_address > UINT32_MAX ||
        residency->byte_size > UINT32_MAX || cache->device_address > UINT32_MAX ||
        cache->byte_size > UINT32_MAX ||
        parameters->output_features == 0) {
        return -1;
    }

    const uint32_t descriptor_address = (uint32_t)scratch->device_address;
    const uint32_t request_address = descriptor_address + descriptor_size;
    volatile struct coral_operator_descriptor *descriptor =
        (volatile struct coral_operator_descriptor *)(uintptr_t)
            descriptor_address;
    volatile struct coral_gptq_paged_request *request =
        (volatile struct coral_gptq_paged_request *)(uintptr_t)request_address;
    volatile uint32_t *descriptor_words = (volatile uint32_t *)descriptor;
    volatile uint32_t *request_words = (volatile uint32_t *)request;
    for (uint32_t index = 0; index < sizeof(*descriptor) / sizeof(uint32_t);
         ++index) {
        descriptor_words[index] = 0;
    }
    for (uint32_t index = 0; index < sizeof(*request) / sizeof(uint32_t);
         ++index) {
        request_words[index] = 0;
    }

    request->magic = CORAL_GPTQ_PAGED_MAGIC;
    request->version = CORAL_GPTQ_PAGED_VERSION;
    request->struct_size = sizeof(*request);
    request->state = CORAL_GPTQ_PAGED_PENDING;
    request->command_id = command_id;
    request->role_id = role_id;
    request->rows = rows;
    request->expert_id = expert_id;
    volatile uint32_t *parameter_destination =
        (volatile uint32_t *)&request->parameters;
    const volatile uint32_t *parameter_source =
        (const volatile uint32_t *)parameters;
    for (uint32_t index = 0;
         index < sizeof(request->parameters) / sizeof(uint32_t); ++index) {
        parameter_destination[index] = parameter_source[index];
    }
    request->input_address = (uint32_t)input->device_address;
    request->input_size = (uint32_t)input->byte_size;
    request->output_address = (uint32_t)output->device_address;
    request->output_size = (uint32_t)output->byte_size;
    request->residency_address = (uint32_t)residency->device_address;
    request->residency_size = (uint32_t)residency->byte_size;
    request->cache_address = (uint32_t)cache->device_address;
    request->cache_size = (uint32_t)cache->byte_size;
    request->output_tile_columns = 64;

    descriptor->magic = CORAL_OPERATOR_ABI_MAGIC;
    descriptor->version = CORAL_OPERATOR_ABI_VERSION;
    descriptor->descriptor_size = sizeof(*descriptor);
    descriptor->opcode = CORAL_OPERATOR_OP_GPTQ_PAGED_MATMUL;
    descriptor->state = CORAL_OPERATOR_STATE_SUBMITTED;
    descriptor->flags = CORAL_OPERATOR_FLAG_CUSTOM_INSTRUCTION;
    descriptor->execution_mode = CORAL_OPERATOR_MODE_HYBRID;
    descriptor->tensor_count = 1;
    descriptor->tensors[0].address = request_address;
    descriptor->tensors[0].size = sizeof(*request);
    descriptor->tensors[0].rank = 1;
    descriptor->tensors[0].dimensions[0] = sizeof(*request);
    descriptor->tensors[0].element_type = CORAL_OPERATOR_ELEMENT_INT8;
    memory_fence();
    __asm__ volatile(".insn r 0x0b, 0, 0, x0, %0, %1"
                     :
                     : "r"(OPERATOR_BASE), "r"(descriptor_address)
                     : "memory");
    volatile uint32_t *status = (volatile uint32_t *)OPERATOR_STATUS;
    while (*status == CORAL_OPERATOR_STATE_RUNNING) {
        __asm__ volatile("nop");
    }
    memory_fence();
    return *status == CORAL_OPERATOR_STATE_COMPLETE &&
        descriptor->state == CORAL_OPERATOR_STATE_COMPLETE &&
        descriptor->error == CORAL_OPERATOR_ERROR_NONE &&
        request->state == CORAL_GPTQ_PAGED_COMPLETE &&
        request->error == CORAL_OPERATOR_ERROR_NONE ? 0 : -1;
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
    page_completion = completion;

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
                     sizeof(struct opennpux_npu_command), header->total_size) ||
        header->parameter_offset > header->total_size ||
        header->parameter_size >
            header->total_size - header->parameter_offset) {
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
    volatile struct opennpux_npu_tensor_binding *route_binding = NULL;
    volatile struct opennpux_npu_tensor_binding *residency_binding = NULL;
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
        if ((bindings[index].flags & OPENNPUX_NPU_BIND_ROUTE_TABLE) != 0) {
            if (route_binding != NULL) {
                finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                       ERROR_ROUTE, 0);
                return 1;
            }
            route_binding = &bindings[index];
        }
        if ((bindings[index].flags & OPENNPUX_NPU_BIND_PAGE_RESIDENCY) != 0) {
            if (residency_binding != NULL) {
                finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                       ERROR_PAGING, 0);
                return 1;
            }
            residency_binding = &bindings[index];
        }
    }
    if ((queue_binding == NULL) != (cache_binding == NULL) ||
        (queue_binding == NULL) != (residency_binding == NULL)) {
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
    volatile const struct opennpux_npu_weight_residency_header *residency = NULL;
    if (residency_binding != NULL) {
        if (residency_binding->device_address < EXTMEM_BASE ||
            residency_binding->byte_size < sizeof(*residency) ||
            residency_binding->device_address > EXTMEM_BASE + COMMAND_BUFFER_SIZE ||
            residency_binding->byte_size > EXTMEM_BASE + COMMAND_BUFFER_SIZE -
                residency_binding->device_address) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_PAGING, 0);
            return 1;
        }
        residency =
            (volatile const struct opennpux_npu_weight_residency_header *)(
                uintptr_t)residency_binding->device_address;
    }
    volatile const struct opennpux_npu_route_table_header *route = NULL;
    if (route_binding != NULL) {
        if (route_binding->device_address < EXTMEM_BASE ||
            route_binding->byte_size < sizeof(*route) ||
            route_binding->device_address > EXTMEM_BASE + COMMAND_BUFFER_SIZE ||
            route_binding->byte_size > EXTMEM_BASE + COMMAND_BUFFER_SIZE -
                route_binding->device_address) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_ROUTE, 0);
            return 1;
        }
        route = (volatile const struct opennpux_npu_route_table_header *)(
            uintptr_t)route_binding->device_address;
        if (route->magic != OPENNPUX_NPU_ROUTE_TABLE_MAGIC ||
            route->version != OPENNPUX_NPU_ROUTE_TABLE_VERSION ||
            route->header_size != sizeof(*route) ||
            route->record_size != sizeof(struct opennpux_npu_route_record) ||
            route->record_count == 0 ||
            route->record_count > OPENNPUX_NPU_MAX_ACTIVE_EXPERTS ||
            route->total_size != sizeof(*route) + route->record_count *
                sizeof(struct opennpux_npu_route_record) ||
            route->total_size > route_binding->byte_size ||
            route->checksum != route_checksum(
                (volatile const uint8_t *)(const volatile void *)route,
                route->total_size)) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_ROUTE, 0);
            return 1;
        }
    }
    volatile const struct opennpux_npu_inference_io *inference_input = NULL;
    volatile struct opennpux_npu_inference_io *inference_output = NULL;
    if ((header->flags & OPENNPUX_NPU_INVOKE_INFERENCE_IO) != 0) {
        if (header->binding_count < 2 ||
            bindings[0].device_address < EXTMEM_BASE ||
            bindings[0].device_address >
                EXTMEM_BASE + COMMAND_BUFFER_SIZE -
                    sizeof(struct opennpux_npu_inference_io) ||
            bindings[0].byte_size <
                sizeof(struct opennpux_npu_inference_io) ||
            bindings[1].device_address < EXTMEM_BASE ||
            bindings[1].device_address >
                EXTMEM_BASE + COMMAND_BUFFER_SIZE -
                    sizeof(struct opennpux_npu_inference_io) ||
            bindings[1].byte_size <
                sizeof(struct opennpux_npu_inference_io)) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_BOUNDS, 0);
            return 1;
        }
        inference_input =
            (volatile const struct opennpux_npu_inference_io *)(uintptr_t)
                bindings[0].device_address;
        inference_output =
            (volatile struct opennpux_npu_inference_io *)(uintptr_t)
                bindings[1].device_address;
        if (inference_input->magic != OPENNPUX_NPU_INFERENCE_IO_MAGIC ||
            inference_input->version != OPENNPUX_NPU_INFERENCE_IO_VERSION ||
            inference_input->struct_size != sizeof(*inference_input) ||
            inference_input->state != OPENNPUX_NPU_INFERENCE_PENDING ||
            inference_input->prompt_size == 0 ||
            inference_input->prompt_size >=
                OPENNPUX_NPU_INFERENCE_PROMPT_BYTES ||
            inference_input->vocabulary_size == 0 ||
            inference_input->max_new_tokens == 0 ||
            inference_input->max_new_tokens > 32 ||
            inference_input->input_token_count == 0 ||
            inference_input->prompt_checksum != byte_checksum(
                (volatile const uint8_t *)inference_input->prompt,
                inference_input->prompt_size)) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_ABI, 0);
            return 1;
        }
        inference_output->magic = OPENNPUX_NPU_INFERENCE_IO_MAGIC;
        inference_output->version = OPENNPUX_NPU_INFERENCE_IO_VERSION;
        inference_output->struct_size = sizeof(*inference_output);
        inference_output->state = OPENNPUX_NPU_INFERENCE_RUNNING;
        inference_output->mode = inference_input->mode;
        inference_output->prompt_checksum = inference_input->prompt_checksum;
        inference_output->vocabulary_size = inference_input->vocabulary_size;
        inference_output->max_new_tokens = inference_input->max_new_tokens;
        inference_output->input_token_count =
            inference_input->input_token_count;
    }
    uint64_t cycles = 0;
    uint32_t relocated = 0;
    uint32_t parameter_checksum = UINT32_C(2166136261);
    const uint32_t execution_steps = inference_input == NULL ? 1 :
        inference_input->max_new_tokens;
    if (header->command_count > UINT32_MAX / execution_steps) {
        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR, ERROR_BOUNDS, 0);
        return 1;
    }
    const uint32_t executed_commands =
        header->command_count * execution_steps;
    trace->weight_checksum = UINT32_C(2166136261);
    for (uint32_t step = 0; step < execution_steps; ++step) {
      uint32_t retired_token = 0;
      for (uint32_t index = 0; index < header->command_count; ++index) {
        if (commands[index].opcode == 0 ||
            commands[index].opcode > OPENNPUX_NPU_TRACE_MAX_OPCODE ||
            commands[index].capability_id == 0 ||
            commands[index].first_binding > header->binding_count ||
            commands[index].binding_count >
                header->binding_count - commands[index].first_binding ||
            commands[index].parameter_size !=
                sizeof(struct opennpux_npu_operator_parameters) ||
            commands[index].parameter_offset > header->parameter_size ||
            commands[index].parameter_size >
                header->parameter_size - commands[index].parameter_offset) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_COMMAND, index);
            return 1;
        }
        volatile const struct opennpux_npu_operator_parameters *parameters =
            (volatile const struct opennpux_npu_operator_parameters *)(
                base + header->parameter_offset +
                commands[index].parameter_offset);
        if (parameters->magic != OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC ||
            parameters->version !=
                OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION ||
            parameters->struct_size != sizeof(*parameters) ||
            parameters->opcode != commands[index].opcode) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_COMMAND, index);
            return 1;
        }
        const uint32_t batch_size =
            commands[index].runtime_shape & OPENNPUX_NPU_RUNTIME_FIELD_MASK;
        const uint32_t invocation_sequence_length =
            (commands[index].runtime_shape >>
             OPENNPUX_NPU_RUNTIME_SEQUENCE_SHIFT) &
            OPENNPUX_NPU_RUNTIME_FIELD_MASK;
        const uint32_t sequence_length = step == 0 ?
            invocation_sequence_length : 1;
        const uint32_t active_expert_count =
            (commands[index].runtime_shape >>
             OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT) &
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
            invocation_sequence_length == 0 ||
            weight_binding >= header->binding_count ||
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
        if (route != NULL && active_expert_count != route->record_count) {
            finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                   ERROR_ROUTE, index);
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
        parameter_checksum ^=
            (uint32_t)(commands[index].parameter_symbol >> 32);
        parameter_checksum *= UINT32_C(16777619);
        volatile const uint32_t *parameter_words =
            (volatile const uint32_t *)(const volatile void *)parameters;
        for (uint32_t word = 0;
             word < sizeof(*parameters) / sizeof(*parameter_words); ++word) {
            parameter_checksum ^= parameter_words[word];
            parameter_checksum *= UINT32_C(16777619);
        }
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
                uint32_t last = 0;
                uint32_t pages = 0;
                uint32_t role_id = 0;
                uint32_t component_id = 0;
                uint64_t expert_id = EXPERT_NONE;
                volatile struct opennpux_npu_weight_queue_header *queue =
                    (volatile struct opennpux_npu_weight_queue_header *)(uintptr_t)
                        queue_binding->device_address;
                volatile uint8_t *cache =
                    (volatile uint8_t *)(uintptr_t)cache_binding->device_address;
                const uint32_t cache_slots =
                    (uint32_t)(cache_binding->byte_size / PAGING_TRANSFER_SIZE);
                do {
                    if (page_weight(queue, cache, cache_slots, residency,
                                    commands[index].command_id, &word,
                                    &completion->stall_cycles, &last,
                                    &role_id, &component_id, &expert_id) != 0) {
                        finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                               ERROR_PAGING, index);
                        return 1;
                    }
                    page_progress(PAGE_PROGRESS_CALL_RETURNED);
                    trace->weight_checksum ^= word;
                    trace->weight_checksum *= UINT32_C(16777619);
                    trace->weight_checksum ^= role_id;
                    trace->weight_checksum *= UINT32_C(16777619);
                    trace->weight_checksum ^= component_id;
                    trace->weight_checksum *= UINT32_C(16777619);
                    ++trace->weight_page_requests;
                    trace->weight_dma_bytes += PAGING_TRANSFER_SIZE;
                    record->weight_dma_bytes += PAGING_TRANSFER_SIZE;
                    ++pages;
                } while (!last && pages < UINT32_C(1048576));
                if (!last) {
                    finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                           ERROR_PAGING, index);
                    return 1;
                }
                if ((header->flags & OPENNPUX_NPU_INVOKE_NUMERICAL) != 0 &&
                    commands[index].opcode == OPENNPUX_NPU_OP_MATMUL &&
                    execute_paged_matmul(
                        commands[index].command_id, parameters,
                        batch_size * sequence_length, role_id, expert_id,
                        &bindings[0], &bindings[1],
                        &bindings[scratch_binding], residency_binding,
                        cache_binding) != 0) {
                    finish(completion, OPENNPUX_NPU_COMPLETION_ERROR,
                           ERROR_COMMAND, index);
                    return 1;
                }
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
        completion->completed_commands =
            step * header->command_count + index + 1;
      }
    }
    trace->command_count = executed_commands;
    completion->npu_cycles = cycles;
    completion->dma_bytes_read = trace->weight_dma_bytes;
    completion->reserved[0] = relocated;
    completion->reserved[1] = parameter_checksum;
    if (route != NULL) {
        completion->reserved0 = route->record_count;
        completion->completion_fence = route->checksum;
    }
    if (inference_output != NULL) {
        const uint32_t result_checksum = inference_input->prompt_checksum ^
            trace->weight_checksum ^ parameter_checksum ^
            completion->completed_commands;
        inference_output->completed_commands =
            completion->completed_commands;
        inference_output->modeled_cycles = cycles;
        inference_output->result_checksum = result_checksum;
        inference_output->next_token =
            result_checksum % inference_output->vocabulary_size;
        inference_output->error = 0;
        memory_fence();
        inference_output->state = OPENNPUX_NPU_INFERENCE_COMPLETE;
    }
    finish(completion, OPENNPUX_NPU_COMPLETION_SUCCESS, 0, UINT32_MAX);
    return 0;
}
