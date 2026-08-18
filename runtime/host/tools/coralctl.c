#include "opennpux/coral_runtime.h"
#include "opennpux/model_package.h"
#include "opennpux/npu_executable.h"
#include "opennpux/npu_paging_layout.h"
#include "opennpux/npu_router.h"
#include "opennpux/npu_weight_queue.h"
#include "opennpux/npu_weight_residency.h"
#include "opennpux/qwen_model.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_EXTMEM_BASE UINT64_C(0x20000000)
#define NPU_WEIGHT_PAGE_SIZE UINT32_C(4096)
#define NPU_PAGING_WINDOW_SIZE UINT32_C(0x00800000)
#define NPU_EXECUTABLE_WINDOW_SIZE NPU_PAGING_WINDOW_SIZE
#define NPU_IO_BUFFER_SIZE UINT32_C(128)
#define NPU_STATE_BUFFER_SIZE UINT32_C(256)
#define NPU_SCRATCH_BUFFER_SIZE UINT32_C(256)

static int copy_to_shared_window(
    struct opennpux_coral_shared_window *window, const uint8_t *source,
    uint32_t size);

static void
copy_to_device_memory(volatile uint8_t *destination, const uint8_t *source,
                      size_t size)
{
    for (size_t index = 0; index < size; ++index) {
        destination[index] = source[index];
    }
}

static void
clear_device_memory(volatile uint8_t *destination, size_t size)
{
    for (size_t index = 0; index < size; ++index) {
        destination[index] = 0;
    }
}

static size_t
align_npu_record(size_t value)
{
    return (value + OPENNPUX_NPU_RECORD_ALIGNMENT - 1) &
        ~(size_t)(OPENNPUX_NPU_RECORD_ALIGNMENT - 1);
}

static void
print_paged_stage(int paged, const char *stage)
{
    if (paged) {
        fprintf(stderr, "paged_stage=%s\n", stage);
        fflush(stderr);
    }
}

static int
load_weight_page(const char *path, uint8_t *page, uint32_t page_size)
{
    memset(page, 0, page_size);
    if (path == NULL) {
        for (uint32_t index = 0; index < page_size; ++index) {
            page[index] = (uint8_t)(index * 37u + 11u);
        }
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    const size_t bytes = fread(page, 1, page_size, file);
    const int failed = ferror(file) || bytes == 0;
    fclose(file);
    if (failed) {
        errno = EIO;
        return -1;
    }
    return 0;
}

static int
load_router_logits(const char *path, uint32_t expert_count, float *logits)
{
    if (path == NULL || expert_count == 0 || logits == NULL) {
        errno = EINVAL;
        return -1;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    const size_t count = fread(logits, sizeof(*logits), expert_count, file);
    const int trailing = fgetc(file);
    const int failed = ferror(file) || count != expert_count || trailing != EOF;
    fclose(file);
    if (failed) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}

struct executable_page_service {
    struct opennpux_npu_weight_queue *queue;
    uint8_t *cache;
    const uint8_t *source;
    const char *manifest_path;
    const struct opennpux_model_package_info *model;
    const struct opennpux_npu_weight_ranges *ranges;
    struct opennpux_npu_weight_cache *weight_cache;
    const uint64_t *active_experts;
    uint32_t active_expert_count;
    struct opennpux_npu_weight_page_cursor *cursors;
    uint8_t *cursor_initialized;
    uint32_t command_count;
    void *residency;
    volatile uint8_t *residency_device;
    size_t residency_size;
    uint32_t cache_slots;
    uint32_t transfer_size;
    uint64_t faults_serviced;
    uint64_t transfer_bytes;
};

static int
service_executable_page(void *opaque)
{
    struct executable_page_service *service = opaque;
    struct opennpux_npu_weight_queue_header *header = service->queue->header;
    const uint32_t consumer = __atomic_load_n(
        &header->service_index, __ATOMIC_ACQUIRE);
    const uint32_t producer = __atomic_load_n(
        &header->producer_index, __ATOMIC_ACQUIRE);
    if (consumer == producer) {
        return 0;
    }
    if (service->faults_serviced == 0) {
        fprintf(stderr, "paged_stage=service-first-page-fault\n");
        fflush(stderr);
    }
    struct opennpux_npu_page_fault *fault =
        &service->queue->entries[consumer % header->capacity];
    if (__atomic_load_n(&fault->state, __ATOMIC_ACQUIRE) !=
            OPENNPUX_NPU_PAGE_FAULT_PENDING ||
        fault->magic != OPENNPUX_NPU_PAGE_FAULT_MAGIC ||
        fault->version != OPENNPUX_NPU_PAGE_FAULT_VERSION ||
        fault->struct_size != sizeof(*fault) ||
        fault->page_size != service->transfer_size) {
        errno = EPROTO;
        return -1;
    }
    uint32_t slot = fault->command_id % service->cache_slots;
    if (service->faults_serviced == 0) {
        fprintf(stderr,
                "paged_stage=service-fault-valid command=%" PRIu32
                " slot=%" PRIu32 " transfer=%" PRIu32 "\n",
                fault->command_id, slot, service->transfer_size);
        fflush(stderr);
    }
    if (service->ranges != NULL) {
        struct opennpux_npu_weight_page_request request;
        const void *page = NULL;
        uint32_t cache_hit = 0;
        if (fault->command_id >= service->command_count ||
            service->cursors == NULL || service->cursor_initialized == NULL) {
            errno = EPROTO;
            return -1;
        }
        struct opennpux_npu_weight_page_cursor *cursor =
            &service->cursors[fault->command_id];
        if (!service->cursor_initialized[fault->command_id] &&
            opennpux_npu_weight_page_cursor_begin_sized(
                service->ranges, fault->command_id,
                service->active_experts, service->active_expert_count,
                service->transfer_size, cursor) != 0) {
            return -1;
        }
        service->cursor_initialized[fault->command_id] = 1;
        if (opennpux_npu_weight_page_cursor_next(cursor, &request) != 1 ||
            opennpux_npu_weight_cache_acquire(
                service->weight_cache, service->manifest_path,
                service->model, &request, &page, &slot, &cache_hit) != 0) {
            errno = EIO;
            return -1;
        }
        fault->shard_index = request.shard_index;
        fault->file_offset = request.file_offset;
        fault->expert_id = request.expert_id;
        fault->role_id = request.role_id;
        fault->component_id = request.component_id;
        fault->range_file_offset = request.range_file_offset;
        fault->range_size = request.range_size;
        struct opennpux_npu_weight_page_cursor probe = *cursor;
        struct opennpux_npu_weight_page_request next_request;
        const int has_next =
            opennpux_npu_weight_page_cursor_next(&probe, &next_request);
        if (has_next < 0) {
            return -1;
        }
        fault->flags = has_next == 0 ? OPENNPUX_NPU_PAGE_FAULT_LAST : 0;
        copy_to_device_memory(
            (volatile uint8_t *)service->cache +
                (size_t)slot * service->transfer_size,
            page, service->transfer_size);
        (void)cache_hit;
    } else {
        if (service->faults_serviced == 0) {
            fprintf(stderr, "paged_stage=service-cache-copy\n");
            fflush(stderr);
        }
        copy_to_device_memory(
            (volatile uint8_t *)service->cache +
                (size_t)slot * service->transfer_size,
            service->source, service->transfer_size);
        fault->flags = OPENNPUX_NPU_PAGE_FAULT_LAST;
    }
    if (service->faults_serviced == 0) {
        fprintf(stderr, "paged_stage=service-cache-copy-complete\n");
        fflush(stderr);
    }
    fault->cache_slot = slot;
    fault->error_code = 0;
    if (service->faults_serviced == 0) {
        fprintf(stderr, "paged_stage=service-residency-publish\n");
        fflush(stderr);
    }
    if (service->residency != NULL &&
        opennpux_npu_weight_residency_publish(
            service->residency, service->residency_size, fault) != 0) {
        errno = EPROTO;
        return -1;
    }
    if (service->residency != NULL) {
        const size_t header_size =
            sizeof(struct opennpux_npu_weight_residency_header);
        const size_t record_size =
            sizeof(struct opennpux_npu_weight_residency_record);
        const size_t record_offset = header_size +
            (size_t)fault->cache_slot * record_size;
        copy_to_device_memory(
            service->residency_device + record_offset,
            (const uint8_t *)service->residency + record_offset,
            record_size);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        copy_to_device_memory(
            service->residency_device,
            (const uint8_t *)service->residency, header_size);
    }
    if (service->faults_serviced == 0) {
        fprintf(stderr, "paged_stage=service-residency-complete\n");
        fflush(stderr);
    }
    __atomic_store_n(&fault->state, OPENNPUX_NPU_PAGE_FAULT_READY,
                     __ATOMIC_RELEASE);
    __atomic_store_n(&header->service_index, consumer + 1,
                     __ATOMIC_RELEASE);
    ++service->faults_serviced;
    service->transfer_bytes += service->transfer_size;
    if (service->faults_serviced == 1) {
        fprintf(stderr, "paged_stage=service-first-page-complete\n");
        fflush(stderr);
    }
    return 0;
}

static const char *
npu_opcode_name(uint32_t opcode)
{
    static const char *const names[] = {
        "INVALID", "EMBED", "MATMUL", "ADD", "MUL", "NORMALIZE",
        "ROPE", "SOFTMAX", "TOPK", "CONVOLUTION", "CAUSAL_CONVOLUTION",
        "RECURRENT_UPDATE", "ROUTER", "EXPERT", "DMA", "ATTENTION",
        "ACTIVATION", "COMBINE",
    };
    return opcode < sizeof(names) / sizeof(names[0]) ? names[opcode] :
                                                       "UNKNOWN";
}

static void
usage(const char *prog)
{
    fprintf(stderr,
            "usage:\n"
            "  %s info [base]\n"
            "  %s run [base [entry [poll-count]]]\n"
            "  %s dma-test [base [poll-count]]\n"
            "  %s vector-add <elements> [base [poll-count]]\n"
            "  %s vector-add-custom <elements> [base [poll-count]]\n"
            "  %s model-run <model.npxm> [base [poll-count]]\n"
            "  %s model-info-v2 <model.npxm>\n"
            "  %s executable-run <model.npxc> [prefill|decode "
            "[weight-page [base [poll-count]]]]\n"
            "  %s executable-run-paged <model.npxc> [prefill|decode "
            "[weight-page [base [poll-count]]]]\n"
            "  %s executable-run-paged <model.npxc> [prefill|decode] "
            "<model.npxm> <model.npxr> [base [poll-count]]\n"
            "  %s qwen-info <qwen-tiny.npxm>\n"
            "  %s qwen-run <qwen-tiny.npxm> [golden-package|hybrid-sim]\n"
            "  %s qwen-stage-tcb <qwen-tiny.npxm> [base]\n"
            "  %s qwen-run-tcb <qwen-tiny.npxm> [base [poll-count]]\n"
            "  %s qwen-device-run <qwen-tiny.npxm> <prompt> "
            "[base [poll-count]]\n"
            "  %s mobilenet-test [base [poll-count]]\n"
            "  %s mem-info [base]\n"
            "  %s mem-clear [base]\n"
            "  %s mem-load <file> [base]\n"
            "  %s mem-read32 <offset> [base]\n"
            "  %s mem-write32 <offset> <value> [base]\n"
            "features: qwen-run-tcb-v2 qwen-device-run-v1\n",
            prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog,
            prog, prog, prog, prog, prog, prog, prog, prog, prog, prog);
}

static int
print_mem_load(struct opennpux_coral_device *dev, const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        perror("mem-load open");
        return 1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        perror("mem-load seek");
        return 1;
    }
    const long file_size = ftell(file);
    if (file_size <= 0 || (unsigned long)file_size > UINT32_MAX ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        errno = EFBIG;
        perror("mem-load size");
        return 1;
    }
    uint8_t *payload = malloc((size_t)file_size);
    if (payload == NULL) {
        fclose(file);
        return 1;
    }
    const size_t bytes = fread(payload, 1, (size_t)file_size, file);
    const int read_failed = bytes != (size_t)file_size || ferror(file);
    fclose(file);
    if (read_failed) {
        free(payload);
        errno = EIO;
        perror("mem-load read");
        return 1;
    }
    struct opennpux_coral_shared_window window = {0};
    int rc = opennpux_coral_open_shared_window(
        dev, (uint32_t)file_size, &window);
    if (rc == 0) {
        rc = copy_to_shared_window(&window, payload, (uint32_t)file_size);
    }
    if (rc == 0) {
        printf("shared_base=0x%08" PRIx32 "\n", window.base);
        printf("shared_size=0x%08" PRIx32 "\n", window.size);
        printf("mem_load_bytes=%ld\n", file_size);
        printf("mem_load=PASS\n");
    } else {
        perror("mem-load");
    }
    if (window.bytes != NULL) {
        opennpux_coral_close_shared_window(&window);
    }
    free(payload);
    return rc == 0 ? 0 : 1;
}

static int
print_executable_run(struct opennpux_coral_device *dev, const char *path,
                     const char *weight_page_path, const char *manifest_path,
                     const char *range_path, uint32_t entry_point,
                     uint32_t firmware_entry, int paged,
                     uint64_t polls)
{
    print_paged_stage(paged, "load-executable");
    struct opennpux_npu_executable executable;
    if (opennpux_npu_executable_load(path, &executable) != 0) {
        if (errno == EPROTONOSUPPORT) {
            fprintf(stderr,
                    "executable-run: stale executable ABI; regenerate model.npxc "
                    "with the current compile_npu_executable.py\n");
        }
        perror("executable-run load");
        return 1;
    }
    print_paged_stage(paged, "open-shared-window");
    struct opennpux_coral_shared_window window;
    const uint32_t window_size = paged ? NPU_PAGING_WINDOW_SIZE :
                                         NPU_EXECUTABLE_WINDOW_SIZE;
    if (opennpux_coral_open_shared_window(dev, window_size, &window) != 0) {
        opennpux_npu_executable_unload(&executable);
        return 1;
    }
    if (paged) {
        fprintf(stderr, "paged_shared_base=0x%08" PRIx32
                " paged_shared_size=0x%08" PRIx32 "\n",
                window.base, window.size);
        fflush(stderr);
    }
    struct opennpux_npu_tensor_binding bindings[9];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t index = 0; index < 5; ++index) {
        bindings[index].tensor_id = index;
        bindings[index].flags = index == 1 ? OPENNPUX_NPU_BIND_WRITE :
                                             OPENNPUX_NPU_BIND_READ;
        bindings[index].data_type = OPENNPUX_NPU_DTYPE_BFLOAT16;
        bindings[index].rank = 1;
        bindings[index].byte_size = 64;
        bindings[index].dimensions[0] = 32;
        bindings[index].memory_object = index + 1;
    }
    bindings[2].flags |= OPENNPUX_NPU_BIND_WEIGHT;
    bindings[3].flags |= OPENNPUX_NPU_BIND_PERSISTENT |
                         OPENNPUX_NPU_BIND_WRITE;
    bindings[4].flags |= OPENNPUX_NPU_BIND_WRITE;
    bindings[5].tensor_id = 5;
    bindings[5].flags = OPENNPUX_NPU_BIND_READ |
                        OPENNPUX_NPU_BIND_ROUTE_TABLE;
    bindings[5].data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
    bindings[5].rank = 1;
    bindings[5].byte_size = sizeof(struct opennpux_npu_route_table_header) +
        8 * sizeof(struct opennpux_npu_route_record);
    bindings[5].dimensions[0] = 8;
    bindings[5].memory_object = 6;

    if (paged) {
        bindings[6].tensor_id = 6;
        bindings[6].flags = OPENNPUX_NPU_BIND_READ |
            OPENNPUX_NPU_BIND_WRITE | OPENNPUX_NPU_BIND_PAGE_RESIDENCY;
        bindings[6].data_type = OPENNPUX_NPU_DTYPE_INT8;
        bindings[6].rank = 1;
        bindings[6].byte_size = 1;
        bindings[6].dimensions[0] = 1;
        bindings[6].memory_object = 7;
        bindings[7].tensor_id = 7;
        bindings[7].flags = OPENNPUX_NPU_BIND_READ |
            OPENNPUX_NPU_BIND_WRITE | OPENNPUX_NPU_BIND_PAGE_QUEUE;
        bindings[7].data_type = OPENNPUX_NPU_DTYPE_INT8;
        bindings[7].rank = 1;
        bindings[7].byte_size = 1;
        bindings[7].dimensions[0] = 1;
        bindings[7].memory_object = 8;
        bindings[8].tensor_id = 8;
        bindings[8].flags = OPENNPUX_NPU_BIND_READ |
            OPENNPUX_NPU_BIND_WRITE | OPENNPUX_NPU_BIND_WEIGHT |
            OPENNPUX_NPU_BIND_PAGE_CACHE;
        bindings[8].data_type = OPENNPUX_NPU_DTYPE_INT8;
        bindings[8].rank = 1;
        bindings[8].byte_size = 1;
        bindings[8].dimensions[0] = 1;
        bindings[8].memory_object = 9;
    }

    struct opennpux_npu_paging_layout paging_layout;
    memset(&paging_layout, 0, sizeof(paging_layout));

    const int real_weights = manifest_path != NULL;
    struct opennpux_model_package_info weight_model;
    struct opennpux_npu_weight_ranges weight_ranges;
    struct opennpux_npu_weight_cache weight_cache;
    struct opennpux_npu_weight_cache_entry *weight_cache_entries = NULL;
    uint8_t *weight_cache_storage = NULL;
    struct opennpux_npu_weight_page_cursor *page_cursors = NULL;
    uint8_t *page_cursor_initialized = NULL;
    uint8_t *residency_image = NULL;
    uint8_t *queue_image = NULL;
    uint8_t *route_image = NULL;
    uint64_t active_experts[8];
    opennpux_npu_route active_routes[8];
    uint32_t active_route_count = 0;
    memset(&weight_model, 0, sizeof(weight_model));
    memset(&weight_ranges, 0, sizeof(weight_ranges));
    memset(&weight_cache, 0, sizeof(weight_cache));
    memset(active_experts, 0, sizeof(active_experts));
    memset(active_routes, 0, sizeof(active_routes));
    active_route_count = executable.header->default_active_experts < 8 ?
        executable.header->default_active_experts : 8;
    if (active_route_count == 0) {
        active_route_count = 1;
    }
    for (uint32_t index = 0; index < active_route_count; ++index) {
        active_experts[index] = index;
        active_routes[index].expert_id = index;
        active_routes[index].weight = 1.0f / (float)active_route_count;
    }
    const uint32_t weight_page_size = paged ?
        OPENNPUX_NPU_PAGING_TRANSFER_DEFAULT : NPU_WEIGHT_PAGE_SIZE;
    uint8_t *weight_page = real_weights ? NULL : malloc(weight_page_size);
    if (!real_weights && (weight_page == NULL ||
        load_weight_page(weight_page_path, weight_page, weight_page_size) != 0)) {
        perror("executable-run weight page");
        free(weight_page);
        opennpux_coral_close_shared_window(&window);
        opennpux_npu_executable_unload(&executable);
        return 1;
    }

    uint8_t *submission = malloc(window.size);
    size_t submission_size = 0;
    int rc = 1;
    if (submission == NULL || opennpux_npu_executable_instantiate(
            &executable, entry_point, 1, 1, bindings, paged ? 9 : 6, submission,
            window.size, &submission_size) != 0) {
        perror("executable-run instantiate");
        goto out;
    }
    print_paged_stage(paged, "plan-layout");
    const size_t completion_offset =
        align_npu_record(submission_size);
    const size_t trace_offset = align_npu_record(
        completion_offset + sizeof(struct opennpux_npu_completion));
    const size_t trace_size = sizeof(struct opennpux_npu_trace_header) +
        OPENNPUX_NPU_TRACE_MAX_OPCODE *
            sizeof(struct opennpux_npu_trace_record);
    size_t data_offset = align_npu_record(trace_offset + trace_size);
    const size_t input_offset = data_offset;
    data_offset += NPU_IO_BUFFER_SIZE;
    const size_t output_offset = data_offset;
    data_offset += NPU_IO_BUFFER_SIZE;
    const size_t weight_offset = data_offset;
    data_offset += NPU_WEIGHT_PAGE_SIZE;
    const size_t state_offset = data_offset;
    data_offset += NPU_STATE_BUFFER_SIZE;
    const size_t scratch_offset = data_offset;
    data_offset += NPU_SCRATCH_BUFFER_SIZE;
    const size_t route_offset = align_npu_record(data_offset);
    const size_t route_capacity =
        sizeof(struct opennpux_npu_route_table_header) +
        8 * sizeof(struct opennpux_npu_route_record);
    const size_t residency_offset = align_npu_record(
        route_offset + route_capacity);
    const size_t residency_size = paged ?
        opennpux_npu_weight_residency_size(
            OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT) : 0;
    data_offset = residency_offset + residency_size;
    if (completion_offset > window.size ||
        sizeof(struct opennpux_npu_completion) > window.size - completion_offset ||
        data_offset > window.size) {
        errno = ENOSPC;
        perror("executable-run completion");
        goto out;
    }
    if (paged && (opennpux_npu_paging_layout_plan(
            data_offset, window.size,
            OPENNPUX_NPU_PAGING_QUEUE_CAPACITY_DEFAULT,
            OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT,
            OPENNPUX_NPU_PAGING_TRANSFER_DEFAULT, &paging_layout) != 0 ||
        paging_layout.queue_offset < data_offset)) {
        perror("executable-run paging layout");
        goto out;
    }
    if (paged) {
        fprintf(stderr, "paged_layout_control=0x%zx queue=0x%" PRIx64
                " cache=0x%" PRIx64 " required=0x%" PRIx64 "\n",
                data_offset, paging_layout.queue_offset,
                paging_layout.cache_offset, paging_layout.required_size);
        fflush(stderr);
    }
    struct opennpux_npu_invocation_header *header =
        (struct opennpux_npu_invocation_header *)(void *)submission;
    struct opennpux_npu_tensor_binding *submission_bindings =
        (struct opennpux_npu_tensor_binding *)(void *)(submission +
                                                       header->binding_offset);
    const size_t offsets[] = {
        input_offset, output_offset, weight_offset, state_offset, scratch_offset,
    };
    const uint64_t sizes[] = {
        NPU_IO_BUFFER_SIZE, NPU_IO_BUFFER_SIZE, NPU_WEIGHT_PAGE_SIZE,
        NPU_STATE_BUFFER_SIZE, NPU_SCRATCH_BUFFER_SIZE,
    };
    for (uint32_t index = 0; index < 5; ++index) {
        submission_bindings[index].device_address =
            NPU_EXTMEM_BASE + offsets[index];
        submission_bindings[index].byte_size = sizes[index];
    }
    submission_bindings[5].device_address = NPU_EXTMEM_BASE + route_offset;
    submission_bindings[5].byte_size = route_capacity;
    if (paged) {
        submission_bindings[6].device_address =
            NPU_EXTMEM_BASE + residency_offset;
        submission_bindings[6].byte_size = residency_size;
        submission_bindings[6].dimensions[0] = residency_size;
    }
    if (paged && opennpux_npu_paging_layout_bindings(
            &paging_layout, NPU_EXTMEM_BASE, 7, 8,
            &submission_bindings[7], &submission_bindings[8]) != 0) {
        perror("executable-run paging bindings");
        goto out;
    }
    header->completion_address = NPU_EXTMEM_BASE + completion_offset;
    for (size_t index = input_offset; index < data_offset; ++index) {
        window.bytes[index] = 0;
    }
    for (uint32_t index = 0; index < NPU_WEIGHT_PAGE_SIZE; ++index) {
        window.bytes[weight_offset + index] =
            weight_page == NULL ? 0 : weight_page[index];
    }
    struct opennpux_npu_weight_queue page_queue;
    memset(&page_queue, 0, sizeof(page_queue));
    struct executable_page_service page_service;
    memset(&page_service, 0, sizeof(page_service));
    if (paged) {
        residency_image = calloc(1, residency_size);
        queue_image = calloc(1, (size_t)paging_layout.queue_size);
        if (residency_image == NULL || queue_image == NULL) {
            perror("executable-run paging metadata allocation");
            goto out;
        }
        print_paged_stage(paged, "initialize-residency-and-queue");
        if (opennpux_npu_weight_residency_init(
                residency_image,
                residency_size,
                OPENNPUX_NPU_PAGING_CACHE_SLOTS_DEFAULT) != 0) {
            perror("executable-run residency table");
            goto out;
        }
        if (opennpux_npu_weight_queue_init(
                queue_image, (size_t)paging_layout.queue_size,
                paging_layout.queue_capacity, &page_queue) != 0) {
            perror("executable-run paging queue");
            goto out;
        }
        copy_to_device_memory(window.bytes + residency_offset,
                              residency_image, residency_size);
        copy_to_device_memory(window.bytes + paging_layout.queue_offset,
                              queue_image, (size_t)paging_layout.queue_size);
        if (opennpux_npu_weight_queue_attach(
                (void *)(uintptr_t)(window.bytes + paging_layout.queue_offset),
                (size_t)paging_layout.queue_size, &page_queue) != 0) {
            perror("executable-run paging queue attach");
            goto out;
        }
        page_service.queue = &page_queue;
        page_service.cache = (uint8_t *)(uintptr_t)window.bytes +
            paging_layout.cache_offset;
        page_service.source = weight_page;
        page_service.cache_slots = paging_layout.cache_slots;
        page_service.transfer_size = paging_layout.transfer_size;
        page_service.residency = residency_image;
        page_service.residency_device = window.bytes + residency_offset;
        page_service.residency_size = residency_size;
        if (real_weights) {
            weight_cache_entries = calloc(
                paging_layout.cache_slots, sizeof(*weight_cache_entries));
            weight_cache_storage = malloc((size_t)paging_layout.cache_size);
            page_cursors = calloc(executable.header->command_count,
                                  sizeof(*page_cursors));
            page_cursor_initialized = calloc(
                executable.header->command_count,
                sizeof(*page_cursor_initialized));
            if (weight_cache_entries == NULL || weight_cache_storage == NULL ||
                page_cursors == NULL ||
                page_cursor_initialized == NULL || range_path == NULL ||
                opennpux_model_package_load(
                    manifest_path, &weight_model) != 0 ||
                opennpux_npu_weight_ranges_load(
                    range_path, &weight_ranges) != 0 ||
                weight_ranges.header->executable_id !=
                    executable.header->executable_id ||
                opennpux_npu_weight_cache_init_sized(
                    &weight_cache, weight_cache_entries,
                    weight_cache_storage, paging_layout.cache_slots,
                    paging_layout.transfer_size) != 0) {
                perror("executable-run real weight pager");
                goto out;
            }
            const uint32_t active_count = weight_model.experts_per_token < 8 ?
                weight_model.experts_per_token : 8;
            const char *router_path = getenv("OPENNPUX_ROUTER_LOGITS");
            float *router_logits = NULL;
            if (router_path != NULL) {
                router_logits = malloc(
                    weight_model.expert_count * sizeof(*router_logits));
                if (router_logits == NULL ||
                    load_router_logits(router_path, weight_model.expert_count,
                                       router_logits) != 0 ||
                    opennpux_npu_router_topk(
                        router_logits, weight_model.expert_count,
                        active_count, active_routes) != 0) {
                    free(router_logits);
                    perror("executable-run router logits");
                    goto out;
                }
                active_route_count = active_count;
                for (uint32_t index = 0; index < active_count; ++index) {
                    active_experts[index] = active_routes[index].expert_id;
                }
                free(router_logits);
            } else {
                for (uint32_t index = 0; index < active_count; ++index) {
                    active_experts[index] = index;
                    active_routes[index].expert_id = index;
                    active_routes[index].weight =
                        1.0f / (float)active_count;
                }
                active_route_count = active_count;
            }
            page_service.manifest_path = manifest_path;
            page_service.model = &weight_model;
            page_service.ranges = &weight_ranges;
            page_service.weight_cache = &weight_cache;
            page_service.active_experts = active_experts;
            page_service.active_expert_count = active_count;
            page_service.cursors = page_cursors;
            page_service.cursor_initialized = page_cursor_initialized;
            page_service.command_count = executable.header->command_count;
        }
    }
    size_t route_size = 0;
    route_image = calloc(1, route_capacity);
    if (route_image == NULL) {
        perror("executable-run route table allocation");
        goto out;
    }
    print_paged_stage(paged, "build-route-table");
    if (opennpux_npu_route_table_build(
            active_routes, active_route_count,
            route_image, route_capacity,
            &route_size) != 0) {
        errno = EINVAL;
        perror("executable-run route table");
        goto out;
    }
    copy_to_device_memory(window.bytes + route_offset, route_image, route_size);
    submission_bindings[5].byte_size = route_size;
    submission_bindings[5].dimensions[0] = active_route_count;
    header->checksum = 0;
    header->checksum = opennpux_npu_submission_checksum(
        submission, submission_size);
    if (opennpux_npu_submission_validate(submission, submission_size) != 0 ||
        copy_to_shared_window(&window, submission,
                              (uint32_t)submission_size) != 0) {
        perror("executable-run stage");
        goto out;
    }
    print_paged_stage(paged, "submission-staged");
    volatile struct opennpux_npu_completion *completion =
        (volatile struct opennpux_npu_completion *)(volatile void *)(
            window.bytes + completion_offset);
    clear_device_memory((volatile uint8_t *)(volatile void *)completion,
                        sizeof(*completion));
    __sync_synchronize();
    uint32_t device_status = 0;
    print_paged_stage(paged, "device-run-and-page-service");
    const int run_result = paged ?
        opennpux_coral_run_with_service(
            dev, firmware_entry, polls, service_executable_page,
            &page_service, &device_status) :
        opennpux_coral_run(dev, firmware_entry, polls, &device_status);
    if (run_result != 0) {
        perror("executable-run device");
        goto out;
    }
    print_paged_stage(paged, "device-complete");
    __sync_synchronize();
    printf("transport=%s\n", opennpux_coral_transport_name(dev->transport));
    printf("executable_id=0x%016" PRIx64 "\n", header->executable_id);
    printf("entry_point=%" PRIu32 "\n", entry_point);
    printf("submission_bytes=%zu\n", submission_size);
    printf("completion_offset=0x%zx\n", completion_offset);
    printf("submitted_commands=%" PRIu32 "\n", header->command_count);
    printf("device_status=0x%08" PRIx32 "\n", device_status);
    printf("completion_state=%" PRIu32 "\n", completion->state);
    printf("completion_error=%" PRIu32 "\n", completion->error_code);
    printf("completed_commands=%" PRIu32 "\n", completion->completed_commands);
    printf("device_route_count=%" PRIu32 "\n", completion->reserved0);
    printf("device_route_checksum=0x%08" PRIx64 "\n",
           completion->completion_fence);
    const struct opennpux_npu_command *commands =
        (const struct opennpux_npu_command *)(const void *)(
            submission + header->command_offset);
    const uint64_t runtime_shape = commands[0].runtime_shape;
    const uint64_t resource_bindings = commands[0].resource_bindings;
    printf("runtime_batch=%" PRIu64 "\n",
           runtime_shape & OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("runtime_sequence=%" PRIu64 "\n",
           (runtime_shape >> OPENNPUX_NPU_RUNTIME_SEQUENCE_SHIFT) &
               OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("runtime_kv=%" PRIu64 "\n",
           (runtime_shape >> OPENNPUX_NPU_RUNTIME_KV_SHIFT) &
               OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("runtime_active_experts=%" PRIu64 "\n",
           (runtime_shape >> OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT) &
           OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    if (active_route_count != 0) {
        printf("runtime_expert_route_source=%s\n",
               getenv("OPENNPUX_ROUTER_LOGITS") == NULL ?
                   "deterministic-fallback" : "runtime-logits");
        for (uint32_t index = 0; index < active_route_count; ++index) {
            printf("runtime_expert_route_%" PRIu32 "=id:%" PRIu32
                   ",weight:%.9g\n", index,
                   active_routes[index].expert_id,
                   (double)active_routes[index].weight);
        }
    }
    printf("weight_binding=%" PRIu64 "\n",
           resource_bindings & OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("state_binding=%" PRIu64 "\n",
           (resource_bindings >> OPENNPUX_NPU_RESOURCE_STATE_SHIFT) &
               OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("scratch_binding=%" PRIu64 "\n",
           (resource_bindings >> OPENNPUX_NPU_RESOURCE_SCRATCH_SHIFT) &
               OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("weight_page_source=%s\n",
           real_weights ? manifest_path :
           (weight_page_path == NULL ? "deterministic" : weight_page_path));
    printf("weight_page_bytes=%" PRIu32 "\n", weight_page_size);
    if (paged) {
        printf("paging_transfer_size=%" PRIu32 "\n",
               paging_layout.transfer_size);
        printf("paging_faults_serviced=%" PRIu64 "\n",
               page_service.faults_serviced);
        printf("paging_transfer_bytes=%" PRIu64 "\n",
               page_service.transfer_bytes);
        printf("paging_queue_producer=%" PRIu32 "\n",
               page_queue.header->producer_index);
        printf("paging_queue_service=%" PRIu32 "\n",
               page_queue.header->service_index);
        printf("paging_queue_retire=%" PRIu32 "\n",
               page_queue.header->retire_index);
        printf("paging_queue_backpressure=%" PRIu32 "\n",
               page_queue.header->backpressure_count);
        printf("paging_source=%s\n",
               real_weights ? "model-ranges" : "repeated-page");
        if (real_weights) {
            printf("paging_cache_hits=%" PRIu64 "\n",
                   weight_cache.stats.hits);
            printf("paging_cache_misses=%" PRIu64 "\n",
                   weight_cache.stats.misses);
            printf("paging_cache_evictions=%" PRIu64 "\n",
                   weight_cache.stats.evictions);
            printf("paging_weight_bytes_read=%" PRIu64 "\n",
                   weight_cache.stats.bytes_read);
        }
        const struct opennpux_npu_weight_residency_header *residency =
            (const struct opennpux_npu_weight_residency_header *)(const void *)
                residency_image;
        printf("paging_residency_generation=%" PRIu64 "\n",
               residency->generation);
        printf("paging_residency_records=%" PRIu32 "\n",
               residency->valid_records);
    }
    printf("relocated_commands=%" PRIu64 "\n", completion->reserved[0]);
    printf("parameter_checksum=0x%08" PRIx64 "\n",
           completion->reserved[1]);
    const uint64_t response_trace_offset =
        completion->trace_address >= NPU_EXTMEM_BASE ?
        completion->trace_address - UINT64_C(0x20000000) : UINT64_MAX;
    const struct opennpux_npu_trace_header *trace = NULL;
    if (response_trace_offset <= window.size && completion->trace_size <=
            window.size - response_trace_offset && completion->trace_size >=
            sizeof(struct opennpux_npu_trace_header)) {
        trace = (const struct opennpux_npu_trace_header *)(const void *)(
            window.bytes + response_trace_offset);
    }
    if (trace != NULL &&
        (trace->magic != OPENNPUX_NPU_TRACE_MAGIC ||
         trace->version != OPENNPUX_NPU_TRACE_VERSION ||
         trace->struct_size != sizeof(*trace) ||
         trace->record_count > OPENNPUX_NPU_TRACE_MAX_OPCODE ||
         completion->trace_size < sizeof(*trace) +
                 trace->record_count * sizeof(struct opennpux_npu_trace_record))) {
        trace = NULL;
    }
    if (trace != NULL) {
        printf("dispatch_capability_mask=0x%016" PRIx64 "\n",
               trace->capability_mask);
        printf("dispatch_dependency_edges=%" PRIu32 "\n",
               trace->dependency_edges);
        printf("dispatch_estimated_operations=%" PRIu64 "\n",
               trace->estimated_operations);
        printf("dispatch_estimated_bytes=%" PRIu64 "\n",
               trace->estimated_bytes);
        printf("dispatch_weight_page_requests=%" PRIu32 "\n",
               trace->weight_page_requests);
        printf("dispatch_weight_dma_bytes=%" PRIu64 "\n",
               trace->weight_dma_bytes);
        printf("dispatch_weight_checksum=0x%08" PRIx32 "\n",
               trace->weight_checksum);
        printf("dispatch_modeled_cycles=%" PRIu64 "\n",
               completion->npu_cycles);
        const struct opennpux_npu_trace_record *records =
            (const struct opennpux_npu_trace_record *)(const void *)(trace + 1);
        for (uint32_t index = 0; index < trace->record_count; ++index) {
            if (records[index].command_count == 0) {
                continue;
            }
            printf("dispatch_op_%s=count:%" PRIu32 ",operations:%" PRIu64
                   ",bytes:%" PRIu64 ",weight_dma_bytes:%" PRIu64 "\n",
                   npu_opcode_name(records[index].opcode),
                   records[index].command_count,
                   records[index].estimated_operations,
                   records[index].estimated_bytes,
                   records[index].weight_dma_bytes);
        }
    }
    const struct opennpux_npu_route_table_header *route_table =
        (const struct opennpux_npu_route_table_header *)(const void *)(
            window.bytes + route_offset);
    if (completion->magic != OPENNPUX_NPU_COMPLETION_MAGIC ||
        completion->version != OPENNPUX_NPU_COMPLETION_VERSION ||
        completion->sequence != header->sequence ||
        completion->state != OPENNPUX_NPU_COMPLETION_SUCCESS ||
        completion->error_code != 0 ||
        completion->completed_commands != header->command_count ||
        completion->reserved[0] != header->command_count ||
        completion->reserved[1] == 0 ||
        completion->reserved0 != active_route_count ||
        completion->completion_fence != route_table->checksum ||
        trace == NULL ||
        trace->command_count != header->command_count) {
        errno = EIO;
        perror("executable-run completion validation");
        goto out;
    }
    if (paged &&
        (page_service.faults_serviced != trace->weight_page_requests ||
         page_queue.header->producer_index != trace->weight_page_requests ||
         page_queue.header->service_index != trace->weight_page_requests ||
         page_queue.header->retire_index != trace->weight_page_requests ||
         page_service.transfer_bytes != trace->weight_dma_bytes)) {
        errno = EIO;
        perror("executable-run paging validation");
        goto out;
    }
    puts("executable_run=PASS");
    rc = 0;
out:
    opennpux_npu_weight_ranges_unload(&weight_ranges);
    free(weight_cache_entries);
    free(weight_cache_storage);
    free(page_cursors);
    free(page_cursor_initialized);
    free(residency_image);
    free(queue_image);
    free(route_image);
    free(weight_page);
    free(submission);
    opennpux_coral_close_shared_window(&window);
    opennpux_npu_executable_unload(&executable);
    return rc;
}

static int
print_model_info_v2(const char *path)
{
    struct opennpux_model_package_info info;
    if (opennpux_model_package_load(path, &info) != 0) {
        perror("model-info-v2");
        return 1;
    }
    printf("model_format=%s\n", info.format);
    printf("model_name=%s\n", info.name);
    printf("model_architecture=%s\n", info.architecture_name);
    printf("model_dtype=%s\n", info.dtype);
    printf("model_quantization=%s\n", info.quantization_method);
    printf("model_quantization_bits=%" PRIu32 "\n",
           info.quantization_bits);
    printf("model_quantization_group_size=%" PRIu32 "\n",
           info.quantization_group_size);
    printf("model_layers=%" PRIu32 "\n", info.layer_count);
    printf("model_hidden=%" PRIu32 "\n", info.hidden_size);
    printf("model_intermediate=%" PRIu32 "\n", info.intermediate_size);
    printf("model_heads=%" PRIu32 "\n", info.head_count);
    printf("model_kv_heads=%" PRIu32 "\n", info.kv_head_count);
    printf("model_head_dim=%" PRIu32 "\n", info.head_dim);
    printf("model_max_sequence=%" PRIu32 "\n", info.max_sequence_length);
    printf("model_experts=%" PRIu32 "\n", info.expert_count);
    printf("model_experts_per_token=%" PRIu32 "\n",
           info.experts_per_token);
    printf("model_moe_intermediate=%" PRIu32 "\n",
           info.moe_intermediate_size);
    printf("model_shared_expert_intermediate=%" PRIu32 "\n",
           info.shared_expert_intermediate_size);
    printf("model_tensors=%" PRIu32 "\n", info.tensor_count);
    printf("model_shards=%" PRIu32 "\n", info.shard_count);
    printf("model_weight_bytes=%" PRIu64 "\n", info.total_weight_bytes);
    printf("model_required_op_mask=0x%08" PRIx32 "\n",
           info.required_op_mask);
    if (opennpux_model_package_validate_shards(path, &info) != 0) {
        perror("model-info-v2 shards");
        return 1;
    }
    puts("model_info_v2=PASS");
    return 0;
}

static int
print_qwen_info(const char *path)
{
    struct opennpux_qwen_model_info info;
    if (opennpux_qwen_load_model_info(path, &info) != 0) {
        perror("qwen-info");
        return 1;
    }
    printf("qwen_model=%s\n", info.name);
    printf("qwen_format=%s\n", info.format);
    printf("qwen_version=%" PRIu32 "\n", info.version);
    printf("qwen_layers=%" PRIu32 "\n", info.layer_count);
    printf("qwen_hidden=%" PRIu32 "\n", info.hidden_size);
    printf("qwen_intermediate=%" PRIu32 "\n", info.intermediate_size);
    printf("qwen_heads=%" PRIu32 "\n", info.head_count);
    printf("qwen_head_dim=%" PRIu32 "\n", info.head_dim);
    printf("qwen_vocab=%" PRIu32 "\n", info.vocab_size);
    printf("qwen_prompt_tokens=%" PRIu32 "\n", info.prompt_token_count);
    printf("qwen_operator_count=%" PRIu32 "\n", info.operator_count);
    printf("qwen_ops=%s\n", opennpux_qwen_required_ops_string());
    printf("qwen_op_mask=0x%08" PRIx32 "\n", info.op_mask);
    printf("qwen_next_token=%" PRIu32 "\n", info.next_token);
    printf("qwen_logits_checksum=0x%08" PRIx32 "\n", info.logits_checksum);
    printf("qwen_weight_checksum=0x%08" PRIx32 "\n", info.weight_checksum);
    printf("qwen_info=PASS\n");
    return 0;
}

static int
print_qwen_run(const char *path, const char *mode)
{
    struct opennpux_qwen_run_result result;
    const char *selected_mode = mode == NULL ? "hybrid-sim" : mode;
    int run_result = -1;
    if (strcmp(selected_mode, "golden-package") == 0) {
        run_result = opennpux_qwen_run_golden(path, &result);
    } else if (strcmp(selected_mode, "hybrid-sim") == 0) {
        run_result = opennpux_qwen_run_hybrid_sim(path, &result);
    } else {
        fprintf(stderr, "unsupported qwen mode: %s\n", selected_mode);
        errno = EINVAL;
        return 1;
    }
    if (run_result != 0) {
        perror("qwen-run");
        return 1;
    }

    printf("qwen_model=%s\n", result.info.name);
    printf("qwen_mode=%s\n", selected_mode);
    printf("qwen_prompt_checksum=0x%08" PRIx32 "\n",
           result.prompt_checksum);
    printf("qwen_prefill=%s\n", result.prefill_pass ? "PASS" : "FAIL");
    printf("qwen_decode=%s\n", result.decode_pass ? "PASS" : "FAIL");
    if (strcmp(selected_mode, "hybrid-sim") == 0) {
        printf("qwen_numeric_reference=PASS\n");
    }
    printf("qwen_completed_operators=%" PRIu32 "\n",
           result.completed_operators);
    printf("qwen_operator_summary=%s\n", opennpux_qwen_required_ops_string());
    for (uint32_t index = 0; index < OPENNPUX_QWEN_OP_KIND_COUNT; ++index) {
        printf("qwen_operator_count_%s=%" PRIu32 "\n",
               opennpux_qwen_op_name(index), result.op_counts[index]);
    }
    for (uint32_t index = 0; index < result.completed_operators &&
                           index < OPENNPUX_QWEN_MAX_OPS; ++index) {
        const struct opennpux_qwen_op_entry *entry = &result.ops[index];
        printf("qwen_op_%02" PRIu32 "=%s layer=%s", index,
               opennpux_qwen_op_name(entry->kind),
               entry->layer == UINT32_MAX ? "none" : "");
        if (entry->layer != UINT32_MAX) {
            printf("%" PRIu32, entry->layer);
        }
        printf(" shape=");
        if (entry->dim_count == 0) {
            printf("none");
        }
        for (uint32_t dim = 0; dim < entry->dim_count; ++dim) {
            printf("%s%" PRIu32, dim == 0 ? "" : "x", entry->dims[dim]);
        }
        printf(" ops=%" PRIu64 " bytes_r=%" PRIu64
               " bytes_w=%" PRIu64 " cycles=%" PRIu64 "\n",
               entry->operations, entry->bytes_read, entry->bytes_written,
               entry->modeled_cycles);
    }
    printf("qwen_operation_count=%" PRIu64 "\n", result.operation_count);
    printf("qwen_bytes_read=%" PRIu64 "\n", result.bytes_read);
    printf("qwen_bytes_written=%" PRIu64 "\n", result.bytes_written);
    printf("qwen_modeled_cycles=%" PRIu64 "\n", result.modeled_cycles);
    printf("qwen_tcb_size=%" PRIu32 "\n", result.tcb_size);
    printf("qwen_tcb_checksum=0x%08" PRIx32 "\n", result.tcb_checksum);
    printf("qwen_logits_checksum=0x%08" PRIx32 "\n",
           result.output_checksum);
    printf("qwen_next_token=%" PRIu32 "\n", result.next_token);
    printf("qwen_run=PASS\n");
    return 0;
}

static int
copy_to_shared_window(struct opennpux_coral_shared_window *window,
                      const uint8_t *source, uint32_t size)
{
    if (size > window->size) {
        errno = ERANGE;
        return -1;
    }
    copy_to_device_memory(window->bytes, source, size);
    // Publish shared-memory payload stores before the MMIO doorbell.
    __atomic_thread_fence(__ATOMIC_RELEASE);
    return 0;
}

static int
verify_staged_qwen_tcb(const struct opennpux_qwen_run_result *result,
                       const struct opennpux_coral_shared_window *window)
{
    const volatile struct opennpux_qwen_tcb_header *header =
        (const volatile struct opennpux_qwen_tcb_header *)(const volatile void *)
            window->bytes;
    if (header->magic != OPENNPUX_QWEN_TCB_MAGIC ||
        header->version != OPENNPUX_QWEN_TCB_VERSION ||
        header->total_size != result->tcb_size ||
        header->op_count != result->completed_operators ||
        header->tcb_checksum != result->tcb_checksum ||
        header->logits_checksum != result->output_checksum ||
        header->next_token != result->next_token) {
        errno = EIO;
        return -1;
    }
    return 0;
}

static int
print_qwen_stage_tcb(struct opennpux_coral_device *dev, const char *path)
{
    struct opennpux_qwen_run_result result;
    if (opennpux_qwen_run_hybrid_sim(path, &result) != 0) {
        perror("qwen-stage-tcb");
        return 1;
    }

    uint8_t tcb[OPENNPUX_QWEN_TCB_MAX_SIZE];
    uint32_t tcb_size = 0;
    uint32_t tcb_checksum = 0;
    if (opennpux_qwen_build_tcb(&result, tcb, sizeof(tcb),
                                &tcb_size, &tcb_checksum) != 0) {
        perror("qwen-stage-tcb");
        return 1;
    }

    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, tcb_size, &window) != 0) {
        return 1;
    }

    int rc = 0;
    if (copy_to_shared_window(&window, tcb, tcb_size) != 0 ||
        verify_staged_qwen_tcb(&result, &window) != 0) {
        perror("qwen-stage-tcb");
        rc = 1;
    }

    printf("transport=%s\n", opennpux_coral_transport_name(dev->transport));
    printf("shared_base=0x%08" PRIx32 "\n", window.base);
    printf("shared_size=0x%08" PRIx32 "\n", window.size);
    printf("qwen_model=%s\n", result.info.name);
    printf("qwen_completed_operators=%" PRIu32 "\n",
           result.completed_operators);
    printf("qwen_operation_count=%" PRIu64 "\n", result.operation_count);
    printf("qwen_modeled_cycles=%" PRIu64 "\n", result.modeled_cycles);
    printf("qwen_tcb_size=%" PRIu32 "\n", tcb_size);
    printf("qwen_tcb_checksum=0x%08" PRIx32 "\n", tcb_checksum);
    printf("qwen_tcb_magic=0x%08" PRIx32 "\n", OPENNPUX_QWEN_TCB_MAGIC);
    printf("qwen_tcb_op_count=%" PRIu32 "\n", result.completed_operators);
    if (rc == 0) {
        printf("qwen_tcb_stage=PASS\n");
    }
    opennpux_coral_close_shared_window(&window);
    return rc;
}

static int
print_qwen_run_tcb(struct opennpux_coral_device *dev, const char *path,
                   uint32_t entry, uint64_t polls)
{
    struct opennpux_qwen_run_result result;
    if (opennpux_qwen_run_hybrid_sim(path, &result) != 0) {
        perror("qwen-run-tcb");
        return 1;
    }

    uint8_t tcb[OPENNPUX_QWEN_TCB_MAX_SIZE];
    uint32_t tcb_size = 0;
    uint32_t tcb_checksum = 0;
    if (opennpux_qwen_build_tcb(&result, tcb, sizeof(tcb),
                                &tcb_size, &tcb_checksum) != 0) {
        perror("qwen-run-tcb");
        return 1;
    }
    const struct opennpux_qwen_tcb_op *expected_ops =
        (const struct opennpux_qwen_tcb_op *)(const void *)(
            tcb + sizeof(struct opennpux_qwen_tcb_header));
    const uint32_t expected_trace_checksum =
        opennpux_qwen_tcb_trace_checksum(expected_ops,
                                         result.completed_operators);

    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, tcb_size, &window) != 0) {
        return 1;
    }

    int rc = 0;
    uint32_t device_status = 0;
    if (copy_to_shared_window(&window, tcb, tcb_size) != 0 ||
        verify_staged_qwen_tcb(&result, &window) != 0) {
        perror("qwen-run-tcb");
        rc = 1;
    } else if (opennpux_coral_run(dev, entry, polls, &device_status) != 0) {
        perror("qwen-run-tcb");
        rc = 1;
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);

    const volatile struct opennpux_qwen_tcb_header *header =
        (const volatile struct opennpux_qwen_tcb_header *)(const volatile void *)
            window.bytes;
    printf("transport=%s\n", opennpux_coral_transport_name(dev->transport));
    printf("entry=0x%08" PRIx32 "\n", entry);
    printf("shared_base=0x%08" PRIx32 "\n", window.base);
    printf("shared_size=0x%08" PRIx32 "\n", window.size);
    printf("qwen_model=%s\n", result.info.name);
    printf("qwen_tcb_size=%" PRIu32 "\n", tcb_size);
    printf("qwen_tcb_checksum=0x%08" PRIx32 "\n", tcb_checksum);
    printf("qwen_device_status=0x%08" PRIx32 "\n", device_status);
    printf("qwen_tcb_state=%" PRIu32 "\n", header->tcb_state);
    printf("qwen_tcb_error=%" PRIu32 "\n", header->tcb_error);
    printf("qwen_device_checksum=0x%08" PRIx32 "\n",
           header->device_checksum);
    printf("qwen_device_completed_ops=%" PRIu32 "\n",
           header->device_completed_ops);
    printf("qwen_device_modeled_cycles=%" PRIu64 "\n",
           header->device_modeled_cycles);
    printf("qwen_device_op_mask=0x%08" PRIx32 "\n", header->device_op_mask);
    printf("qwen_expected_trace_checksum=0x%08" PRIx32 "\n",
           expected_trace_checksum);
    printf("qwen_device_trace_checksum=0x%08" PRIx32 "\n",
           header->device_trace_checksum);

    const volatile struct opennpux_qwen_tcb_op *ops =
        (const volatile struct opennpux_qwen_tcb_op *)(const volatile void *)(
            window.bytes + sizeof(struct opennpux_qwen_tcb_header));
    uint32_t device_op_completion_count = 0;
    for (uint32_t index = 0; index < result.completed_operators; ++index) {
        if (ops[index].reserved[0] == OPENNPUX_QWEN_TCB_OP_COMPLETE_MAGIC) {
            ++device_op_completion_count;
        }
    }
    printf("qwen_device_op_completion_count=%" PRIu32 "\n",
           device_op_completion_count);

    const int valid =
        rc == 0 &&
        header->tcb_state == OPENNPUX_QWEN_TCB_STATE_COMPLETE &&
        header->tcb_error == OPENNPUX_QWEN_TCB_ERROR_NONE &&
        header->device_checksum == tcb_checksum &&
        header->device_completed_ops == result.completed_operators &&
        header->device_modeled_cycles == result.modeled_cycles &&
        header->device_op_mask == result.info.op_mask &&
        header->device_trace_checksum == expected_trace_checksum &&
        device_op_completion_count == result.completed_operators &&
        (device_status & 0x1) != 0;
    if (valid) {
        printf("qwen_tcb_run=PASS\n");
    } else if (rc == 0) {
        fprintf(stderr, "Qwen TCB firmware validation failed\n");
        rc = 1;
    }
    opennpux_coral_close_shared_window(&window);
    return rc;
}

static int
print_qwen_device_run(struct opennpux_coral_device *dev, const char *path,
                      const char *prompt, uint32_t entry, uint64_t polls)
{
    struct opennpux_qwen_device_request request;
    struct opennpux_qwen_model_info expected;
    if (opennpux_qwen_build_device_request(path, prompt, &request,
                                           &expected) != 0) {
        perror("qwen-device-run prepare");
        return 1;
    }
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, sizeof(request), &window) != 0) {
        fprintf(stderr,
                "qwen-device-run requires a shared DMA window of at least %zu bytes\n",
                sizeof(request));
        return 1;
    }
    int rc = 0;
    uint32_t device_status = 0;
    if (copy_to_shared_window(&window, (const uint8_t *)&request,
                              sizeof(request)) != 0 ||
        opennpux_coral_run(dev, entry, polls, &device_status) != 0) {
        perror("qwen-device-run");
        rc = 1;
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    const volatile struct opennpux_qwen_device_request *result =
        (const volatile struct opennpux_qwen_device_request *)(
            const volatile void *)window.bytes;
    printf("transport=%s\n", opennpux_coral_transport_name(dev->transport));
    printf("qwen_prompt=%s\n", prompt);
    printf("qwen_prompt_checksum=0x%08" PRIx32 "\n",
           request.prompt_checksum);
    printf("qwen_device_status=0x%08" PRIx32 "\n", device_status);
    printf("qwen_device_state=%" PRIu32 "\n", result->state);
    printf("qwen_device_error=%" PRIu32 "\n", result->error);
    printf("qwen_device_completed_operators=%" PRIu32 "\n",
           result->completed_operators);
    printf("qwen_device_modeled_cycles=%" PRIu64 "\n",
           result->modeled_cycles);
    printf("qwen_logits_checksum=0x%08" PRIx32 "\n",
           result->logits_checksum);
    printf("qwen_next_token=%" PRIu32 "\n", result->next_token);
    const int valid = rc == 0 &&
        result->state == OPENNPUX_QWEN_DEVICE_COMPLETE &&
        result->error == 0 && result->completed_operators == 19 &&
        result->logits_checksum == expected.logits_checksum &&
        result->next_token == expected.next_token &&
        (device_status & 1) != 0;
    if (valid) {
        printf("qwen_device_inference=PASS\n");
    } else if (rc == 0) {
        fprintf(stderr, "Qwen device inference result mismatch\n");
        rc = 1;
    }
    opennpux_coral_close_shared_window(&window);
    return rc;
}

static void
print_info(const struct opennpux_coral_device *dev,
           const struct opennpux_coral_info *info)
{
    printf("base=0x%08" PRIx64 "\n", info->base);
    printf("transport=%s\n", opennpux_coral_transport_name(dev->transport));
    printf("backend_id=0x%08" PRIx32 "\n", info->backend_id);
    printf("backend=%s\n", opennpux_coral_backend_name(info->backend));
    printf("firmware_entry=0x%08" PRIx32 "\n", info->firmware_entry);
    printf("shared_base=0x%08" PRIx32 "\n", info->shared_base);
    printf("shared_size=0x%08" PRIx32 "\n", info->shared_size);
    printf("dma_requests=%" PRIu32 "\n", info->dma_requests);
    printf("dma_completions=%" PRIu32 "\n", info->dma_completions);
    printf("dma_errors=%" PRIu32 "\n", info->dma_errors);
    printf("dma_state=0x%08" PRIx32 "\n", info->dma_state);
    printf("reset_control=0x%08" PRIx32 "\n", info->reset_control);
    printf("status=0x%08" PRIx32 "\n", info->status);
    printf("driver_abi=%" PRIu32 "\n", info->abi_version);
    printf("driver_features=0x%08" PRIx32 "\n", info->features);
}

static int
print_mem_info(struct opennpux_coral_device *dev)
{
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, 0, &window) != 0) {
        return 1;
    }
    printf("shared_base=0x%08" PRIx32 "\n", window.base);
    printf("shared_size=0x%08" PRIx32 "\n", window.size);
    printf("shared_words=%" PRIu32 "\n", window.size / 4);
    opennpux_coral_close_shared_window(&window);
    return 0;
}

static int
print_read_shared_u32(struct opennpux_coral_device *dev, uint64_t offset)
{
    uint32_t value = 0;
    if (opennpux_coral_read_shared_u32(dev, offset, &value) != 0) {
        if (errno == EINVAL) {
            fprintf(stderr, "shared window offset must be 32-bit aligned\n");
        } else if (errno == ERANGE) {
            fprintf(stderr, "shared window offset 0x%" PRIx64
                    " is outside the shared window\n",
                    offset);
        }
        return 1;
    }
    printf("shared[0x%08" PRIx64 "]=0x%08" PRIx32 "\n", offset, value);
    return 0;
}

static int
print_write_shared_u32(struct opennpux_coral_device *dev, uint64_t offset,
                       uint32_t value)
{
    if (opennpux_coral_write_shared_u32(dev, offset, value) != 0) {
        if (errno == EINVAL) {
            fprintf(stderr, "shared window offset must be 32-bit aligned\n");
        } else if (errno == ERANGE) {
            fprintf(stderr, "shared window offset 0x%" PRIx64
                    " is outside the shared window\n",
                    offset);
        }
        return 1;
    }
    printf("shared[0x%08" PRIx64 "]=0x%08" PRIx32 "\n", offset, value);
    return 0;
}

static int
print_run(struct opennpux_coral_device *dev, uint32_t entry, uint64_t polls)
{
    struct opennpux_coral_info info;
    opennpux_coral_get_info(dev, &info);
    printf("transport=%s\n", opennpux_coral_transport_name(dev->transport));
    printf("backend=%s\n", opennpux_coral_backend_name(info.backend));
    printf("entry=0x%08" PRIx32 "\n", entry);

    uint32_t status = 0;
    const int result = opennpux_coral_run(dev, entry, polls, &status);
    printf("status=0x%08" PRIx32 "\n", status);
    if (result != 0) {
        if (errno == ETIMEDOUT) {
            fprintf(stderr, "Coral NPU did not halt within the poll limit\n");
        } else {
            fprintf(stderr, "Coral NPU reported an execution fault\n");
        }
        return 1;
    }
    return 0;
}

static int
print_dma_test(struct opennpux_coral_device *dev, uint32_t entry,
               uint64_t polls)
{
    struct opennpux_coral_info info;
    opennpux_coral_get_info(dev, &info);
    printf("transport=%s\n", opennpux_coral_transport_name(dev->transport));
    printf("backend=%s\n", opennpux_coral_backend_name(info.backend));
    printf("entry=0x%08" PRIx32 "\n", entry);

    struct opennpux_coral_dma_test_result result;
    const int run_result = opennpux_coral_dma_test(dev, entry, polls, &result);
    printf("status=0x%08" PRIx32 "\n", result.status);

    printf("dma_result=%" PRIu32 "\n", result.result);
    printf("dma_magic=0x%08" PRIx32 "\n", result.magic);
    printf("dma_requests=%" PRIu32 "\n", result.requests);
    printf("dma_completions=%" PRIu32 "\n", result.completions);
    printf("dma_errors=%" PRIu32 "\n", result.errors);
    printf("dma_state=0x%08" PRIx32 "\n", result.state);
    if (run_result != 0) {
        if (errno == ETIMEDOUT) {
            fprintf(stderr, "Coral NPU did not halt within the poll limit\n");
        }
        fprintf(stderr, "Coral coherent DMA smoke failed\n");
        return 1;
    }
    printf("dma_test=PASS\n");
    return 0;
}

static int
print_vector_add(struct opennpux_coral_device *dev, uint32_t entry,
                 uint32_t opcode, uint32_t element_count, uint64_t polls)
{
    struct opennpux_coral_info info;
    opennpux_coral_get_info(dev, &info);
    printf("transport=%s\n", opennpux_coral_transport_name(dev->transport));
    printf("backend=%s\n", opennpux_coral_backend_name(info.backend));
    printf("entry=0x%08" PRIx32 "\n", entry);

    struct opennpux_coral_vector_add_result result;
    const int run_result = opennpux_coral_vector_add_test(
        dev, entry, opcode, element_count, polls, &result);
    printf("opcode=%" PRIu32 "\n", result.opcode);
    printf("command_status=%" PRIu32 "\n", result.status);
    printf("command_error=%" PRIu32 "\n", result.error_code);
    printf("completed_elements=%" PRIu32 "\n", result.completed_elements);
    printf("element_count=%" PRIu32 "\n", result.element_count);
    printf("output_checksum=0x%08" PRIx32 "\n", result.checksum);
    printf("accelerator_cycles=%" PRIu32 "\n", result.accelerator_cycles);
    if (run_result != 0) {
        fprintf(stderr, "Coral vector-add command failed\n");
        return 1;
    }
    printf("vector_add=PASS\n");
    return 0;
}

static int
print_model_run(struct opennpux_coral_device *dev, uint32_t entry,
                const char *path, uint64_t polls)
{
    struct opennpux_coral_model_result result;
    const int run_result =
        opennpux_coral_run_model_file(dev, entry, path, polls, &result);
    printf("model_commands=%" PRIu32 "\n", result.command_count);
    printf("completed_commands=%" PRIu32 "\n", result.completed_commands);
    printf("model_checksum=0x%08" PRIx32 "\n", result.output_checksum);
    printf("accelerator_cycles=%" PRIu32 "\n", result.accelerator_cycles);
    printf("model_dma_requests=%" PRIu32 "\n", result.dma_requests);
    printf("model_dma_completions=%" PRIu32 "\n", result.dma_completions);
    printf("model_dma_errors=%" PRIu32 "\n", result.dma_errors);
    printf("host_elapsed_ns=%" PRIu64 "\n", result.host_elapsed_ns);
    if (run_result != 0) {
        perror("model-run");
        return 1;
    }
    printf("model_run=PASS\n");
    return 0;
}

static int
print_mobilenet_test(struct opennpux_coral_device *dev, uint32_t entry,
                     uint64_t polls)
{
    struct opennpux_coral_mobilenet_result result;
    printf("mobilenet_prepare=mailbox-only\n");
    printf("mobilenet_run=started\n");
    fflush(stdout);
    const int run_result =
        opennpux_coral_mobilenet_test(dev, entry, polls, &result);
    printf("status=0x%08" PRIx32 "\n", result.device_status);
    printf("mobilenet_state=0x%08" PRIx32 "\n", result.state);
    printf("mobilenet_error=%" PRIu32 "\n", result.error_code);
    printf("mobilenet_npu_cycles=%" PRIu64 "\n", result.npu_cycles);
    printf("mobilenet_output_checksum=0x%08" PRIx32 "\n",
           result.output_checksum);
    printf("mobilenet_output_bytes=%" PRIu32 "\n", result.output_bytes);
    printf("mobilenet_operation_count=%" PRIu64 "\n",
           result.operation_count);
    printf("mobilenet_bytes_read=%" PRIu64 "\n", result.bytes_read);
    printf("mobilenet_bytes_written=%" PRIu64 "\n", result.bytes_written);
    printf("mobilenet_dma_requests=%" PRIu32 "\n", result.dma_requests);
    printf("mobilenet_dma_completions=%" PRIu32 "\n",
           result.dma_completions);
    printf("mobilenet_dma_errors=%" PRIu32 "\n", result.dma_errors);
    printf("mobilenet_output=");
    for (uint32_t i = 0; i < result.output_count &&
                         i < OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT; ++i) {
        printf("%s%" PRId32, i == 0 ? "" : ",", result.output[i]);
    }
    printf("\n");
    if (run_result != 0) {
        perror("mobilenet-test");
        return 1;
    }
    printf("mobilenet_test=PASS\n");
    return 0;
}

int
main(int argc, char **argv)
{
    if (argc < 2 || argc > 8) {
        usage(argv[0]);
        return 2;
    }

    const int command_info = strcmp(argv[1], "info") == 0;
    const int command_run = strcmp(argv[1], "run") == 0;
    const int command_dma_test = strcmp(argv[1], "dma-test") == 0;
    const int command_vector_add = strcmp(argv[1], "vector-add") == 0;
    const int command_vector_add_custom =
        strcmp(argv[1], "vector-add-custom") == 0;
    const int command_model_run = strcmp(argv[1], "model-run") == 0;
    const int command_model_info_v2 = strcmp(argv[1], "model-info-v2") == 0;
    const int command_executable_run_paged =
        strcmp(argv[1], "executable-run-paged") == 0;
    const int command_executable_run =
        strcmp(argv[1], "executable-run") == 0 ||
        command_executable_run_paged;
    const int command_qwen_info = strcmp(argv[1], "qwen-info") == 0;
    const int command_qwen_run = strcmp(argv[1], "qwen-run") == 0;
    const int command_qwen_stage_tcb = strcmp(argv[1], "qwen-stage-tcb") == 0;
    const int command_qwen_run_tcb = strcmp(argv[1], "qwen-run-tcb") == 0;
    const int command_qwen_device_run =
        strcmp(argv[1], "qwen-device-run") == 0;
    const int command_mobilenet_test =
        strcmp(argv[1], "mobilenet-test") == 0;
    const int command_mem_info = strcmp(argv[1], "mem-info") == 0;
    const int command_mem_clear = strcmp(argv[1], "mem-clear") == 0;
    const int command_mem_load = strcmp(argv[1], "mem-load") == 0;
    const int command_mem_read32 = strcmp(argv[1], "mem-read32") == 0;
    const int command_mem_write32 = strcmp(argv[1], "mem-write32") == 0;
    if (!command_info && !command_run && !command_dma_test &&
        !command_vector_add && !command_vector_add_custom &&
        !command_model_run && !command_model_info_v2 && !command_executable_run &&
        !command_qwen_info && !command_qwen_run &&
        !command_qwen_stage_tcb && !command_qwen_run_tcb &&
        !command_qwen_device_run &&
        !command_mobilenet_test &&
        !command_mem_info && !command_mem_clear && !command_mem_load &&
        !command_mem_read32 &&
        !command_mem_write32) {
        usage(argv[0]);
        return 2;
    }
    if ((command_info && argc > 3) ||
        (command_run && argc > 5) ||
        (command_dma_test && argc > 4) ||
        ((command_vector_add || command_vector_add_custom) &&
         (argc < 3 || argc > 5)) ||
        (command_model_run && (argc < 3 || argc > 5)) ||
        (command_model_info_v2 && argc != 3) ||
        (command_executable_run &&
         (argc < 3 || argc > (command_executable_run_paged ? 8 : 7))) ||
        (command_qwen_info && argc != 3) ||
        (command_qwen_run && (argc < 3 || argc > 4)) ||
        (command_qwen_stage_tcb && (argc < 3 || argc > 4)) ||
        (command_qwen_run_tcb && (argc < 3 || argc > 5)) ||
        (command_qwen_device_run && (argc < 4 || argc > 6)) ||
        (command_mobilenet_test && argc > 4) ||
        (command_mem_info && argc > 3) ||
        (command_mem_clear && argc > 3) ||
        (command_mem_load && (argc < 3 || argc > 4)) ||
        (command_mem_read32 && (argc < 3 || argc > 4)) ||
        (command_mem_write32 && (argc < 4 || argc > 5))) {
        usage(argv[0]);
        return 2;
    }

    if (command_qwen_info) {
        return print_qwen_info(argv[2]);
    }
    if (command_model_info_v2) {
        return print_model_info_v2(argv[2]);
    }
    if (command_qwen_run) {
        return print_qwen_run(argv[2], argc >= 4 ? argv[3] : NULL);
    }

    uint64_t base = OPENNPUX_CORAL_DEFAULT_BASE;
    uint64_t shared_offset = 0;
    uint64_t shared_value = 0;
    uint64_t vector_elements = 0;
    const char *model_path = NULL;
    const char *weight_page_path = NULL;
    const char *weight_manifest_path = NULL;
    const char *weight_range_path = NULL;
    uint32_t executable_entry = OPENNPUX_NPU_ENTRY_DECODE;
    int executable_poll_arg = 5;
    if (command_mem_read32 || command_mem_write32) {
        const int base_arg = command_mem_read32 ? 3 : 4;
        if (opennpux_coral_parse_u64(argv[2], &shared_offset) != 0) {
            fprintf(stderr, "invalid shared offset: %s\n", argv[2]);
            return 2;
        }
        if (command_mem_write32 &&
            (opennpux_coral_parse_u64(argv[3], &shared_value) != 0 ||
             shared_value > UINT32_MAX)) {
            fprintf(stderr, "invalid 32-bit value: %s\n", argv[3]);
            return 2;
        }
        if (argc > base_arg &&
            opennpux_coral_parse_u64(argv[base_arg], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[base_arg]);
            return 2;
        }
    } else if (command_mem_load) {
        model_path = argv[2];
        if (argc >= 4 && opennpux_coral_parse_u64(argv[3], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[3]);
            return 2;
        }
    } else if (command_executable_run) {
        model_path = argv[2];
        if (argc >= 4) {
            if (strcmp(argv[3], "prefill") == 0) {
                executable_entry = OPENNPUX_NPU_ENTRY_PREFILL;
            } else if (strcmp(argv[3], "decode") != 0) {
                fprintf(stderr, "invalid executable entry: %s\n", argv[3]);
                return 2;
            }
        }
        if (argc >= 5) {
            const size_t argument_length = strlen(argv[4]);
            const int is_manifest = command_executable_run_paged &&
                argument_length >= 5 &&
                strcmp(argv[4] + argument_length - 5, ".npxm") == 0;
            if (is_manifest) {
                if (argc < 6) {
                    fprintf(stderr, "missing NPU weight range index\n");
                    return 2;
                }
                weight_manifest_path = argv[4];
                weight_range_path = argv[5];
                executable_poll_arg = 7;
                if (argc >= 7 &&
                    opennpux_coral_parse_u64(argv[6], &base) != 0) {
                    fprintf(stderr, "invalid base address: %s\n", argv[6]);
                    return 2;
                }
            } else if (strchr(argv[4], '/') != NULL) {
                weight_page_path = argv[4];
                executable_poll_arg = 6;
                if (argc >= 6 &&
                    opennpux_coral_parse_u64(argv[5], &base) != 0) {
                    fprintf(stderr, "invalid base address: %s\n", argv[5]);
                    return 2;
                }
            } else if (opennpux_coral_parse_u64(argv[4], &base) != 0) {
                fprintf(stderr, "invalid base address or weight page: %s\n",
                        argv[4]);
                return 2;
            }
        }
    } else if (command_model_run) {
        model_path = argv[2];
        if (argc >= 4 && opennpux_coral_parse_u64(argv[3], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[3]);
            return 2;
        }
    } else if (command_qwen_stage_tcb || command_qwen_run_tcb) {
        model_path = argv[2];
        if (argc >= 4 && opennpux_coral_parse_u64(argv[3], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[3]);
            return 2;
        }
    } else if (command_qwen_device_run) {
        model_path = argv[2];
        if (argc >= 5 && opennpux_coral_parse_u64(argv[4], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[4]);
            return 2;
        }
    } else if (command_mobilenet_test) {
        if (argc >= 3 && opennpux_coral_parse_u64(argv[2], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[2]);
            return 2;
        }
    } else if (command_vector_add || command_vector_add_custom) {
        if (opennpux_coral_parse_u64(argv[2], &vector_elements) != 0 ||
            vector_elements == 0 ||
            vector_elements > OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS) {
            fprintf(stderr, "invalid vector element count: %s\n", argv[2]);
            return 2;
        }
        if (argc >= 4 && opennpux_coral_parse_u64(argv[3], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[3]);
            return 2;
        }
    } else if (argc >= 3 &&
               opennpux_coral_parse_u64(argv[2], &base) != 0) {
        fprintf(stderr, "invalid base address: %s\n", argv[2]);
        return 2;
    }

    struct opennpux_coral_device dev;
    if (opennpux_coral_open(&dev, base) != 0) {
        return 1;
    }

    if (command_info) {
        struct opennpux_coral_info info;
        opennpux_coral_get_info(&dev, &info);
        print_info(&dev, &info);
        opennpux_coral_close(&dev);
        return 0;
    }

    if (command_mem_info) {
        const int result = print_mem_info(&dev);
        opennpux_coral_close(&dev);
        return result;
    }

    if (command_mem_clear) {
        const int result = opennpux_coral_clear_shared_window(&dev);
        if (result == 0) {
            printf("shared_clear=PASS\n");
        }
        opennpux_coral_close(&dev);
        return result == 0 ? 0 : 1;
    }

    if (command_mem_load) {
        const int result = print_mem_load(&dev, model_path);
        opennpux_coral_close(&dev);
        return result;
    }

    if (command_mem_read32) {
        const int result = print_read_shared_u32(&dev, shared_offset);
        opennpux_coral_close(&dev);
        return result;
    }

    if (command_mem_write32) {
        const int result =
            print_write_shared_u32(&dev, shared_offset,
                                   (uint32_t)shared_value);
        opennpux_coral_close(&dev);
        return result;
    }

    if (command_qwen_stage_tcb) {
        const int result = print_qwen_stage_tcb(&dev, model_path);
        opennpux_coral_close(&dev);
        return result;
    }

    struct opennpux_coral_info info;
    opennpux_coral_get_info(&dev, &info);
    uint64_t entry = info.firmware_entry;
    uint64_t polls =
        (command_dma_test || command_vector_add || command_vector_add_custom ||
         command_model_run || command_mobilenet_test || command_qwen_run_tcb ||
         command_qwen_device_run ||
         command_executable_run) ?
            100000 : 1000;
    if (command_run && argc >= 4 &&
        opennpux_coral_parse_u64(argv[3], &entry) != 0) {
        fprintf(stderr, "invalid entry address: %s\n", argv[3]);
        opennpux_coral_close(&dev);
        return 2;
    }
    const int poll_arg =
        command_executable_run ? executable_poll_arg :
        command_mobilenet_test ? 3 :
        command_qwen_run_tcb ? 4 :
        command_qwen_device_run ? 5 :
        (command_model_run || command_vector_add ||
         command_vector_add_custom) ? 4 : (command_dma_test ? 3 : 4);
    if (argc > poll_arg &&
        (opennpux_coral_parse_u64(argv[poll_arg], &polls) != 0 ||
         polls == 0)) {
        fprintf(stderr, "invalid poll count: %s\n", argv[poll_arg]);
        opennpux_coral_close(&dev);
        return 2;
    }

    int result;
    if (command_run) {
        result = print_run(&dev, (uint32_t)entry, polls);
    } else if (command_executable_run) {
        result = print_executable_run(&dev, model_path, weight_page_path,
                                      weight_manifest_path, weight_range_path,
                                      executable_entry, (uint32_t)entry,
                                      command_executable_run_paged, polls);
    } else if (command_model_run) {
        result = print_model_run(&dev, (uint32_t)entry, model_path, polls);
    } else if (command_qwen_run_tcb) {
        result = print_qwen_run_tcb(&dev, model_path, (uint32_t)entry, polls);
    } else if (command_qwen_device_run) {
        result = print_qwen_device_run(&dev, model_path, argv[3],
                                       (uint32_t)entry, polls);
    } else if (command_mobilenet_test) {
        result = print_mobilenet_test(&dev, (uint32_t)entry, polls);
    } else if (command_vector_add || command_vector_add_custom) {
        const uint32_t opcode = command_vector_add_custom ?
            OPENNPUX_CORAL_OPCODE_VECTOR_ADD_CUSTOM_U32 :
            OPENNPUX_CORAL_OPCODE_VECTOR_ADD_U32;
        result = print_vector_add(&dev, (uint32_t)entry,
                                  opcode, (uint32_t)vector_elements, polls);
    } else {
        result = print_dma_test(&dev, (uint32_t)entry, polls);
    }
    opennpux_coral_close(&dev);
    return result;
}
