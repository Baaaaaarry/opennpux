#include "opennpux/coral_runtime.h"
#include "opennpux/model_package.h"
#include "opennpux/npu_executable.h"
#include "opennpux/qwen_model.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_EXTMEM_BASE UINT64_C(0x20000000)
#define NPU_WEIGHT_PAGE_SIZE UINT32_C(4096)
#define NPU_IO_BUFFER_SIZE UINT32_C(128)
#define NPU_STATE_BUFFER_SIZE UINT32_C(256)
#define NPU_SCRATCH_BUFFER_SIZE UINT32_C(256)

static int copy_to_shared_window(
    struct opennpux_coral_shared_window *window, const uint8_t *source,
    uint32_t size);

static size_t
align_npu_record(size_t value)
{
    return (value + OPENNPUX_NPU_RECORD_ALIGNMENT - 1) &
        ~(size_t)(OPENNPUX_NPU_RECORD_ALIGNMENT - 1);
}

static int
load_weight_page(const char *path, uint8_t page[NPU_WEIGHT_PAGE_SIZE])
{
    memset(page, 0, NPU_WEIGHT_PAGE_SIZE);
    if (path == NULL) {
        for (uint32_t index = 0; index < NPU_WEIGHT_PAGE_SIZE; ++index) {
            page[index] = (uint8_t)(index * 37u + 11u);
        }
        return 0;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    const size_t bytes = fread(page, 1, NPU_WEIGHT_PAGE_SIZE, file);
    const int failed = ferror(file) || bytes == 0;
    fclose(file);
    if (failed) {
        errno = EIO;
        return -1;
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
            "  %s qwen-info <qwen-tiny.npxm>\n"
            "  %s qwen-run <qwen-tiny.npxm> [golden-package|hybrid-sim]\n"
            "  %s qwen-stage-tcb <qwen-tiny.npxm> [base]\n"
            "  %s qwen-run-tcb <qwen-tiny.npxm> [base [poll-count]]\n"
            "  %s mobilenet-test [base [poll-count]]\n"
            "  %s mem-info [base]\n"
            "  %s mem-clear [base]\n"
            "  %s mem-read32 <offset> [base]\n"
            "  %s mem-write32 <offset> <value> [base]\n"
            "features: qwen-run-tcb-v2\n",
            prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog,
            prog, prog, prog, prog, prog, prog);
}

static int
print_executable_run(struct opennpux_coral_device *dev, const char *path,
                     const char *weight_page_path, uint32_t entry_point,
                     uint32_t firmware_entry,
                     uint64_t polls)
{
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
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, 65536, &window) != 0) {
        opennpux_npu_executable_unload(&executable);
        return 1;
    }
    struct opennpux_npu_tensor_binding bindings[5];
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

    uint8_t weight_page[NPU_WEIGHT_PAGE_SIZE];
    if (load_weight_page(weight_page_path, weight_page) != 0) {
        perror("executable-run weight page");
        opennpux_coral_close_shared_window(&window);
        opennpux_npu_executable_unload(&executable);
        return 1;
    }

    uint8_t *submission = malloc(window.size);
    size_t submission_size = 0;
    int rc = 1;
    if (submission == NULL || opennpux_npu_executable_instantiate(
            &executable, entry_point, 1, 1, bindings, 5, submission,
            window.size, &submission_size) != 0) {
        perror("executable-run instantiate");
        goto out;
    }
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
    if (completion_offset > window.size ||
        sizeof(struct opennpux_npu_completion) > window.size - completion_offset ||
        data_offset > window.size) {
        errno = ENOSPC;
        perror("executable-run completion");
        goto out;
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
    header->completion_address = NPU_EXTMEM_BASE + completion_offset;
    header->checksum = 0;
    header->checksum = opennpux_npu_submission_checksum(
        submission, submission_size);
    if (opennpux_npu_submission_validate(submission, submission_size) != 0 ||
        copy_to_shared_window(&window, submission, (uint32_t)submission_size) != 0) {
        perror("executable-run stage");
        goto out;
    }
    for (size_t index = input_offset; index < data_offset; ++index) {
        window.bytes[index] = 0;
    }
    for (uint32_t index = 0; index < NPU_WEIGHT_PAGE_SIZE; ++index) {
        window.bytes[weight_offset + index] = weight_page[index];
    }
    volatile struct opennpux_npu_completion *completion =
        (volatile struct opennpux_npu_completion *)(volatile void *)(
            window.bytes + completion_offset);
    memset((void *)(uintptr_t)completion, 0, sizeof(*completion));
    __sync_synchronize();
    uint32_t device_status = 0;
    if (opennpux_coral_run(dev, firmware_entry, polls, &device_status) != 0) {
        perror("executable-run device");
        goto out;
    }
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
    printf("weight_binding=%" PRIu64 "\n",
           resource_bindings & OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("state_binding=%" PRIu64 "\n",
           (resource_bindings >> OPENNPUX_NPU_RESOURCE_STATE_SHIFT) &
               OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("scratch_binding=%" PRIu64 "\n",
           (resource_bindings >> OPENNPUX_NPU_RESOURCE_SCRATCH_SHIFT) &
               OPENNPUX_NPU_RUNTIME_FIELD_MASK);
    printf("weight_page_source=%s\n",
           weight_page_path == NULL ? "deterministic" : weight_page_path);
    printf("weight_page_bytes=%" PRIu32 "\n", NPU_WEIGHT_PAGE_SIZE);
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
    if (completion->magic != OPENNPUX_NPU_COMPLETION_MAGIC ||
        completion->version != OPENNPUX_NPU_COMPLETION_VERSION ||
        completion->sequence != header->sequence ||
        completion->state != OPENNPUX_NPU_COMPLETION_SUCCESS ||
        completion->error_code != 0 ||
        completion->completed_commands != header->command_count ||
        completion->reserved[0] != header->command_count ||
        completion->reserved[1] == 0 || trace == NULL ||
        trace->command_count != header->command_count) {
        errno = EIO;
        perror("executable-run completion validation");
        goto out;
    }
    puts("executable_run=PASS");
    rc = 0;
out:
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
    for (uint32_t index = 0; index < size; ++index) {
        window->bytes[index] = source[index];
    }
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
    __sync_synchronize();

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
    if (argc < 2 || argc > 7) {
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
    const int command_executable_run = strcmp(argv[1], "executable-run") == 0;
    const int command_qwen_info = strcmp(argv[1], "qwen-info") == 0;
    const int command_qwen_run = strcmp(argv[1], "qwen-run") == 0;
    const int command_qwen_stage_tcb = strcmp(argv[1], "qwen-stage-tcb") == 0;
    const int command_qwen_run_tcb = strcmp(argv[1], "qwen-run-tcb") == 0;
    const int command_mobilenet_test =
        strcmp(argv[1], "mobilenet-test") == 0;
    const int command_mem_info = strcmp(argv[1], "mem-info") == 0;
    const int command_mem_clear = strcmp(argv[1], "mem-clear") == 0;
    const int command_mem_read32 = strcmp(argv[1], "mem-read32") == 0;
    const int command_mem_write32 = strcmp(argv[1], "mem-write32") == 0;
    if (!command_info && !command_run && !command_dma_test &&
        !command_vector_add && !command_vector_add_custom &&
        !command_model_run && !command_model_info_v2 && !command_executable_run &&
        !command_qwen_info && !command_qwen_run &&
        !command_qwen_stage_tcb && !command_qwen_run_tcb &&
        !command_mobilenet_test &&
        !command_mem_info && !command_mem_clear && !command_mem_read32 &&
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
        (command_executable_run && (argc < 3 || argc > 7)) ||
        (command_qwen_info && argc != 3) ||
        (command_qwen_run && (argc < 3 || argc > 4)) ||
        (command_qwen_stage_tcb && (argc < 3 || argc > 4)) ||
        (command_qwen_run_tcb && (argc < 3 || argc > 5)) ||
        (command_mobilenet_test && argc > 4) ||
        (command_mem_info && argc > 3) ||
        (command_mem_clear && argc > 3) ||
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
            if (strchr(argv[4], '/') != NULL) {
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
                                      executable_entry, (uint32_t)entry, polls);
    } else if (command_model_run) {
        result = print_model_run(&dev, (uint32_t)entry, model_path, polls);
    } else if (command_qwen_run_tcb) {
        result = print_qwen_run_tcb(&dev, model_path, (uint32_t)entry, polls);
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
