#include "opennpux/coral_runtime.h"
#include "opennpux/qwen_model.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
            "  %s qwen-info <qwen-tiny.npxm>\n"
            "  %s qwen-run <qwen-tiny.npxm> [golden-package|hybrid-sim]\n"
            "  %s mobilenet-test [base [poll-count]]\n"
            "  %s mem-info [base]\n"
            "  %s mem-clear [base]\n"
            "  %s mem-read32 <offset> [base]\n"
            "  %s mem-write32 <offset> <value> [base]\n",
            prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog,
            prog, prog);
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
    printf("qwen_logits_checksum=0x%08" PRIx32 "\n",
           result.output_checksum);
    printf("qwen_next_token=%" PRIu32 "\n", result.next_token);
    printf("qwen_run=PASS\n");
    return 0;
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
    if (argc < 2 || argc > 6) {
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
    const int command_qwen_info = strcmp(argv[1], "qwen-info") == 0;
    const int command_qwen_run = strcmp(argv[1], "qwen-run") == 0;
    const int command_mobilenet_test =
        strcmp(argv[1], "mobilenet-test") == 0;
    const int command_mem_info = strcmp(argv[1], "mem-info") == 0;
    const int command_mem_clear = strcmp(argv[1], "mem-clear") == 0;
    const int command_mem_read32 = strcmp(argv[1], "mem-read32") == 0;
    const int command_mem_write32 = strcmp(argv[1], "mem-write32") == 0;
    if (!command_info && !command_run && !command_dma_test &&
        !command_vector_add && !command_vector_add_custom &&
        !command_model_run && !command_qwen_info && !command_qwen_run &&
        !command_mobilenet_test && !command_mem_info && !command_mem_clear &&
        !command_mem_read32 && !command_mem_write32) {
        usage(argv[0]);
        return 2;
    }
    if ((command_info && argc > 3) ||
        (command_run && argc > 5) ||
        (command_dma_test && argc > 4) ||
        ((command_vector_add || command_vector_add_custom) &&
         (argc < 3 || argc > 5)) ||
        (command_model_run && (argc < 3 || argc > 5)) ||
        (command_qwen_info && argc != 3) ||
        (command_qwen_run && (argc < 3 || argc > 4)) ||
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
    if (command_qwen_run) {
        return print_qwen_run(argv[2], argc >= 4 ? argv[3] : NULL);
    }

    uint64_t base = OPENNPUX_CORAL_DEFAULT_BASE;
    uint64_t shared_offset = 0;
    uint64_t shared_value = 0;
    uint64_t vector_elements = 0;
    const char *model_path = NULL;
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
    } else if (command_model_run) {
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

    struct opennpux_coral_info info;
    opennpux_coral_get_info(&dev, &info);
    uint64_t entry = info.firmware_entry;
    uint64_t polls =
        (command_dma_test || command_vector_add || command_vector_add_custom ||
         command_model_run || command_mobilenet_test) ?
            100000 : 1000;
    if (command_run && argc >= 4 &&
        opennpux_coral_parse_u64(argv[3], &entry) != 0) {
        fprintf(stderr, "invalid entry address: %s\n", argv[3]);
        opennpux_coral_close(&dev);
        return 2;
    }
    const int poll_arg =
        command_mobilenet_test ? 3 :
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
    } else if (command_model_run) {
        result = print_model_run(&dev, (uint32_t)entry, model_path, polls);
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
