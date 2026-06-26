#include "opennpux/coral_runtime.h"

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
            "  %s mem-info [base]\n"
            "  %s mem-clear [base]\n"
            "  %s mem-read32 <offset> [base]\n"
            "  %s mem-write32 <offset> <value> [base]\n",
            prog, prog, prog, prog, prog, prog, prog);
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
    const int command_mem_info = strcmp(argv[1], "mem-info") == 0;
    const int command_mem_clear = strcmp(argv[1], "mem-clear") == 0;
    const int command_mem_read32 = strcmp(argv[1], "mem-read32") == 0;
    const int command_mem_write32 = strcmp(argv[1], "mem-write32") == 0;
    if (!command_info && !command_run && !command_dma_test &&
        !command_mem_info && !command_mem_clear && !command_mem_read32 &&
        !command_mem_write32) {
        usage(argv[0]);
        return 2;
    }
    if ((command_info && argc > 3) ||
        (command_run && argc > 5) ||
        (command_dma_test && argc > 4) ||
        (command_mem_info && argc > 3) ||
        (command_mem_clear && argc > 3) ||
        (command_mem_read32 && (argc < 3 || argc > 4)) ||
        (command_mem_write32 && (argc < 4 || argc > 5))) {
        usage(argv[0]);
        return 2;
    }

    uint64_t base = OPENNPUX_CORAL_DEFAULT_BASE;
    uint64_t shared_offset = 0;
    uint64_t shared_value = 0;
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
    uint64_t polls = command_dma_test ? 100000 : 1000;
    if (command_run && argc >= 4 &&
        opennpux_coral_parse_u64(argv[3], &entry) != 0) {
        fprintf(stderr, "invalid entry address: %s\n", argv[3]);
        opennpux_coral_close(&dev);
        return 2;
    }
    const int poll_arg = command_dma_test ? 3 : 4;
    if (argc > poll_arg &&
        (opennpux_coral_parse_u64(argv[poll_arg], &polls) != 0 ||
         polls == 0)) {
        fprintf(stderr, "invalid poll count: %s\n", argv[poll_arg]);
        opennpux_coral_close(&dev);
        return 2;
    }

    const int result = command_run ?
        print_run(&dev, (uint32_t)entry, polls) :
        print_dma_test(&dev, (uint32_t)entry, polls);
    opennpux_coral_close(&dev);
    return result;
}
