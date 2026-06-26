#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define DEFAULT_BASE UINT64_C(0x1d000000)
#define RESET_CONTROL UINT64_C(0x30000)
#define PC_START UINT64_C(0x30004)
#define STATUS UINT64_C(0x30008)
#define DMA_ERRORS UINT64_C(0x30fe0)
#define DMA_REQUESTS UINT64_C(0x30fe4)
#define DMA_COMPLETIONS UINT64_C(0x30fe8)
#define DMA_STATE UINT64_C(0x30fec)
#define SHARED_BASE UINT64_C(0x30ff0)
#define SHARED_SIZE UINT64_C(0x30ff4)
#define FIRMWARE_ENTRY UINT64_C(0x30ff8)
#define BACKEND_ID UINT64_C(0x30ffc)

#define BACKEND_STAGE_A UINT32_C(0x4e505501)
#define BACKEND_VERILATED UINT32_C(0x4e505502)

struct coral_regs {
    int fd;
    size_t page_size;
    void *mapping;
    uint64_t page_base;
};

struct coral_shared_window {
    uint32_t base;
    uint32_t size;
    size_t map_size;
    void *mapping;
    volatile uint8_t *bytes;
};

static uint32_t
read_reg(struct coral_regs *regs, uint64_t base, uint64_t offset);

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

static int
parse_u64(const char *text, uint64_t *value)
{
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 0);
    if (errno != 0 || end == text || *end != '\0') {
        return -1;
    }
    *value = parsed;
    return 0;
}

static const char *
backend_name(uint32_t id)
{
    switch (id) {
      case BACKEND_STAGE_A:
        return "stage-a";
      case BACKEND_VERILATED:
        return "verilated-coral";
      default:
        return "unknown";
    }
}

static int
open_regs(struct coral_regs *regs, uint64_t base)
{
    memset(regs, 0, sizeof(*regs));
    regs->fd = -1;
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        perror("sysconf(_SC_PAGESIZE)");
        return -1;
    }

    regs->page_size = (size_t)page_size;
    const uint64_t csr_addr = base + RESET_CONTROL;
    const uint64_t page_mask = (uint64_t)page_size - 1;
    regs->page_base = csr_addr & ~page_mask;
    regs->fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (regs->fd < 0) {
        perror("open(/dev/mem)");
        return -1;
    }

    regs->mapping = mmap(NULL, regs->page_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, regs->fd, (off_t)regs->page_base);
    if (regs->mapping == MAP_FAILED) {
        regs->mapping = NULL;
        perror("mmap(/dev/mem)");
        close(regs->fd);
        regs->fd = -1;
        return -1;
    }
    return 0;
}
static void
close_regs(struct coral_regs *regs)
{
    if (regs->mapping != NULL) {
        munmap(regs->mapping, regs->page_size);
    }
    if (regs->fd >= 0) {
        close(regs->fd);
    }
}

static int
open_shared_window(struct coral_regs *regs, uint64_t base, size_t min_size,
                   struct coral_shared_window *window)
{
    memset(window, 0, sizeof(*window));
    window->base = read_reg(regs, base, SHARED_BASE);
    window->size = read_reg(regs, base, SHARED_SIZE);
    if (window->size < min_size) {
        fprintf(stderr,
                "Coral shared DMA window is too small: size=0x%08" PRIx32
                " required=0x%zx\n",
                window->size, min_size);
        return -1;
    }

    const uint64_t page_mask = (uint64_t)regs->page_size - 1;
    const uint64_t map_base = window->base & ~page_mask;
    const size_t map_offset = window->base & page_mask;
    window->map_size = map_offset + window->size;
    window->mapping = mmap(NULL, window->map_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, regs->fd, (off_t)map_base);
    if (window->mapping == MAP_FAILED) {
        window->mapping = NULL;
        perror("mmap(Coral shared DMA window)");
        return -1;
    }
    window->bytes = (volatile uint8_t *)window->mapping + map_offset;
    return 0;
}

static void
close_shared_window(struct coral_shared_window *window)
{
    if (window->mapping != NULL) {
        munmap(window->mapping, window->map_size);
    }
    memset(window, 0, sizeof(*window));
}

static int
check_shared_u32_access(const struct coral_shared_window *window,
                        uint64_t offset)
{
    if ((offset & 0x3) != 0) {
        fprintf(stderr, "shared window offset must be 32-bit aligned\n");
        return -1;
    }
    if (offset > UINT32_MAX || offset + sizeof(uint32_t) > window->size) {
        fprintf(stderr,
                "shared window offset 0x%" PRIx64
                " is outside size 0x%08" PRIx32 "\n",
                offset, window->size);
        return -1;
    }
    return 0;
}

static volatile uint32_t *
reg_ptr(struct coral_regs *regs, uint64_t base, uint64_t offset)
{
    const uint64_t addr = base + offset;
    if (addr < regs->page_base ||
        addr + sizeof(uint32_t) > regs->page_base + regs->page_size) {
        return NULL;
    }
    return (volatile uint32_t *)((uint8_t *)regs->mapping +
                                 (addr - regs->page_base));
}

static uint32_t
read_reg(struct coral_regs *regs, uint64_t base, uint64_t offset)
{
    volatile uint32_t *reg = reg_ptr(regs, base, offset);
    if (reg == NULL) {
        fprintf(stderr, "register offset 0x%" PRIx64 " is not mapped\n",
                offset);
        exit(1);
    }
    return *reg;
}

static void
write_reg(struct coral_regs *regs, uint64_t base, uint64_t offset,
          uint32_t value)
{
    volatile uint32_t *reg = reg_ptr(regs, base, offset);
    if (reg == NULL) {
        fprintf(stderr, "register offset 0x%" PRIx64 " is not mapped\n",
                offset);
        exit(1);
    }
    *reg = value;
    __sync_synchronize();
}

static void
print_info(struct coral_regs *regs, uint64_t base)
{
    const uint32_t backend = read_reg(regs, base, BACKEND_ID);
    printf("base=0x%08" PRIx64 "\n", base);
    printf("backend_id=0x%08" PRIx32 "\n", backend);
    printf("backend=%s\n", backend_name(backend));
    printf("firmware_entry=0x%08" PRIx32 "\n",
           read_reg(regs, base, FIRMWARE_ENTRY));
    printf("shared_base=0x%08" PRIx32 "\n",
           read_reg(regs, base, SHARED_BASE));
    printf("shared_size=0x%08" PRIx32 "\n",
           read_reg(regs, base, SHARED_SIZE));
    printf("dma_requests=%" PRIu32 "\n",
           read_reg(regs, base, DMA_REQUESTS));
    printf("dma_completions=%" PRIu32 "\n",
           read_reg(regs, base, DMA_COMPLETIONS));
    printf("dma_errors=%" PRIu32 "\n",
           read_reg(regs, base, DMA_ERRORS));
    printf("dma_state=0x%08" PRIx32 "\n",
           read_reg(regs, base, DMA_STATE));
    printf("reset_control=0x%08" PRIx32 "\n",
           read_reg(regs, base, RESET_CONTROL));
    printf("status=0x%08" PRIx32 "\n", read_reg(regs, base, STATUS));
}

static int
print_mem_info(struct coral_regs *regs, uint64_t base)
{
    struct coral_shared_window window;
    if (open_shared_window(regs, base, 0, &window) != 0) {
        return 1;
    }
    printf("shared_base=0x%08" PRIx32 "\n", window.base);
    printf("shared_size=0x%08" PRIx32 "\n", window.size);
    printf("shared_words=%" PRIu32 "\n", window.size / 4);
    close_shared_window(&window);
    return 0;
}

static int
clear_shared_window(struct coral_regs *regs, uint64_t base)
{
    struct coral_shared_window window;
    if (open_shared_window(regs, base, 0, &window) != 0) {
        return 1;
    }
    memset((void *)window.bytes, 0, window.size);
    __sync_synchronize();
    printf("shared_clear=PASS\n");
    close_shared_window(&window);
    return 0;
}

static int
read_shared_u32(struct coral_regs *regs, uint64_t base, uint64_t offset)
{
    struct coral_shared_window window;
    if (open_shared_window(regs, base, sizeof(uint32_t), &window) != 0) {
        return 1;
    }
    if (check_shared_u32_access(&window, offset) != 0) {
        close_shared_window(&window);
        return 1;
    }

    volatile uint32_t *word =
        (volatile uint32_t *)(window.bytes + offset);
    printf("shared[0x%08" PRIx64 "]=0x%08" PRIx32 "\n",
           offset, *word);
    close_shared_window(&window);
    return 0;
}

static int
write_shared_u32(struct coral_regs *regs, uint64_t base, uint64_t offset,
                 uint32_t value)
{
    struct coral_shared_window window;
    if (open_shared_window(regs, base, sizeof(uint32_t), &window) != 0) {
        return 1;
    }
    if (check_shared_u32_access(&window, offset) != 0) {
        close_shared_window(&window);
        return 1;
    }

    volatile uint32_t *word =
        (volatile uint32_t *)(window.bytes + offset);
    *word = value;
    __sync_synchronize();
    printf("shared[0x%08" PRIx64 "]=0x%08" PRIx32 "\n",
           offset, value);
    close_shared_window(&window);
    return 0;
}

static int
run_npu(struct coral_regs *regs, uint64_t base, uint32_t entry,
        uint64_t polls)
{
    printf("backend=%s\n",
           backend_name(read_reg(regs, base, BACKEND_ID)));
    printf("entry=0x%08" PRIx32 "\n", entry);
    write_reg(regs, base, PC_START, entry);
    write_reg(regs, base, RESET_CONTROL, 1);
    write_reg(regs, base, RESET_CONTROL, 0);

    uint32_t status = 0;
    for (uint64_t i = 0; i < polls; ++i) {
        status = read_reg(regs, base, STATUS);
        if ((status & 0x3) != 0) {
            break;
        }
    }
    printf("status=0x%08" PRIx32 "\n", status);

    if ((status & 0x2) != 0) {
        fprintf(stderr, "Coral NPU reported an execution fault\n");
        return 1;
    }
    if ((status & 0x1) == 0) {
        fprintf(stderr, "Coral NPU did not halt within the poll limit\n");
        return 1;
    }
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

    uint64_t base = DEFAULT_BASE;
    uint64_t shared_offset = 0;
    uint64_t shared_value = 0;
    if (command_mem_read32 || command_mem_write32) {
        const int base_arg = command_mem_read32 ? 3 : 4;
        if (parse_u64(argv[2], &shared_offset) != 0) {
            fprintf(stderr, "invalid shared offset: %s\n", argv[2]);
            return 2;
        }
        if (command_mem_write32 &&
            (parse_u64(argv[3], &shared_value) != 0 ||
             shared_value > UINT32_MAX)) {
            fprintf(stderr, "invalid 32-bit value: %s\n", argv[3]);
            return 2;
        }
        if (argc > base_arg && parse_u64(argv[base_arg], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[base_arg]);
            return 2;
        }
    } else {
        if (argc >= 3 && parse_u64(argv[2], &base) != 0) {
            fprintf(stderr, "invalid base address: %s\n", argv[2]);
            return 2;
        }
    }

    struct coral_regs regs;
    if (open_regs(&regs, base) != 0) {
        return 1;
    }

    if (command_info) {
        print_info(&regs, base);
        close_regs(&regs);
        return 0;
    }

    if (command_mem_info) {
        const int result = print_mem_info(&regs, base);
        close_regs(&regs);
        return result;
    }

    if (command_mem_clear) {
        const int result = clear_shared_window(&regs, base);
        close_regs(&regs);
        return result;
    }

    if (command_mem_read32) {
        const int result = read_shared_u32(&regs, base, shared_offset);
        close_regs(&regs);
        return result;
    }

    if (command_mem_write32) {
        const int result =
            write_shared_u32(&regs, base, shared_offset,
                             (uint32_t)shared_value);
        close_regs(&regs);
        return result;
    }

    uint64_t entry = read_reg(&regs, base, FIRMWARE_ENTRY);
    uint64_t polls = command_dma_test ? 100000 : 1000;
    if (command_run && argc >= 4 && parse_u64(argv[3], &entry) != 0) {
        fprintf(stderr, "invalid entry address: %s\n", argv[3]);
        close_regs(&regs);
        return 2;
    }
    const int poll_arg = command_dma_test ? 3 : 4;
    if (argc > poll_arg &&
        (parse_u64(argv[poll_arg], &polls) != 0 || polls == 0)) {
        fprintf(stderr, "invalid poll count: %s\n", argv[poll_arg]);
        close_regs(&regs);
        return 2;
    }

    if (command_run) {
        const int result = run_npu(&regs, base, (uint32_t)entry, polls);
        close_regs(&regs);
        return result;
    }

    struct coral_shared_window window;
    if (open_shared_window(&regs, base, 16, &window) != 0) {
        close_regs(&regs);
        return 1;
    }

    volatile uint32_t *shared =
        (volatile uint32_t *)window.bytes;
    shared[0] = 7;
    shared[1] = 35;
    shared[2] = 0;
    __sync_synchronize();

    const int run_result =
        run_npu(&regs, base, (uint32_t)entry, polls);
    __sync_synchronize();
    printf("dma_result=%" PRIu32 "\n", shared[0]);
    printf("dma_magic=0x%08" PRIx32 "\n", shared[2]);
    printf("dma_requests=%" PRIu32 "\n",
           read_reg(&regs, base, DMA_REQUESTS));
    printf("dma_completions=%" PRIu32 "\n",
           read_reg(&regs, base, DMA_COMPLETIONS));
    printf("dma_errors=%" PRIu32 "\n",
           read_reg(&regs, base, DMA_ERRORS));
    printf("dma_state=0x%08" PRIx32 "\n",
           read_reg(&regs, base, DMA_STATE));
    const int valid = shared[0] == 42 &&
                      shared[2] == UINT32_C(0x4e505544);
    close_shared_window(&window);
    close_regs(&regs);

    if (run_result != 0 || !valid) {
        fprintf(stderr, "Coral coherent DMA smoke failed\n");
        return 1;
    }
    printf("dma_test=PASS\n");
    return 0;
}
