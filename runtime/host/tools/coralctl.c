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

static void
usage(const char *prog)
{
    fprintf(stderr,
            "usage:\n"
            "  %s info [base]\n"
            "  %s run [base [entry [poll-count]]]\n",
            prog, prog);
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
    printf("reset_control=0x%08" PRIx32 "\n",
           read_reg(regs, base, RESET_CONTROL));
    printf("status=0x%08" PRIx32 "\n", read_reg(regs, base, STATUS));
}

int
main(int argc, char **argv)
{
    if (argc < 2 || argc > 5) {
        usage(argv[0]);
        return 2;
    }

    const int command_info = strcmp(argv[1], "info") == 0;
    const int command_run = strcmp(argv[1], "run") == 0;
    if (!command_info && !command_run) {
        usage(argv[0]);
        return 2;
    }
    if ((command_info && argc > 3) || (command_run && argc > 5)) {
        usage(argv[0]);
        return 2;
    }

    uint64_t base = DEFAULT_BASE;
    if (argc >= 3 && parse_u64(argv[2], &base) != 0) {
        fprintf(stderr, "invalid base address: %s\n", argv[2]);
        return 2;
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

    uint64_t entry = read_reg(&regs, base, FIRMWARE_ENTRY);
    uint64_t polls = 1000;
    if (argc >= 4 && parse_u64(argv[3], &entry) != 0) {
        fprintf(stderr, "invalid entry address: %s\n", argv[3]);
        close_regs(&regs);
        return 2;
    }
    if (argc >= 5 && (parse_u64(argv[4], &polls) != 0 || polls == 0)) {
        fprintf(stderr, "invalid poll count: %s\n", argv[4]);
        close_regs(&regs);
        return 2;
    }

    printf("backend=%s\n",
           backend_name(read_reg(&regs, base, BACKEND_ID)));
    printf("entry=0x%08" PRIx64 "\n", entry);
    write_reg(&regs, base, PC_START, (uint32_t)entry);
    write_reg(&regs, base, RESET_CONTROL, 1);
    write_reg(&regs, base, RESET_CONTROL, 0);

    uint32_t status = 0;
    for (uint64_t i = 0; i < polls; ++i) {
        status = read_reg(&regs, base, STATUS);
        if ((status & 0x3) != 0) {
            break;
        }
    }
    printf("status=0x%08" PRIx32 "\n", status);
    close_regs(&regs);

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
