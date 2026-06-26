#include "opennpux/coral_runtime.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

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

const char *
opennpux_coral_backend_name(enum opennpux_coral_backend backend)
{
    switch (backend) {
      case OPENNPUX_CORAL_BACKEND_STAGE_A:
        return "stage-a";
      case OPENNPUX_CORAL_BACKEND_VERILATED:
        return "verilated-coral";
      case OPENNPUX_CORAL_BACKEND_UNKNOWN:
      default:
        return "unknown";
    }
}

enum opennpux_coral_backend
opennpux_coral_decode_backend(uint32_t id)
{
    switch (id) {
      case BACKEND_STAGE_A:
        return OPENNPUX_CORAL_BACKEND_STAGE_A;
      case BACKEND_VERILATED:
        return OPENNPUX_CORAL_BACKEND_VERILATED;
      default:
        return OPENNPUX_CORAL_BACKEND_UNKNOWN;
    }
}

int
opennpux_coral_parse_u64(const char *text, uint64_t *value)
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

int
opennpux_coral_check_shared_u32_access(uint32_t size, uint64_t offset)
{
    if ((offset & 0x3) != 0) {
        errno = EINVAL;
        return -1;
    }
    if (offset > UINT32_MAX || offset + sizeof(uint32_t) > size) {
        errno = ERANGE;
        return -1;
    }
    return 0;
}

int
opennpux_coral_open(struct opennpux_coral_device *dev, uint64_t base)
{
    memset(dev, 0, sizeof(*dev));
    dev->fd = -1;
    dev->base = base;

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        perror("sysconf(_SC_PAGESIZE)");
        return -1;
    }

    dev->page_size = (size_t)page_size;
    const uint64_t csr_addr = base + RESET_CONTROL;
    const uint64_t page_mask = (uint64_t)page_size - 1;
    dev->page_base = csr_addr & ~page_mask;
    dev->fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (dev->fd < 0) {
        perror("open(/dev/mem)");
        return -1;
    }

    dev->mapping = mmap(NULL, dev->page_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED, dev->fd, (off_t)dev->page_base);
    if (dev->mapping == MAP_FAILED) {
        dev->mapping = NULL;
        perror("mmap(/dev/mem)");
        close(dev->fd);
        dev->fd = -1;
        return -1;
    }
    return 0;
}

void
opennpux_coral_close(struct opennpux_coral_device *dev)
{
    if (dev->mapping != NULL) {
        munmap(dev->mapping, dev->page_size);
    }
    if (dev->fd >= 0) {
        close(dev->fd);
    }
    memset(dev, 0, sizeof(*dev));
    dev->fd = -1;
}

static volatile uint32_t *
reg_ptr(struct opennpux_coral_device *dev, uint64_t offset)
{
    const uint64_t addr = dev->base + offset;
    if (addr < dev->page_base ||
        addr + sizeof(uint32_t) > dev->page_base + dev->page_size) {
        return NULL;
    }
    return (volatile uint32_t *)((uint8_t *)dev->mapping +
                                 (addr - dev->page_base));
}

uint32_t
opennpux_coral_read_reg(struct opennpux_coral_device *dev, uint64_t offset)
{
    volatile uint32_t *reg = reg_ptr(dev, offset);
    if (reg == NULL) {
        fprintf(stderr, "register offset 0x%" PRIx64 " is not mapped\n",
                offset);
        exit(1);
    }
    return *reg;
}

void
opennpux_coral_write_reg(struct opennpux_coral_device *dev, uint64_t offset,
                         uint32_t value)
{
    volatile uint32_t *reg = reg_ptr(dev, offset);
    if (reg == NULL) {
        fprintf(stderr, "register offset 0x%" PRIx64 " is not mapped\n",
                offset);
        exit(1);
    }
    *reg = value;
    __sync_synchronize();
}

void
opennpux_coral_get_info(struct opennpux_coral_device *dev,
                        struct opennpux_coral_info *info)
{
    memset(info, 0, sizeof(*info));
    info->base = dev->base;
    info->backend_id = opennpux_coral_read_reg(dev, BACKEND_ID);
    info->backend = opennpux_coral_decode_backend(info->backend_id);
    info->firmware_entry = opennpux_coral_read_reg(dev, FIRMWARE_ENTRY);
    info->shared_base = opennpux_coral_read_reg(dev, SHARED_BASE);
    info->shared_size = opennpux_coral_read_reg(dev, SHARED_SIZE);
    info->dma_requests = opennpux_coral_read_reg(dev, DMA_REQUESTS);
    info->dma_completions = opennpux_coral_read_reg(dev, DMA_COMPLETIONS);
    info->dma_errors = opennpux_coral_read_reg(dev, DMA_ERRORS);
    info->dma_state = opennpux_coral_read_reg(dev, DMA_STATE);
    info->reset_control = opennpux_coral_read_reg(dev, RESET_CONTROL);
    info->status = opennpux_coral_read_reg(dev, STATUS);
}

int
opennpux_coral_open_shared_window(
    struct opennpux_coral_device *dev, size_t min_size,
    struct opennpux_coral_shared_window *window)
{
    memset(window, 0, sizeof(*window));
    window->base = opennpux_coral_read_reg(dev, SHARED_BASE);
    window->size = opennpux_coral_read_reg(dev, SHARED_SIZE);
    if (window->size < min_size) {
        fprintf(stderr,
                "Coral shared DMA window is too small: size=0x%08" PRIx32
                " required=0x%zx\n",
                window->size, min_size);
        return -1;
    }

    const uint64_t page_mask = (uint64_t)dev->page_size - 1;
    const uint64_t map_base = window->base & ~page_mask;
    const size_t map_offset = window->base & page_mask;
    window->map_size = map_offset + window->size;
    window->mapping = mmap(NULL, window->map_size, PROT_READ | PROT_WRITE,
                           MAP_SHARED, dev->fd, (off_t)map_base);
    if (window->mapping == MAP_FAILED) {
        window->mapping = NULL;
        perror("mmap(Coral shared DMA window)");
        return -1;
    }
    window->bytes = (volatile uint8_t *)window->mapping + map_offset;
    return 0;
}

void
opennpux_coral_close_shared_window(
    struct opennpux_coral_shared_window *window)
{
    if (window->mapping != NULL) {
        munmap(window->mapping, window->map_size);
    }
    memset(window, 0, sizeof(*window));
}

int
opennpux_coral_clear_shared_window(struct opennpux_coral_device *dev)
{
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, 0, &window) != 0) {
        return -1;
    }
    memset((void *)window.bytes, 0, window.size);
    __sync_synchronize();
    opennpux_coral_close_shared_window(&window);
    return 0;
}

int
opennpux_coral_read_shared_u32(struct opennpux_coral_device *dev,
                               uint64_t offset, uint32_t *value)
{
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, sizeof(uint32_t), &window) !=
        0) {
        return -1;
    }
    if (opennpux_coral_check_shared_u32_access(window.size, offset) != 0) {
        opennpux_coral_close_shared_window(&window);
        return -1;
    }
    volatile uint32_t *word = (volatile uint32_t *)(window.bytes + offset);
    *value = *word;
    opennpux_coral_close_shared_window(&window);
    return 0;
}

int
opennpux_coral_write_shared_u32(struct opennpux_coral_device *dev,
                                uint64_t offset, uint32_t value)
{
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, sizeof(uint32_t), &window) !=
        0) {
        return -1;
    }
    if (opennpux_coral_check_shared_u32_access(window.size, offset) != 0) {
        opennpux_coral_close_shared_window(&window);
        return -1;
    }
    volatile uint32_t *word = (volatile uint32_t *)(window.bytes + offset);
    *word = value;
    __sync_synchronize();
    opennpux_coral_close_shared_window(&window);
    return 0;
}

int
opennpux_coral_run(struct opennpux_coral_device *dev, uint32_t entry,
                   uint64_t polls, uint32_t *final_status)
{
    opennpux_coral_write_reg(dev, PC_START, entry);
    opennpux_coral_write_reg(dev, RESET_CONTROL, 1);
    opennpux_coral_write_reg(dev, RESET_CONTROL, 0);

    uint32_t status = 0;
    for (uint64_t i = 0; i < polls; ++i) {
        status = opennpux_coral_read_reg(dev, STATUS);
        if ((status & 0x3) != 0) {
            break;
        }
    }
    if (final_status != NULL) {
        *final_status = status;
    }

    if ((status & 0x2) != 0) {
        errno = EIO;
        return -1;
    }
    if ((status & 0x1) == 0) {
        errno = ETIMEDOUT;
        return -1;
    }
    return 0;
}

int
opennpux_coral_dma_test(struct opennpux_coral_device *dev, uint32_t entry,
                        uint64_t polls,
                        struct opennpux_coral_dma_test_result *result)
{
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, 16, &window) != 0) {
        return -1;
    }

    volatile uint32_t *shared = (volatile uint32_t *)window.bytes;
    shared[0] = 7;
    shared[1] = 35;
    shared[2] = 0;
    __sync_synchronize();

    uint32_t status = 0;
    const int run_result = opennpux_coral_run(dev, entry, polls, &status);
    __sync_synchronize();

    memset(result, 0, sizeof(*result));
    result->status = status;
    result->result = shared[0];
    result->magic = shared[2];
    result->requests = opennpux_coral_read_reg(dev, DMA_REQUESTS);
    result->completions = opennpux_coral_read_reg(dev, DMA_COMPLETIONS);
    result->errors = opennpux_coral_read_reg(dev, DMA_ERRORS);
    result->state = opennpux_coral_read_reg(dev, DMA_STATE);

    const int run_errno = errno;
    const int valid = result->result == 42 &&
                      result->magic == OPENNPUX_CORAL_DMA_MAGIC;
    opennpux_coral_close_shared_window(&window);
    if (run_result != 0) {
        errno = run_errno;
        return -1;
    }
    if (!valid) {
        errno = EIO;
        return -1;
    }
    return 0;
}
