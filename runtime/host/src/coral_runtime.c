#define _POSIX_C_SOURCE 200809L

#include "opennpux/coral_runtime.h"
#include "opennpux/npu_xgraph_lowering.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
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
#define RUNTIME_SYNC_OFFSET UINT64_C(0x30fd0)
#define RUNTIME_SYNC_SIZE UINT64_C(0x30fd4)
#define RUNTIME_SYNC_CONTROL UINT64_C(0x30fd8)
#define RUNTIME_SYNC_STATUS UINT64_C(0x30fdc)
#define FIRMWARE_ENTRY UINT64_C(0x30ff8)
#define BACKEND_ID UINT64_C(0x30ffc)

#define BACKEND_STAGE_A UINT32_C(0x4e505501)
#define BACKEND_VERILATED UINT32_C(0x4e505502)
#define CORAL_LOCAL_EXTMEM_BASE UINT32_C(0x20000000)

static void
initialize_xgraph_request(
    struct opennpux_npu_functional_request *request,
    struct opennpux_npu_operator_parameters *parameters, uint32_t command_id,
    uint32_t opcode, uint32_t rows, uint32_t features)
{
    memset(request, 0, sizeof(*request));
    request->magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
    request->version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
    request->struct_size = sizeof(*request);
    request->opcode = opcode;
    request->command_id = command_id;
    request->rows = rows;
    request->features = features;

    memset(parameters, 0, sizeof(*parameters));
    parameters->magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
    parameters->version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
    parameters->struct_size = sizeof(*parameters);
    parameters->opcode = opcode;
}

static int
add_xgraph_operand(struct opennpux_npu_functional_request *request,
                   uint32_t role, uint32_t offset, uint32_t byte_size)
{
    if (request->operand_count >= OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS ||
        offset > UINT32_MAX - CORAL_LOCAL_EXTMEM_BASE) {
        errno = EOVERFLOW;
        return -1;
    }
    struct opennpux_npu_functional_operand *operand =
        &request->operands[request->operand_count++];
    operand->role = role;
    operand->address = CORAL_LOCAL_EXTMEM_BASE + offset;
    operand->byte_size = byte_size;
    operand->reserved = 0;
    return 0;
}

const char *
opennpux_coral_transport_name(enum opennpux_coral_transport transport)
{
    switch (transport) {
      case OPENNPUX_CORAL_TRANSPORT_DEVMEM:
        return "devmem";
      case OPENNPUX_CORAL_TRANSPORT_DRIVER:
        return "driver";
      case OPENNPUX_CORAL_TRANSPORT_UNKNOWN:
      default:
        return "unknown";
    }
}

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

    const char *forced = getenv("OPENNPUX_CORAL_TRANSPORT");
    const int force_devmem = forced != NULL && strcmp(forced, "devmem") == 0;
    const int force_driver = forced != NULL && strcmp(forced, "driver") == 0;
    const char *driver_path = getenv("OPENNPUX_CORAL_DEVICE");
    if (driver_path == NULL || driver_path[0] == '\0') {
        driver_path = OPENNPUX_CORAL_DEVICE_PATH;
    }

    if (!force_devmem) {
        dev->fd = open(driver_path, O_RDWR);
        if (dev->fd >= 0) {
            struct opennpux_coral_ioc_caps caps;
            memset(&caps, 0, sizeof(caps));
            if (ioctl(dev->fd, OPENNPUX_CORAL_IOC_GET_CAPS, &caps) == 0) {
                if (caps.abi_version != OPENNPUX_CORAL_ABI_VERSION) {
                    fprintf(stderr,
                            "unsupported Coral driver ABI version: %" PRIu32
                            "\n",
                            caps.abi_version);
                    close(dev->fd);
                    dev->fd = -1;
                    errno = EPROTO;
                    return -1;
                }
                dev->abi_version = caps.abi_version;
                dev->features = caps.features;
            } else if (errno != ENOTTY) {
                perror("ioctl(OPENNPUX_CORAL_IOC_GET_CAPS)");
                close(dev->fd);
                dev->fd = -1;
                return -1;
            }
            dev->transport = OPENNPUX_CORAL_TRANSPORT_DRIVER;
            return 0;
        }
        if (force_driver) {
            perror("open(OPENNPUX_CORAL_DEVICE)");
            return -1;
        }
    }

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
    dev->transport = OPENNPUX_CORAL_TRANSPORT_DEVMEM;
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
    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER) {
        struct opennpux_coral_info info;
        opennpux_coral_get_info(dev, &info);
        switch (offset) {
          case RESET_CONTROL:
            return info.reset_control;
          case STATUS:
            return info.status;
          case DMA_ERRORS:
            return info.dma_errors;
          case DMA_REQUESTS:
            return info.dma_requests;
          case DMA_COMPLETIONS:
            return info.dma_completions;
          case DMA_STATE:
            return info.dma_state;
          case SHARED_BASE:
            return info.shared_base;
          case SHARED_SIZE:
            return info.shared_size;
          case FIRMWARE_ENTRY:
            return info.firmware_entry;
          case BACKEND_ID:
            return info.backend_id;
          default:
            fprintf(stderr, "driver cannot read raw register offset 0x%"
                    PRIx64 "\n", offset);
            exit(1);
        }
    }

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
    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER) {
        fprintf(stderr, "driver backend does not allow raw register writes "
                "offset=0x%" PRIx64 " value=0x%08" PRIx32 "\n",
                offset, value);
        exit(1);
    }

    volatile uint32_t *reg = reg_ptr(dev, offset);
    if (reg == NULL) {
        fprintf(stderr, "register offset 0x%" PRIx64 " is not mapped\n",
                offset);
        exit(1);
    }
    *reg = value;
    __sync_synchronize();
}

int
opennpux_coral_sync_shared_to_extmem(
    struct opennpux_coral_device *dev, uint32_t offset, uint32_t size)
{
    if (dev == NULL || size == 0 ||
        dev->transport != OPENNPUX_CORAL_TRANSPORT_DEVMEM) {
        errno = EOPNOTSUPP;
        return -1;
    }
    opennpux_coral_write_reg(dev, RUNTIME_SYNC_OFFSET, offset);
    opennpux_coral_write_reg(dev, RUNTIME_SYNC_SIZE, size);
    opennpux_coral_write_reg(dev, RUNTIME_SYNC_CONTROL, 1);
    const uint32_t status = opennpux_coral_read_reg(
        dev, RUNTIME_SYNC_STATUS);
    if (status != 1) {
        errno = status == 2 ? EINVAL : EIO;
        return -1;
    }
    return 0;
}

int
opennpux_coral_sync_extmem_to_shared(
    struct opennpux_coral_device *dev, uint32_t offset, uint32_t size)
{
    if (dev == NULL || size == 0 ||
        dev->transport != OPENNPUX_CORAL_TRANSPORT_DEVMEM) {
        errno = EOPNOTSUPP;
        return -1;
    }
    opennpux_coral_write_reg(dev, RUNTIME_SYNC_OFFSET, offset);
    opennpux_coral_write_reg(dev, RUNTIME_SYNC_SIZE, size);
    opennpux_coral_write_reg(dev, RUNTIME_SYNC_CONTROL, 2);
    const uint32_t status = opennpux_coral_read_reg(
        dev, RUNTIME_SYNC_STATUS);
    if (status != 1) {
        errno = status == 2 ? EINVAL : EIO;
        return -1;
    }
    return 0;
}

void
opennpux_coral_get_info(struct opennpux_coral_device *dev,
                        struct opennpux_coral_info *info)
{
    memset(info, 0, sizeof(*info));
    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER) {
        struct opennpux_coral_ioc_info ioc_info;
        memset(&ioc_info, 0, sizeof(ioc_info));
        if (ioctl(dev->fd, OPENNPUX_CORAL_IOC_GET_INFO, &ioc_info) != 0) {
            perror("ioctl(OPENNPUX_CORAL_IOC_GET_INFO)");
            exit(1);
        }
        info->base = ioc_info.base;
        info->backend_id = ioc_info.backend_id;
        info->backend = opennpux_coral_decode_backend(info->backend_id);
        info->firmware_entry = ioc_info.firmware_entry;
        info->shared_base = ioc_info.shared_base;
        info->shared_size = ioc_info.shared_size;
        info->dma_requests = ioc_info.dma_requests;
        info->dma_completions = ioc_info.dma_completions;
        info->dma_errors = ioc_info.dma_errors;
        info->dma_state = ioc_info.dma_state;
        info->reset_control = ioc_info.reset_control;
        info->status = ioc_info.status;
        info->abi_version = dev->abi_version;
        info->features = dev->features;
        return;
    }

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
    info->abi_version = 0;
    info->features = 0;
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

    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER &&
        (dev->features & OPENNPUX_CORAL_FEATURE_SHARED_MMAP) == 0) {
        fprintf(stderr, "Coral driver does not support shared-window mmap\n");
        errno = ENOTSUP;
        return -1;
    }

    const uint64_t page_mask = (uint64_t)dev->page_size - 1;
    const uint64_t map_base = dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER ?
        0 : (window->base & ~page_mask);
    const size_t map_offset = dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER ?
        0 : (window->base & page_mask);
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

static void
zero_volatile_bytes(volatile uint8_t *destination, size_t size)
{
    for (size_t offset = 0; offset < size; ++offset) {
        destination[offset] = 0;
    }
}

static void
copy_to_volatile_bytes(volatile uint8_t *destination, const void *source,
                       size_t size)
{
    const uint8_t *source_bytes = (const uint8_t *)source;
    for (size_t offset = 0; offset < size; ++offset) {
        destination[offset] = source_bytes[offset];
    }
}

static void
copy_from_volatile_bytes(void *destination, const volatile uint8_t *source,
                         size_t size)
{
    uint8_t *destination_bytes = (uint8_t *)destination;
    for (size_t offset = 0; offset < size; ++offset) {
        destination_bytes[offset] = source[offset];
    }
}

static uint32_t
checksum_bytes(const void *data, size_t size)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t checksum = UINT32_C(2166136261);
    for (size_t offset = 0; offset < size; ++offset) {
        checksum ^= bytes[offset];
        checksum *= UINT32_C(16777619);
    }
    return checksum;
}

static int
compare_floats(const float *actual, const float *expected, size_t count,
               float *max_abs_error)
{
    float maximum = 0.0f;
    int valid = 1;
    for (size_t index = 0; index < count; ++index) {
        const float error = fabsf(actual[index] - expected[index]);
        const float tolerance = 1.0e-5f + 1.0e-4f * fabsf(expected[index]);
        if (!isfinite(actual[index]) || error > tolerance) {
            valid = 0;
        }
        if (error > maximum || !isfinite(error)) {
            maximum = error;
        }
    }
    *max_abs_error = maximum;
    return valid;
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
opennpux_coral_start(struct opennpux_coral_device *dev, uint32_t entry)
{
    if (dev == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER) {
        if ((dev->features & OPENNPUX_CORAL_FEATURE_ASYNC_START) == 0) {
            errno = ENOTSUP;
            return -1;
        }
        const struct opennpux_coral_ioc_start start = {
            .entry = entry,
            .flags = 0,
        };
        return ioctl(dev->fd, OPENNPUX_CORAL_IOC_START, &start);
    }
    if (dev->transport != OPENNPUX_CORAL_TRANSPORT_DEVMEM) {
        errno = EINVAL;
        return -1;
    }
    opennpux_coral_write_reg(dev, PC_START, entry);
    opennpux_coral_write_reg(dev, RESET_CONTROL, 1);
    opennpux_coral_write_reg(dev, RESET_CONTROL, 0);
    return 0;
}

int
opennpux_coral_status(struct opennpux_coral_device *dev, uint32_t *status)
{
    if (dev == NULL || status == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER) {
        struct opennpux_coral_info info;
        opennpux_coral_get_info(dev, &info);
        *status = info.status;
        return 0;
    }
    if (dev->transport != OPENNPUX_CORAL_TRANSPORT_DEVMEM) {
        errno = EINVAL;
        return -1;
    }
    *status = opennpux_coral_read_reg(dev, STATUS);
    return 0;
}

int
opennpux_coral_reset(struct opennpux_coral_device *dev)
{
    if (dev == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER) {
        if ((dev->features & OPENNPUX_CORAL_FEATURE_RESET) == 0) {
            errno = ENOTSUP;
            return -1;
        }
        return ioctl(dev->fd, OPENNPUX_CORAL_IOC_RESET);
    }
    if (dev->transport != OPENNPUX_CORAL_TRANSPORT_DEVMEM) {
        errno = EINVAL;
        return -1;
    }
    opennpux_coral_write_reg(dev, RESET_CONTROL, 1);
    return 0;
}

int
opennpux_coral_run_with_service(
    struct opennpux_coral_device *dev, uint32_t entry, uint64_t polls,
    opennpux_coral_service_callback service, void *opaque,
    uint32_t *final_status)
{
    if (polls == 0 || opennpux_coral_start(dev, entry) != 0) {
        if (polls == 0) {
            errno = EINVAL;
        }
        return -1;
    }
    uint32_t status = 0;
    for (uint64_t index = 0; index < polls; ++index) {
        if (opennpux_coral_status(dev, &status) != 0) {
            goto fail;
        }
        if ((status & 0x3) != 0) {
            break;
        }
        if (service != NULL && service(opaque) < 0) {
            goto fail;
        }
    }
    if (final_status != NULL) {
        *final_status = status;
    }
    if ((status & 0x2) != 0) {
        errno = EIO;
        goto fail;
    }
    if ((status & 0x1) == 0) {
        errno = ETIMEDOUT;
        goto fail;
    }
    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER &&
        opennpux_coral_reset(dev) != 0) {
        return -1;
    }
    return 0;

fail:
    {
        const int error = errno == 0 ? EIO : errno;
        (void)opennpux_coral_reset(dev);
        errno = error;
    }
    return -1;
}

int
opennpux_coral_run(struct opennpux_coral_device *dev, uint32_t entry,
                   uint64_t polls, uint32_t *final_status)
{
    if (dev->transport == OPENNPUX_CORAL_TRANSPORT_DRIVER) {
        const uint32_t async_features =
            OPENNPUX_CORAL_FEATURE_ASYNC_START |
            OPENNPUX_CORAL_FEATURE_POLL_COMPLETION;
        if ((dev->features & async_features) == async_features) {
            struct opennpux_coral_ioc_start start;
            memset(&start, 0, sizeof(start));
            start.entry = entry;
            if (ioctl(dev->fd, OPENNPUX_CORAL_IOC_START, &start) != 0) {
                return -1;
            }

            struct pollfd pfd = {
                .fd = dev->fd,
                .events = POLLIN,
            };
            const int timeout = polls > INT_MAX ? INT_MAX : (int)polls;
            int poll_result;
            do {
                poll_result = poll(&pfd, 1, timeout);
            } while (poll_result < 0 && errno == EINTR);

            struct opennpux_coral_info info;
            opennpux_coral_get_info(dev, &info);
            if (final_status != NULL) {
                *final_status = info.status;
            }
            if (poll_result < 0) {
                ioctl(dev->fd, OPENNPUX_CORAL_IOC_RESET);
                return -1;
            }
            if (poll_result == 0) {
                ioctl(dev->fd, OPENNPUX_CORAL_IOC_RESET);
                errno = ETIMEDOUT;
                return -1;
            }
            if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0 ||
                (info.status & 0x2) != 0) {
                ioctl(dev->fd, OPENNPUX_CORAL_IOC_RESET);
                errno = EIO;
                return -1;
            }
            if ((info.status & 0x1) == 0) {
                ioctl(dev->fd, OPENNPUX_CORAL_IOC_RESET);
                errno = EIO;
                return -1;
            }
            if (ioctl(dev->fd, OPENNPUX_CORAL_IOC_RESET) != 0) {
                return -1;
            }
            return 0;
        }

        struct opennpux_coral_ioc_run run;
        memset(&run, 0, sizeof(run));
        run.entry = entry;
        run.polls = polls;
        if (ioctl(dev->fd, OPENNPUX_CORAL_IOC_RUN, &run) != 0) {
            return -1;
        }
        if (final_status != NULL) {
            *final_status = run.status;
        }
        if ((run.status & 0x2) != 0) {
            errno = EIO;
            return -1;
        }
        if ((run.status & 0x1) == 0) {
            errno = ETIMEDOUT;
            return -1;
        }
        return 0;
    }

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

static int
submit_vector_add(
    struct opennpux_coral_device *dev, uint32_t entry, uint32_t opcode,
    const uint32_t *host_input0, const uint32_t *host_input1,
    uint32_t element_count, uint32_t expected_checksum, uint64_t polls,
    struct opennpux_coral_vector_add_result *result)
{
    memset(result, 0, sizeof(*result));
    result->element_count = element_count;
    result->opcode = opcode;
    if (element_count == 0 ||
        element_count > OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS ||
        (opcode != OPENNPUX_CORAL_OPCODE_VECTOR_ADD_U32 &&
         opcode != OPENNPUX_CORAL_OPCODE_VECTOR_ADD_CUSTOM_U32)) {
        errno = EINVAL;
        return -1;
    }

    const size_t required = OPENNPUX_CORAL_OUTPUT_OFFSET +
                            element_count * sizeof(uint32_t);
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, required, &window) != 0) {
        return -1;
    }

    volatile struct opennpux_coral_command *command =
        (volatile struct opennpux_coral_command *)(
            window.bytes + OPENNPUX_CORAL_COMMAND_OFFSET);
    volatile uint32_t *input0 = (volatile uint32_t *)(
        window.bytes + OPENNPUX_CORAL_INPUT0_OFFSET);
    volatile uint32_t *input1 = (volatile uint32_t *)(
        window.bytes + OPENNPUX_CORAL_INPUT1_OFFSET);
    volatile uint32_t *output = (volatile uint32_t *)(
        window.bytes + OPENNPUX_CORAL_OUTPUT_OFFSET);

    memset((void *)command, 0, sizeof(*command));
    for (uint32_t i = 0; i < element_count; ++i) {
        input0[i] = host_input0[i];
        input1[i] = host_input1[i];
        output[i] = 0;
    }

    command->magic = OPENNPUX_CORAL_COMMAND_MAGIC;
    command->abi_version = OPENNPUX_CORAL_COMMAND_ABI_VERSION;
    command->struct_size = sizeof(*command);
    command->opcode = opcode;
    command->sequence = 1;
    command->element_count = element_count;
    command->input0_offset = OPENNPUX_CORAL_INPUT0_OFFSET;
    command->input1_offset = OPENNPUX_CORAL_INPUT1_OFFSET;
    command->output_offset = OPENNPUX_CORAL_OUTPUT_OFFSET;
    command->output_size = element_count * sizeof(uint32_t);
    command->status = OPENNPUX_CORAL_COMMAND_PENDING;
    __sync_synchronize();

    uint32_t device_status = 0;
    const int run_result =
        opennpux_coral_run(dev, entry, polls, &device_status);
    const int run_errno = errno;
    __sync_synchronize();

    result->status = command->status;
    result->error_code = command->error_code;
    result->completed_elements = command->completed_elements;
    result->accelerator_cycles = command->accelerator_cycles;

    int valid = result->status == OPENNPUX_CORAL_COMMAND_COMPLETE &&
                result->error_code == OPENNPUX_CORAL_COMMAND_ERROR_NONE &&
                result->completed_elements == element_count;
    uint32_t checksum = 0;
    for (uint32_t i = 0; i < element_count; ++i) {
        const uint32_t expected = host_input0[i] + host_input1[i];
        checksum += output[i];
        if (output[i] != expected) {
            valid = 0;
        }
    }
    result->checksum = checksum;
    if (checksum != expected_checksum) {
        valid = 0;
    }
    opennpux_coral_close_shared_window(&window);

    if (run_result != 0) {
        errno = run_errno;
        return -1;
    }
    if (!valid || (device_status & 0x1) == 0) {
        errno = EIO;
        return -1;
    }
    return 0;
}

int
opennpux_coral_vector_add_test(
    struct opennpux_coral_device *dev, uint32_t entry, uint32_t opcode,
    uint32_t element_count, uint64_t polls,
    struct opennpux_coral_vector_add_result *result)
{
    uint32_t input0[OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS];
    uint32_t input1[OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS];
    uint32_t checksum = 0;

    if (element_count == 0 ||
        element_count > OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS) {
        errno = EINVAL;
        return -1;
    }
    for (uint32_t i = 0; i < element_count; ++i) {
        input0[i] = i + 1;
        input1[i] = (i + 1) * 2;
        checksum += input0[i] + input1[i];
    }
    return submit_vector_add(dev, entry, opcode, input0, input1,
                             element_count, checksum, polls, result);
}

static int
model_range_valid(uint32_t file_size, uint32_t offset, uint32_t bytes)
{
    return offset <= file_size && bytes <= file_size - offset;
}

int
opennpux_coral_run_model_file(
    struct opennpux_coral_device *dev, uint32_t entry, const char *path,
    uint64_t polls, struct opennpux_coral_model_result *result)
{
    memset(result, 0, sizeof(*result));
    struct timespec start_time;
    struct timespec end_time;
    memset(&start_time, 0, sizeof(start_time));
    memset(&end_time, 0, sizeof(end_time));
    int model_fd = open(path, O_RDONLY);
    if (model_fd < 0) {
        return -1;
    }

    struct stat st;
    if (fstat(model_fd, &st) != 0 || st.st_size < 0 ||
        (uint64_t)st.st_size > UINT32_MAX) {
        close(model_fd);
        errno = EINVAL;
        return -1;
    }
    const uint32_t file_size = (uint32_t)st.st_size;
    uint8_t *file = malloc(file_size == 0 ? 1 : file_size);
    if (file == NULL) {
        close(model_fd);
        return -1;
    }

    size_t received = 0;
    while (received < file_size) {
        const ssize_t count = read(model_fd, file + received,
                                   file_size - received);
        if (count <= 0) {
            const int saved_errno = count == 0 ? EIO : errno;
            free(file);
            close(model_fd);
            errno = saved_errno;
            return -1;
        }
        received += (size_t)count;
    }
    close(model_fd);

    if (file_size < sizeof(struct opennpux_coral_model_header)) {
        free(file);
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_coral_model_header *header =
        (const struct opennpux_coral_model_header *)file;
    const uint32_t command_bytes =
        header->command_count * sizeof(struct opennpux_coral_model_command);
    if (header->magic != OPENNPUX_CORAL_MODEL_MAGIC ||
        header->version != OPENNPUX_CORAL_MODEL_VERSION ||
        header->header_size != sizeof(*header) ||
        header->file_size != file_size || header->command_count == 0 ||
        header->command_count > OPENNPUX_CORAL_MODEL_MAX_COMMANDS ||
        (header->command_offset & 3) != 0 ||
        !model_range_valid(file_size, header->command_offset,
                           command_bytes)) {
        free(file);
        errno = EINVAL;
        return -1;
    }

    result->command_count = header->command_count;
    struct opennpux_coral_info before;
    struct opennpux_coral_info after;
    opennpux_coral_get_info(dev, &before);
    const int timing_available =
        clock_gettime(CLOCK_MONOTONIC, &start_time) == 0;
    const struct opennpux_coral_model_command *commands =
        (const struct opennpux_coral_model_command *)(
            file + header->command_offset);
    for (uint32_t index = 0; index < header->command_count; ++index) {
        const struct opennpux_coral_model_command *model_command =
            &commands[index];
        const uint32_t tensor_bytes =
            model_command->element_count * sizeof(uint32_t);
        if (model_command->element_count == 0 ||
            model_command->element_count >
                OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS ||
            (model_command->input0_offset & 3) != 0 ||
            (model_command->input1_offset & 3) != 0 ||
            !model_range_valid(file_size, model_command->input0_offset,
                               tensor_bytes) ||
            !model_range_valid(file_size, model_command->input1_offset,
                               tensor_bytes)) {
            free(file);
            errno = EINVAL;
            return -1;
        }

        uint32_t input0[OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS];
        uint32_t input1[OPENNPUX_CORAL_TENSOR_MAX_ELEMENTS];
        memcpy(input0, file + model_command->input0_offset, tensor_bytes);
        memcpy(input1, file + model_command->input1_offset, tensor_bytes);

        struct opennpux_coral_vector_add_result command_result;
        if (submit_vector_add(dev, entry, model_command->opcode,
                              input0, input1, model_command->element_count,
                              model_command->expected_checksum, polls,
                              &command_result) != 0) {
            free(file);
            return -1;
        }
        ++result->completed_commands;
        result->output_checksum += command_result.checksum;
        result->accelerator_cycles += command_result.accelerator_cycles;
    }

    opennpux_coral_get_info(dev, &after);
    result->dma_requests = after.dma_requests - before.dma_requests;
    result->dma_completions =
        after.dma_completions - before.dma_completions;
    result->dma_errors = after.dma_errors - before.dma_errors;
    if (result->dma_requests == 0 ||
        result->dma_requests != result->dma_completions ||
        result->dma_errors != 0) {
        free(file);
        errno = EIO;
        return -1;
    }

    if (timing_available && clock_gettime(CLOCK_MONOTONIC, &end_time) == 0) {
        const uint64_t start_ns =
            (uint64_t)start_time.tv_sec * UINT64_C(1000000000) +
            (uint64_t)start_time.tv_nsec;
        const uint64_t end_ns =
            (uint64_t)end_time.tv_sec * UINT64_C(1000000000) +
            (uint64_t)end_time.tv_nsec;
        if (end_ns >= start_ns) {
            result->host_elapsed_ns = end_ns - start_ns;
        }
    }

    free(file);
    return 0;
}

int
opennpux_coral_mobilenet_test(
    struct opennpux_coral_device *dev, uint32_t entry, uint64_t polls,
    struct opennpux_coral_mobilenet_result *result)
{
    memset(result, 0, sizeof(*result));
    const size_t required = OPENNPUX_CORAL_MOBILENET_MAILBOX_OFFSET +
                            sizeof(struct opennpux_coral_mobilenet_mailbox);
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, required, &window) != 0) {
        return -1;
    }

    volatile uint8_t *mailbox_bytes =
        window.bytes + OPENNPUX_CORAL_MOBILENET_MAILBOX_OFFSET;
    memset((void *)mailbox_bytes, 0,
           sizeof(struct opennpux_coral_mobilenet_mailbox));
    __sync_synchronize();
    struct opennpux_coral_info before;
    struct opennpux_coral_info after;
    opennpux_coral_get_info(dev, &before);

    const int run_result =
        opennpux_coral_run(dev, entry, polls, &result->device_status);
    const int run_errno = errno;
    __sync_synchronize();

    volatile const struct opennpux_coral_mobilenet_mailbox *mailbox =
        (volatile const struct opennpux_coral_mobilenet_mailbox *)mailbox_bytes;
    result->state = mailbox->state;
    result->error_code = mailbox->error_code;
    result->output_count = mailbox->output_count;
    for (uint32_t i = 0; i < OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT; ++i) {
        result->output[i] = mailbox->output[i];
    }
    result->output_checksum = mailbox->output_checksum;
    result->output_bytes = mailbox->output_bytes;
    result->operation_count = mailbox->operation_count;
    result->bytes_read = mailbox->bytes_read;
    result->bytes_written = mailbox->bytes_written;
    result->npu_cycles = ((uint64_t)mailbox->cycle_high << 32) |
                         mailbox->cycle_low;
    const int mailbox_valid =
        mailbox->magic == OPENNPUX_CORAL_MOBILENET_MAGIC &&
        mailbox->version == OPENNPUX_CORAL_MOBILENET_VERSION &&
        result->state == OPENNPUX_CORAL_MOBILENET_COMPLETE &&
        result->error_code == OPENNPUX_CORAL_MOBILENET_ERROR_NONE &&
        result->output_count == OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT &&
        result->output_bytes != 0;

    opennpux_coral_get_info(dev, &after);
    result->dma_requests = after.dma_requests - before.dma_requests;
    result->dma_completions =
        after.dma_completions - before.dma_completions;
    result->dma_errors = after.dma_errors - before.dma_errors;
    opennpux_coral_close_shared_window(&window);

    if (run_result != 0) {
        errno = run_errno;
        return -1;
    }
    if (!mailbox_valid || result->dma_requests == 0 ||
        result->dma_requests != result->dma_completions ||
        result->dma_errors != 0) {
        errno = EIO;
        return -1;
    }
    return 0;
}

int
opennpux_coral_generic_test(
    struct opennpux_coral_device *dev, uint32_t entry, uint64_t polls,
    struct opennpux_coral_generic_test_result *result)
{
    memset(result, 0, sizeof(*result));
    const size_t required = OPENNPUX_CORAL_GENERIC_TEST_MAILBOX_OFFSET +
                            sizeof(struct opennpux_coral_generic_test_mailbox);
    struct opennpux_coral_shared_window window;
    if (opennpux_coral_open_shared_window(dev, required, &window) != 0) {
        return -1;
    }

    volatile uint8_t *mailbox_bytes =
        window.bytes + OPENNPUX_CORAL_GENERIC_TEST_MAILBOX_OFFSET;
    memset((void *)mailbox_bytes, 0,
           sizeof(struct opennpux_coral_generic_test_mailbox));
    __sync_synchronize();
    struct opennpux_coral_info before;
    struct opennpux_coral_info after;
    opennpux_coral_get_info(dev, &before);

    const int run_result =
        opennpux_coral_run(dev, entry, polls, &result->device_status);
    const int run_errno = errno;
    __sync_synchronize();

    volatile const struct opennpux_coral_generic_test_mailbox *mailbox =
        (volatile const struct opennpux_coral_generic_test_mailbox *)
            mailbox_bytes;
    result->state = mailbox->state;
    result->error_code = mailbox->error_code;
    result->output_count = mailbox->output_count;
    for (uint32_t i = 0; i < OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT; ++i) {
        result->output[i] = mailbox->output[i];
    }
    result->output_checksum = mailbox->output_checksum;
    result->output_bytes = mailbox->output_bytes;
    result->operation_count = mailbox->operation_count;
    result->bytes_read = mailbox->bytes_read;
    result->bytes_written = mailbox->bytes_written;
    result->npu_cycles = ((uint64_t)mailbox->cycle_high << 32) |
                         mailbox->cycle_low;
    const int mailbox_valid =
        mailbox->magic == OPENNPUX_CORAL_GENERIC_TEST_MAGIC &&
        mailbox->version == OPENNPUX_CORAL_GENERIC_TEST_VERSION &&
        result->state == OPENNPUX_CORAL_GENERIC_TEST_COMPLETE &&
        result->error_code == OPENNPUX_CORAL_GENERIC_TEST_ERROR_NONE &&
        result->output_count == OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT &&
        result->output_bytes == OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT *
                                sizeof(result->output[0]);

    opennpux_coral_get_info(dev, &after);
    result->dma_requests = after.dma_requests - before.dma_requests;
    result->dma_completions =
        after.dma_completions - before.dma_completions;
    result->dma_errors = after.dma_errors - before.dma_errors;
    opennpux_coral_close_shared_window(&window);

    if (run_result != 0) {
        errno = run_errno;
        return -1;
    }
    if (!mailbox_valid || result->dma_requests == 0 ||
        result->dma_requests != result->dma_completions ||
        result->dma_errors != 0) {
        errno = EIO;
        return -1;
    }
    return 0;
}

int
opennpux_coral_xgraph_test(
    struct opennpux_coral_device *dev, uint32_t entry, uint64_t polls,
    struct opennpux_coral_generic_test_result *result)
{
    enum {
        tokens = OPENNPUX_XGRAPH_DATA_OFFSET,
        embedding = OPENNPUX_XGRAPH_DATA_OFFSET + 0x100,
        matrix = OPENNPUX_XGRAPH_DATA_OFFSET + 0x200,
        bias = OPENNPUX_XGRAPH_DATA_OFFSET + 0x300,
        scale = OPENNPUX_XGRAPH_DATA_OFFSET + 0x400,
        norm_weight = OPENNPUX_XGRAPH_DATA_OFFSET + 0x500,
        rope_table = OPENNPUX_XGRAPH_DATA_OFFSET + 0x600,
        tensor0 = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1000,
        tensor1 = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1100,
        tensor2 = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1200,
        tensor3 = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1300,
        tensor4 = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1400,
        tensor5 = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1500,
        tensor6 = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1600,
        tensor7 = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1700,
        packed_topk = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1800,
        gptq_qweight = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1a00,
        gptq_qzeros = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1b00,
        gptq_scales = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1c00,
        gptq_input = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1d00,
        gptq_scratch = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1e00,
        gptq_output = OPENNPUX_XGRAPH_DATA_OFFSET + 0x1f00,
        required_size = OPENNPUX_XGRAPH_DATA_OFFSET + 0x2000,
    };
    static const uint32_t token_values[2] = {0, 1};
    static const float embedding_values[8] = {
        1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 9.0f, 2.0f, 3.0f,
    };
    static const float matrix_values[16] = {
        1.0f, 2.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    };
    static const float bias_values[8] = {
        0.5f, -1.0f, 1.0f, -0.5f, -0.25f, 0.75f, -1.0f, 1.25f,
    };
    static const float scale_values[8] = {
        0.5f, 1.25f, 0.75f, 1.5f, 1.1f, 3.0f, 1.3f, 0.6f,
    };
    static const float norm_weight_values[4] = {
        0.75f, 1.25f, 0.5f, 1.0f,
    };
    static const float rope_values[16] = {
        0.95f, 0.95f, 0.95f, 0.95f,
        0.95f, 0.95f, 0.95f, 0.95f,
        0.3122499f, 0.3122499f, 0.3122499f, 0.3122499f,
        0.3122499f, 0.3122499f, 0.3122499f, 0.3122499f,
    };
    static const uint32_t gptq_qweight_values[2] = {
        UINT32_C(0x99999999), UINT32_C(0xaaaaaaaa),
    };
    static const uint32_t gptq_qzeros_values[2] = {
        UINT32_C(0x00000077), UINT32_C(0x00000077),
    };
    static const float gptq_scale_values[4] = {
        0.5f, 1.0f, 0.5f, 1.0f,
    };
    static const float gptq_input_values[16] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    static const float gptq_expected_values[4] = {
        18.0f, 72.0f, 4.0f, 16.0f,
    };
    union {
        float value;
        uint32_t bits;
    } epsilon = {1.0e-5f};
    memset(result, 0, sizeof(*result));
    enum { request_count = 10, command_capacity = 9 };
    struct opennpux_npu_functional_request requests[request_count];
    struct opennpux_npu_operator_parameters parameters[request_count];
    struct opennpux_npu_xgraph_lowering_options options[request_count];
    struct opennpux_xgraph_command commands[command_capacity];
    uint32_t command_origins[command_capacity];
    memset(options, 0, sizeof(options));

    initialize_xgraph_request(&requests[0], &parameters[0], 0,
                              OPENNPUX_NPU_OP_EMBED, 2, 4);
    parameters[0].input_features = 2;
    initialize_xgraph_request(&requests[1], &parameters[1], 1,
                              OPENNPUX_NPU_OP_MATMUL, 2, 4);
    parameters[1].input_features = 4;
    parameters[1].output_features = 4;
    initialize_xgraph_request(&requests[2], &parameters[2], 2,
                              OPENNPUX_NPU_OP_ADD, 2, 4);
    initialize_xgraph_request(&requests[3], &parameters[3], 3,
                              OPENNPUX_NPU_OP_MUL, 2, 4);
    initialize_xgraph_request(&requests[4], &parameters[4], 4,
                              OPENNPUX_NPU_OP_NORMALIZE, 2, 4);
    requests[4].epsilon = epsilon.value;
    initialize_xgraph_request(&requests[5], &parameters[5], 5,
                              OPENNPUX_NPU_OP_ROPE, 2, 4);
    options[5].rope_layout = OPENNPUX_NPU_XGRAPH_ROPE_ADJACENT;
    initialize_xgraph_request(&requests[6], &parameters[6], 6,
                              OPENNPUX_NPU_OP_ACTIVATION, 2, 4);
    options[6].activation = OPENNPUX_NPU_XGRAPH_ACTIVATION_SILU;
    initialize_xgraph_request(&requests[7], &parameters[7], 7,
                              OPENNPUX_NPU_OP_SOFTMAX, 1, 8);
    initialize_xgraph_request(&requests[8], &parameters[8], 8,
                              OPENNPUX_NPU_OP_TOPK, 1, 8);
    requests[8].top_k = 1;
    options[8].topk_packed_address =
        CORAL_LOCAL_EXTMEM_BASE + packed_topk;
    options[8].topk_packed_size = 2 * sizeof(uint32_t);
    initialize_xgraph_request(&requests[9], &parameters[9], 9,
                              OPENNPUX_NPU_OP_MATMUL, 2, 8);
    parameters[9].flags = OPENNPUX_NPU_PARAMETER_GPTQ;
    parameters[9].input_features = 8;
    parameters[9].output_features = 2;
    parameters[9].quantization_bits = 4;
    parameters[9].quantization_group_size = 4;
    parameters[9].quantized_zero_bias = 1;
    parameters[9].scale_data_type = OPENNPUX_NPU_DTYPE_FLOAT32;

#define ADD_XGRAPH_OPERAND(index, role, offset, size)                         \
    do {                                                                      \
        if (add_xgraph_operand(&requests[(index)], (role), (offset),          \
                               (size)) != 0) {                                \
            return -1;                                                       \
        }                                                                     \
    } while (0)
    ADD_XGRAPH_OPERAND(0, OPENNPUX_NPU_OPERAND_WEIGHT, embedding,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(0, OPENNPUX_NPU_OPERAND_INPUT_INDICES, tokens,
                       sizeof(token_values));
    ADD_XGRAPH_OPERAND(0, OPENNPUX_NPU_OPERAND_OUTPUT, tensor0,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(1, OPENNPUX_NPU_OPERAND_INPUT, tensor0,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(1, OPENNPUX_NPU_OPERAND_WEIGHT, matrix,
                       sizeof(matrix_values));
    ADD_XGRAPH_OPERAND(1, OPENNPUX_NPU_OPERAND_OUTPUT, tensor1,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(2, OPENNPUX_NPU_OPERAND_INPUT, tensor1,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(2, OPENNPUX_NPU_OPERAND_SECONDARY, bias,
                       sizeof(bias_values));
    ADD_XGRAPH_OPERAND(2, OPENNPUX_NPU_OPERAND_OUTPUT, tensor2,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(3, OPENNPUX_NPU_OPERAND_INPUT, tensor2,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(3, OPENNPUX_NPU_OPERAND_SECONDARY, scale,
                       sizeof(scale_values));
    ADD_XGRAPH_OPERAND(3, OPENNPUX_NPU_OPERAND_OUTPUT, tensor3,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(4, OPENNPUX_NPU_OPERAND_INPUT, tensor3,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(4, OPENNPUX_NPU_OPERAND_WEIGHT, norm_weight,
                       sizeof(norm_weight_values));
    ADD_XGRAPH_OPERAND(4, OPENNPUX_NPU_OPERAND_OUTPUT, tensor4,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(5, OPENNPUX_NPU_OPERAND_INPUT, tensor4,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(5, OPENNPUX_NPU_OPERAND_SECONDARY, rope_table,
                       sizeof(rope_values));
    ADD_XGRAPH_OPERAND(5, OPENNPUX_NPU_OPERAND_OUTPUT, tensor5,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(6, OPENNPUX_NPU_OPERAND_INPUT, tensor5,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(6, OPENNPUX_NPU_OPERAND_OUTPUT, tensor6,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(7, OPENNPUX_NPU_OPERAND_INPUT, tensor6,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(7, OPENNPUX_NPU_OPERAND_OUTPUT, tensor7,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(8, OPENNPUX_NPU_OPERAND_INPUT, tensor7,
                       sizeof(embedding_values));
    ADD_XGRAPH_OPERAND(9, OPENNPUX_NPU_OPERAND_INPUT, gptq_input,
                       sizeof(gptq_input_values));
    ADD_XGRAPH_OPERAND(9, OPENNPUX_NPU_OPERAND_OUTPUT, gptq_output,
                       sizeof(gptq_expected_values));
    ADD_XGRAPH_OPERAND(9, OPENNPUX_NPU_OPERAND_QWEIGHT, gptq_qweight,
                       sizeof(gptq_qweight_values));
    ADD_XGRAPH_OPERAND(9, OPENNPUX_NPU_OPERAND_QZEROS, gptq_qzeros,
                       sizeof(gptq_qzeros_values));
    ADD_XGRAPH_OPERAND(9, OPENNPUX_NPU_OPERAND_SCALES, gptq_scales,
                       sizeof(gptq_scale_values));
#undef ADD_XGRAPH_OPERAND

    struct opennpux_coral_shared_window window;
    fprintf(stderr, "xgraph_stage=open-shared-window required=0x%zx\n",
            (size_t)required_size);
    if (opennpux_coral_open_shared_window(dev, required_size, &window) != 0) {
        return -1;
    }
    fprintf(stderr,
            "xgraph_stage=shared-window-open base=0x%08" PRIx32
            " size=0x%08" PRIx32 "\n",
            window.base, window.size);
    volatile uint8_t *mailbox_bytes =
        window.bytes + OPENNPUX_CORAL_GENERIC_TEST_MAILBOX_OFFSET;
    volatile struct opennpux_xgraph_header *header =
        (volatile struct opennpux_xgraph_header *)(void *)(
            window.bytes + OPENNPUX_XGRAPH_OFFSET);
    fprintf(stderr, "xgraph_stage=stage-tensors offset=0x%08x\n", tokens);
    copy_to_volatile_bytes(window.bytes + tokens, token_values,
                           sizeof(token_values));
    copy_to_volatile_bytes(window.bytes + embedding, embedding_values,
                           sizeof(embedding_values));
    copy_to_volatile_bytes(window.bytes + matrix, matrix_values,
                           sizeof(matrix_values));
    copy_to_volatile_bytes(window.bytes + bias, bias_values,
                           sizeof(bias_values));
    copy_to_volatile_bytes(window.bytes + scale, scale_values,
                           sizeof(scale_values));
    copy_to_volatile_bytes(window.bytes + norm_weight, norm_weight_values,
                           sizeof(norm_weight_values));
    copy_to_volatile_bytes(window.bytes + rope_table, rope_values,
                           sizeof(rope_values));
    copy_to_volatile_bytes(window.bytes + gptq_qweight,
                           gptq_qweight_values,
                           sizeof(gptq_qweight_values));
    copy_to_volatile_bytes(window.bytes + gptq_qzeros, gptq_qzeros_values,
                           sizeof(gptq_qzeros_values));
    copy_to_volatile_bytes(window.bytes + gptq_scales, gptq_scale_values,
                           sizeof(gptq_scale_values));
    copy_to_volatile_bytes(window.bytes + gptq_input, gptq_input_values,
                           sizeof(gptq_input_values));
    struct opennpux_coral_info before;
    struct opennpux_coral_info after;
    opennpux_coral_get_info(dev, &before);
    volatile const struct opennpux_coral_generic_test_mailbox *mailbox =
        (volatile const struct opennpux_coral_generic_test_mailbox *)
            mailbox_bytes;
    uint32_t request_offset = 0;
    uint32_t total_commands = 0;
    uint32_t final_batch_commands = 0;
    int run_result = 0;
    int run_errno = 0;
    while (request_offset < request_count) {
        uint32_t requests_consumed = 0;
        uint32_t commands_emitted = 0;
        struct opennpux_npu_xgraph_lowering_failure lowering_failure;
        const int lowering_result = opennpux_npu_xgraph_lower_batch(
            requests + request_offset, parameters + request_offset,
            options + request_offset, request_count - request_offset,
            CORAL_LOCAL_EXTMEM_BASE, required_size,
            CORAL_LOCAL_EXTMEM_BASE + gptq_scratch, 0x100, commands,
            command_capacity, command_origins, &requests_consumed,
            &commands_emitted, &lowering_failure);
        if (lowering_result != 0 || requests_consumed == 0 ||
            commands_emitted == 0) {
            fprintf(stderr,
                    "xgraph_stage=lowering-failed request=%" PRIu32
                    " command_id=%" PRIu32 " opcode=%" PRIu32
                    " error=%" PRId32 " consumed=%" PRIu32
                    " emitted=%" PRIu32 "\n",
                    request_offset + lowering_failure.command_index,
                    lowering_failure.command_id, lowering_failure.opcode,
                    lowering_failure.error_code, requests_consumed,
                    commands_emitted);
            run_errno = errno == 0 ? EIO : errno;
            run_result = -1;
            break;
        }
        for (uint32_t index = 0; index < commands_emitted; ++index) {
            if (command_origins[index] < request_offset ||
                command_origins[index] >=
                    request_offset + requests_consumed) {
                run_errno = EIO;
                run_result = -1;
                break;
            }
        }
        if (run_result != 0) {
            break;
        }

        const uint32_t batch = result->batch_count;
        const int final_batch =
            request_offset + requests_consumed == request_count;
        fprintf(stderr, "xgraph_stage=clear-mailbox offset=0x%08" PRIx32
                        " batch=%" PRIu32 "\n",
                OPENNPUX_CORAL_GENERIC_TEST_MAILBOX_OFFSET, batch);
        zero_volatile_bytes(
            mailbox_bytes, sizeof(struct opennpux_coral_generic_test_mailbox));
        zero_volatile_bytes((volatile uint8_t *)(void *)header,
                            sizeof(*header));
        copy_to_volatile_bytes((volatile uint8_t *)(void *)(header + 1),
                               commands,
                               commands_emitted * sizeof(commands[0]));
        header->magic = OPENNPUX_XGRAPH_MAGIC;
        header->version = OPENNPUX_XGRAPH_VERSION;
        header->header_size = sizeof(*header);
        header->command_size = sizeof(commands[0]);
        header->command_count = commands_emitted;
        header->total_size =
            sizeof(*header) + commands_emitted * sizeof(commands[0]);
        header->output_offset = packed_topk;
        header->output_bytes = 2 * sizeof(uint32_t);
        header->reserved[OPENNPUX_XGRAPH_BATCH_SEQUENCE] = batch;
        header->reserved[OPENNPUX_XGRAPH_BATCH_FIRST_REQUEST] =
            request_offset;
        header->reserved[OPENNPUX_XGRAPH_BATCH_REQUEST_COUNT] =
            requests_consumed;
        header->reserved[OPENNPUX_XGRAPH_BATCH_FLAGS] =
            final_batch ? OPENNPUX_XGRAPH_BATCH_FLAG_FINAL : 0;
        header->reserved[OPENNPUX_XGRAPH_BATCH_FIRST_COMMAND] =
            total_commands;
        __sync_synchronize();
        header->state = OPENNPUX_XGRAPH_STATE_READY;
        __sync_synchronize();
        fprintf(stderr,
                "xgraph_stage=submission-ready source=generic-lowering"
                " batch=%" PRIu32 " requests=%" PRIu32
                " commands=%" PRIu32 " final=%d\n",
                batch, requests_consumed, commands_emitted, final_batch);

        run_result =
            opennpux_coral_run(dev, entry, polls, &result->device_status);
        run_errno = errno;
        __sync_synchronize();
        if (run_result != 0 ||
            header->state != OPENNPUX_XGRAPH_STATE_COMPLETE ||
            header->error != OPENNPUX_XGRAPH_ERROR_NONE ||
            header->completed_commands != commands_emitted ||
            mailbox->magic != OPENNPUX_CORAL_GENERIC_TEST_MAGIC ||
            mailbox->state != OPENNPUX_CORAL_GENERIC_TEST_COMPLETE ||
            mailbox->error_code != OPENNPUX_CORAL_GENERIC_TEST_ERROR_NONE ||
            mailbox->output[0] != (int32_t)commands_emitted) {
            if (run_result == 0) {
                run_result = -1;
                run_errno = EIO;
            }
            break;
        }

        result->state = mailbox->state;
        result->error_code = mailbox->error_code;
        result->output_count = mailbox->output_count;
        result->operation_count += mailbox->operation_count;
        result->bytes_read += mailbox->bytes_read;
        result->bytes_written += mailbox->bytes_written;
        result->npu_cycles += ((uint64_t)mailbox->cycle_high << 32) |
                              mailbox->cycle_low;
        ++result->batch_count;
        result->completed_requests += requests_consumed;
        result->completed_commands += commands_emitted;
        final_batch_commands = commands_emitted;
        request_offset += requests_consumed;
        total_commands += commands_emitted;
    }
    if (run_result != 0 || result->completed_requests != request_count ||
        result->completed_commands != 12 || result->batch_count != 2) {
        opennpux_coral_close_shared_window(&window);
        errno = run_errno == 0 ? EIO : run_errno;
        return -1;
    }
    result->output[0] = (int32_t)result->completed_commands;
    uint32_t packed_topk_result[2] = {0};
    copy_from_volatile_bytes(packed_topk_result,
                             window.bytes + packed_topk,
                             sizeof(packed_topk_result));
    /* Packed TopK is [value bits, index]; mailbox output is [index, value]. */
    result->output[1] = (int32_t)packed_topk_result[1];
    result->output[2] = (int32_t)packed_topk_result[0];
    result->output[3] = 0;
    result->output_count = OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT;
    result->output_bytes = sizeof(result->output);
    result->output_checksum =
        checksum_bytes(result->output, sizeof(result->output));
    static const uint32_t tensor_offsets[8] = {
        tensor0, tensor1, tensor2, tensor3,
        tensor4, tensor5, tensor6, tensor7,
    };
    float actual[8][8] = {{0}};
    float expected[8][8] = {{0}};
    for (size_t operation = 0; operation < 8; ++operation) {
        copy_from_volatile_bytes(actual[operation],
            window.bytes + tensor_offsets[operation],
            sizeof(actual[operation]));
    }
    memcpy(expected[0], embedding_values, sizeof(expected[0]));
    for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            float accumulator = 0.0f;
            for (size_t inner = 0; inner < 4; ++inner) {
                accumulator += expected[0][row * 4 + inner] *
                               matrix_values[inner * 4 + column];
            }
            expected[1][row * 4 + column] = accumulator;
        }
    }
    for (size_t index = 0; index < 8; ++index) {
        expected[2][index] = expected[1][index] + bias_values[index];
        expected[3][index] = expected[2][index] * scale_values[index];
    }
    for (size_t row = 0; row < 2; ++row) {
        float sum_squares = 0.0f;
        for (size_t feature = 0; feature < 4; ++feature) {
            const float value = expected[3][row * 4 + feature];
            sum_squares += value * value;
        }
        const float inverse_rms =
            1.0f / sqrtf(sum_squares / 4.0f + epsilon.value);
        for (size_t feature = 0; feature < 4; ++feature) {
            expected[4][row * 4 + feature] =
                expected[3][row * 4 + feature] * inverse_rms *
                norm_weight_values[feature];
        }
    }
    for (size_t row = 0; row < 2; ++row) {
        for (size_t feature = 0; feature < 4; ++feature) {
            const size_t index = row * 4 + feature;
            const size_t rotated = row * 4 + (feature ^ 1);
            const float sign = (feature & 1) == 0 ? -1.0f : 1.0f;
            expected[5][index] =
                expected[4][index] * rope_values[index] +
                sign * expected[4][rotated] * rope_values[8 + index];
        }
    }
    for (size_t index = 0; index < 8; ++index) {
        const float value = expected[5][index];
        expected[6][index] = value / (1.0f + expf(-value));
    }
    float maximum = expected[6][0];
    for (size_t index = 1; index < 8; ++index) {
        if (expected[6][index] > maximum) {
            maximum = expected[6][index];
        }
    }
    float sum = 0.0f;
    for (size_t index = 0; index < 8; ++index) {
        expected[7][index] = expf(expected[6][index] - maximum);
        sum += expected[7][index];
    }
    for (size_t index = 0; index < 8; ++index) {
        expected[7][index] /= sum;
    }

    result->failed_operator = UINT32_MAX;
    int operators_valid = 1;
    for (uint32_t operation = 0; operation < 8; ++operation) {
        result->operator_checksums[operation] =
            checksum_bytes(actual[operation], sizeof(actual[operation]));
        result->operator_pass[operation] =
            compare_floats(actual[operation], expected[operation], 8,
                           &result->operator_max_abs_error[operation]);
        if (!result->operator_pass[operation]) {
            if (result->failed_operator == UINT32_MAX) {
                result->failed_operator = operation;
            }
            operators_valid = 0;
        }
        ++result->validated_operators;
    }
    uint32_t actual_topk[2] = {0};
    copy_from_volatile_bytes(actual_topk, window.bytes + packed_topk,
                             sizeof(actual_topk));
    result->operator_checksums[8] =
        checksum_bytes(actual_topk, sizeof(actual_topk));
    uint32_t expected_index = 0;
    for (uint32_t index = 1; index < 8; ++index) {
        if (expected[7][index] > expected[7][expected_index]) {
            expected_index = index;
        }
    }
    union {
        float value;
        uint32_t bits;
    } expected_topk = {expected[7][expected_index]};
    union {
        float value;
        uint32_t bits;
    } actual_topk_value = {.bits = actual_topk[0]};
    const int topk_valid =
        compare_floats(&actual_topk_value.value, &expected_topk.value, 1,
                       &result->operator_max_abs_error[8]) &&
        actual_topk[1] == expected_index;
    result->operator_pass[8] = topk_valid;
    if (!topk_valid) {
        if (result->failed_operator == UINT32_MAX) {
            result->failed_operator = 8;
        }
        operators_valid = 0;
    }
    ++result->validated_operators;

    float actual_gptq[4] = {0};
    copy_from_volatile_bytes(actual_gptq, window.bytes + gptq_output,
                             sizeof(actual_gptq));
    result->operator_checksums[9] =
        checksum_bytes(actual_gptq, sizeof(actual_gptq));
    result->operator_pass[9] = compare_floats(
        actual_gptq, gptq_expected_values, 4,
        &result->operator_max_abs_error[9]);
    if (!result->operator_pass[9]) {
        if (result->failed_operator == UINT32_MAX) {
            result->failed_operator = 9;
        }
        operators_valid = 0;
    }
    ++result->validated_operators;

    const int valid =
        header->state == OPENNPUX_XGRAPH_STATE_COMPLETE &&
        header->error == OPENNPUX_XGRAPH_ERROR_NONE &&
        header->completed_commands == final_batch_commands &&
        mailbox->magic == OPENNPUX_CORAL_GENERIC_TEST_MAGIC &&
        mailbox->state == OPENNPUX_CORAL_GENERIC_TEST_COMPLETE &&
        mailbox->error_code == OPENNPUX_CORAL_GENERIC_TEST_ERROR_NONE &&
        result->output_count == OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT &&
        result->output[0] == (int32_t)result->completed_commands &&
        result->output[1] == 5 &&
        result->completed_requests == request_count &&
        result->completed_commands == 12 &&
        result->batch_count == 2 &&
        operators_valid;

    opennpux_coral_get_info(dev, &after);
    result->dma_requests = after.dma_requests - before.dma_requests;
    result->dma_completions =
        after.dma_completions - before.dma_completions;
    result->dma_errors = after.dma_errors - before.dma_errors;
    opennpux_coral_close_shared_window(&window);
    if (!valid) {
        errno = EIO;
        return -1;
    }
    return 0;
}
