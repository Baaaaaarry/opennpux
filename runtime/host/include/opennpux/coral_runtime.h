#ifndef OPENNPUX_CORAL_RUNTIME_H
#define OPENNPUX_CORAL_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#include "opennpux/coral_uapi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_CORAL_DEFAULT_BASE UINT64_C(0x1d000000)
#define OPENNPUX_CORAL_DMA_MAGIC UINT32_C(0x4e505544)

enum opennpux_coral_backend {
    OPENNPUX_CORAL_BACKEND_UNKNOWN = 0,
    OPENNPUX_CORAL_BACKEND_STAGE_A = 1,
    OPENNPUX_CORAL_BACKEND_VERILATED = 2,
};

enum opennpux_coral_transport {
    OPENNPUX_CORAL_TRANSPORT_UNKNOWN = 0,
    OPENNPUX_CORAL_TRANSPORT_DEVMEM = 1,
    OPENNPUX_CORAL_TRANSPORT_DRIVER = 2,
};

struct opennpux_coral_device {
    int fd;
    size_t page_size;
    void *mapping;
    uint64_t page_base;
    uint64_t base;
    enum opennpux_coral_transport transport;
};

struct opennpux_coral_info {
    uint64_t base;
    uint32_t backend_id;
    enum opennpux_coral_backend backend;
    uint32_t firmware_entry;
    uint32_t shared_base;
    uint32_t shared_size;
    uint32_t dma_requests;
    uint32_t dma_completions;
    uint32_t dma_errors;
    uint32_t dma_state;
    uint32_t reset_control;
    uint32_t status;
};

struct opennpux_coral_shared_window {
    uint32_t base;
    uint32_t size;
    size_t map_size;
    void *mapping;
    volatile uint8_t *bytes;
};

struct opennpux_coral_dma_test_result {
    uint32_t status;
    uint32_t result;
    uint32_t magic;
    uint32_t requests;
    uint32_t completions;
    uint32_t errors;
    uint32_t state;
};

const char *opennpux_coral_backend_name(enum opennpux_coral_backend backend);
const char *opennpux_coral_transport_name(
    enum opennpux_coral_transport transport);
enum opennpux_coral_backend opennpux_coral_decode_backend(uint32_t id);
int opennpux_coral_parse_u64(const char *text, uint64_t *value);
int opennpux_coral_check_shared_u32_access(uint32_t size, uint64_t offset);

int opennpux_coral_open(struct opennpux_coral_device *dev, uint64_t base);
void opennpux_coral_close(struct opennpux_coral_device *dev);

uint32_t opennpux_coral_read_reg(struct opennpux_coral_device *dev,
                                 uint64_t offset);
void opennpux_coral_write_reg(struct opennpux_coral_device *dev,
                              uint64_t offset, uint32_t value);
void opennpux_coral_get_info(struct opennpux_coral_device *dev,
                             struct opennpux_coral_info *info);

int opennpux_coral_open_shared_window(
    struct opennpux_coral_device *dev, size_t min_size,
    struct opennpux_coral_shared_window *window);
void opennpux_coral_close_shared_window(
    struct opennpux_coral_shared_window *window);
int opennpux_coral_clear_shared_window(struct opennpux_coral_device *dev);
int opennpux_coral_read_shared_u32(struct opennpux_coral_device *dev,
                                   uint64_t offset, uint32_t *value);
int opennpux_coral_write_shared_u32(struct opennpux_coral_device *dev,
                                    uint64_t offset, uint32_t value);

int opennpux_coral_run(struct opennpux_coral_device *dev, uint32_t entry,
                       uint64_t polls, uint32_t *status);
int opennpux_coral_dma_test(struct opennpux_coral_device *dev, uint32_t entry,
                            uint64_t polls,
                            struct opennpux_coral_dma_test_result *result);

#ifdef __cplusplus
}
#endif

#endif
