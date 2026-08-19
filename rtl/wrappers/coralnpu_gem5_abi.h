#ifndef OPENNPUX_CORALNPU_GEM5_ABI_H_
#define OPENNPUX_CORALNPU_GEM5_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CORAL_GEM5_ABI_VERSION 7u
#define CORAL_GEM5_AXI_DATA_BYTES 16u
#define CORAL_GEM5_DMA_DATA_BYTES 4096u

enum coral_gem5_step_result {
    CORAL_GEM5_STEP_ERROR = -1,
    CORAL_GEM5_STEP_RUNNING = 0,
    CORAL_GEM5_STEP_HALTED = 1,
    CORAL_GEM5_STEP_DMA_WAIT = 2,
    CORAL_GEM5_STEP_WFI = 3,
    CORAL_GEM5_STEP_FAULT = 4,
    CORAL_GEM5_STEP_EXTERNAL_WAIT = 5,
};

enum coral_gem5_dma_type {
    CORAL_GEM5_DMA_READ = 0,
    CORAL_GEM5_DMA_WRITE = 1,
};

typedef struct coral_gem5_dma_request {
    uint32_t type;
    uint32_t addr;
    uint32_t size;
    uint32_t id;
    uint8_t data[CORAL_GEM5_DMA_DATA_BYTES];
} coral_gem5_dma_request;

typedef struct coral_gem5_handle coral_gem5_handle;

uint32_t coral_gem5_abi_version(void);
coral_gem5_handle *coral_gem5_create(void);
void coral_gem5_destroy(coral_gem5_handle *handle);
int coral_gem5_reset(coral_gem5_handle *handle);
int coral_gem5_mmio_read(
    coral_gem5_handle *handle, uint32_t addr, void *data, size_t size);
int coral_gem5_mmio_write(
    coral_gem5_handle *handle, uint32_t addr, const void *data, size_t size);
int coral_gem5_step(coral_gem5_handle *handle, uint32_t cycles);
uint64_t coral_gem5_cycle_count(coral_gem5_handle *handle);
int coral_gem5_dma_request_get(
    coral_gem5_handle *handle, coral_gem5_dma_request *request);
int coral_gem5_dma_complete(
    coral_gem5_handle *handle, const void *data, size_t size, int error);
int coral_gem5_extmem_enable(coral_gem5_handle *handle, int enable);
int coral_gem5_extmem_read(
    coral_gem5_handle *handle, uint32_t addr, void *data, size_t size);
int coral_gem5_extmem_write(
    coral_gem5_handle *handle, uint32_t addr, const void *data, size_t size);
int coral_gem5_operator_mode(coral_gem5_handle *handle, uint32_t mode);

#ifdef __cplusplus
}
#endif

#endif
