#ifndef OPENNPUX_CORALNPU_GEM5_ABI_H_
#define OPENNPUX_CORALNPU_GEM5_ABI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CORAL_GEM5_ABI_VERSION 1u

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

#ifdef __cplusplus
}
#endif

#endif
