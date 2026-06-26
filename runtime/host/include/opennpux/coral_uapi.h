#ifndef OPENNPUX_CORAL_UAPI_H
#define OPENNPUX_CORAL_UAPI_H

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
typedef __u32 opennpux_coral_u32;
typedef __u64 opennpux_coral_u64;
#else
#include <stdint.h>
#include <sys/ioctl.h>
typedef uint32_t opennpux_coral_u32;
typedef uint64_t opennpux_coral_u64;
#endif

#define OPENNPUX_CORAL_DEVICE_PATH "/dev/opennpux-coral"
#define OPENNPUX_CORAL_IOC_MAGIC 'N'

struct opennpux_coral_ioc_info {
    opennpux_coral_u64 base;
    opennpux_coral_u32 backend_id;
    opennpux_coral_u32 firmware_entry;
    opennpux_coral_u32 shared_base;
    opennpux_coral_u32 shared_size;
    opennpux_coral_u32 dma_requests;
    opennpux_coral_u32 dma_completions;
    opennpux_coral_u32 dma_errors;
    opennpux_coral_u32 dma_state;
    opennpux_coral_u32 reset_control;
    opennpux_coral_u32 status;
};

struct opennpux_coral_ioc_run {
    opennpux_coral_u32 entry;
    opennpux_coral_u32 status;
    opennpux_coral_u64 polls;
};

#define OPENNPUX_CORAL_IOC_GET_INFO \
    _IOR(OPENNPUX_CORAL_IOC_MAGIC, 0x00, struct opennpux_coral_ioc_info)
#define OPENNPUX_CORAL_IOC_RUN \
    _IOWR(OPENNPUX_CORAL_IOC_MAGIC, 0x01, struct opennpux_coral_ioc_run)

#endif
