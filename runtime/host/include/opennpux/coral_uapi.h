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
#define OPENNPUX_CORAL_ABI_VERSION 1

#define OPENNPUX_CORAL_FEATURE_SHARED_MMAP (1U << 0)
#define OPENNPUX_CORAL_FEATURE_ASYNC_START (1U << 1)
#define OPENNPUX_CORAL_FEATURE_POLL_COMPLETION (1U << 2)
#define OPENNPUX_CORAL_FEATURE_RESET (1U << 3)
#define OPENNPUX_CORAL_FEATURE_EXTMEM_SYNC (1U << 4)

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

struct opennpux_coral_ioc_caps {
    opennpux_coral_u32 abi_version;
    opennpux_coral_u32 features;
    opennpux_coral_u32 reserved[6];
};

struct opennpux_coral_ioc_start {
    opennpux_coral_u32 entry;
    opennpux_coral_u32 flags;
};

struct opennpux_coral_ioc_sync {
    opennpux_coral_u32 offset;
    opennpux_coral_u32 size;
};

#define OPENNPUX_CORAL_IOC_GET_INFO \
    _IOR(OPENNPUX_CORAL_IOC_MAGIC, 0x00, struct opennpux_coral_ioc_info)
#define OPENNPUX_CORAL_IOC_RUN \
    _IOWR(OPENNPUX_CORAL_IOC_MAGIC, 0x01, struct opennpux_coral_ioc_run)
#define OPENNPUX_CORAL_IOC_GET_CAPS \
    _IOR(OPENNPUX_CORAL_IOC_MAGIC, 0x02, struct opennpux_coral_ioc_caps)
#define OPENNPUX_CORAL_IOC_START \
    _IOW(OPENNPUX_CORAL_IOC_MAGIC, 0x03, struct opennpux_coral_ioc_start)
#define OPENNPUX_CORAL_IOC_RESET \
    _IO(OPENNPUX_CORAL_IOC_MAGIC, 0x04)
#define OPENNPUX_CORAL_IOC_SYNC_TO_EXTMEM \
    _IOW(OPENNPUX_CORAL_IOC_MAGIC, 0x05, struct opennpux_coral_ioc_sync)
#define OPENNPUX_CORAL_IOC_SYNC_FROM_EXTMEM \
    _IOW(OPENNPUX_CORAL_IOC_MAGIC, 0x06, struct opennpux_coral_ioc_sync)

#endif
