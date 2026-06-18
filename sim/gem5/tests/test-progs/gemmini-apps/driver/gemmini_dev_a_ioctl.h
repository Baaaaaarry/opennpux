#ifndef GEMMINI_DEV_A_IOCTL_H
#define GEMMINI_DEV_A_IOCTL_H

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <stdint.h>
#include <sys/ioctl.h>
#endif

#define GEMMINI_DEV_A_IOC_MAGIC 'G'

enum gemmini_dev_a_opcode {
    GEMMINI_OP_CONV2D = 0,
    GEMMINI_OP_CONV2D_GEMM,
    GEMMINI_OP_CONV3D,
    GEMMINI_OP_CONV3D_GEMM,
    GEMMINI_OP_MAXPOOL,
    GEMMINI_OP_MAXPOOL_GEMM,
    GEMMINI_OP_RELU,
    GEMMINI_OP_MM,
    GEMMINI_OP_MM_GEMM,
};

struct gemmini_dev_a_req {
#ifdef __KERNEL__
    __u64 m_user;
    __u64 k_user;
    __u64 o_user;
    __u32 m_size;
    __u32 k_size;
    __u32 opcode;
    __u32 reserved;
#else
    uint64_t m_user;
    uint64_t k_user;
    uint64_t o_user;
    uint32_t m_size;
    uint32_t k_size;
    uint32_t opcode;
    uint32_t reserved;
#endif
};

#define GEMMINI_DEV_A_IOC_RUN _IOWR(GEMMINI_DEV_A_IOC_MAGIC, 0, struct gemmini_dev_a_req)

#endif
