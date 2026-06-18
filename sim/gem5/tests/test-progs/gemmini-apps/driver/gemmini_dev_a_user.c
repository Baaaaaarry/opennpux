#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "gemmini_dev_a_ioctl.h"

static void fill_seq(float *buf, size_t elems, float base)
{
    for (size_t i = 0; i < elems; ++i)
        buf[i] = base + (float)i;
}

static void print_matrix(float *buf, unsigned int m_size)
{
    for (unsigned int i = 0; i < m_size; ++i) {
        for (unsigned int j = 0; j < m_size; ++j)
            printf("%6.1f ", buf[i * m_size + j]);
        printf("\n");
    }
}

int main(int argc, char **argv)
{
    unsigned int m_size = 2;
    size_t elems;
    float *a = NULL, *b = NULL, *c = NULL;
    int fd, ret;
    struct gemmini_dev_a_req req;

    if (argc > 1) {
        m_size = (unsigned int)atoi(argv[1]);
        if (m_size == 0) {
            fprintf(stderr, "Invalid m_size\n");
            return 1;
        }
    }

    elems = (size_t)m_size * (size_t)m_size;
    a = calloc(elems, sizeof(float));
    b = calloc(elems, sizeof(float));
    c = calloc(elems, sizeof(float));
    if (!a || !b || !c) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    fill_seq(a, elems, 1.0f);
    fill_seq(b, elems, 1.0f);

    fd = open("/dev/gemmini_dev_a", O_RDWR);
    if (fd < 0) {
        fprintf(stderr, "open /dev/gemmini_dev_a failed: %s\n", strerror(errno));
        return 1;
    }

    memset(&req, 0, sizeof(req));
    req.m_user = (uint64_t)(uintptr_t)a;
    req.k_user = (uint64_t)(uintptr_t)b;
    req.o_user = (uint64_t)(uintptr_t)c;
    req.m_size = m_size;
    req.k_size = 0;
    req.opcode = GEMMINI_OP_MM;

    ret = ioctl(fd, GEMMINI_DEV_A_IOC_RUN, &req);
    if (ret != 0) {
        fprintf(stderr, "ioctl failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }

    printf("GemminiDevA MM result (m_size=%u):\n", m_size);
    print_matrix(c, m_size);

    close(fd);
    free(a);
    free(b);
    free(c);
    return 0;
}
