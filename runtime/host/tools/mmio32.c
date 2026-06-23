#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void
usage(const char *prog)
{
    fprintf(stderr,
            "usage:\n"
            "  %s read <phys_addr>\n"
            "  %s write <phys_addr> <value>\n",
            prog, prog);
}

static int
parse_u64(const char *s, uint64_t *value)
{
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(s, &end, 0);
    if (errno != 0 || end == s || *end != '\0') {
        return -1;
    }
    *value = parsed;
    return 0;
}

int
main(int argc, char **argv)
{
    if (argc != 3 && argc != 4) {
        usage(argv[0]);
        return 2;
    }

    const int do_read = strcmp(argv[1], "read") == 0;
    const int do_write = strcmp(argv[1], "write") == 0;
    if (!do_read && !do_write) {
        usage(argv[0]);
        return 2;
    }

    if ((do_read && argc != 3) || (do_write && argc != 4)) {
        usage(argv[0]);
        return 2;
    }

    uint64_t addr = 0;
    if (parse_u64(argv[2], &addr) != 0) {
        fprintf(stderr, "invalid address: %s\n", argv[2]);
        return 2;
    }

    uint64_t write_value = 0;
    if (do_write && parse_u64(argv[3], &write_value) != 0) {
        fprintf(stderr, "invalid value: %s\n", argv[3]);
        return 2;
    }

    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        perror("sysconf(_SC_PAGESIZE)");
        return 1;
    }

    const uint64_t page_mask = (uint64_t)page_size - 1;
    const off_t page_base = (off_t)(addr & ~page_mask);
    const size_t page_offset = (size_t)(addr & page_mask);

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open(/dev/mem)");
        return 1;
    }

    void *mapping = mmap(NULL, (size_t)page_size, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, page_base);
    if (mapping == MAP_FAILED) {
        perror("mmap(/dev/mem)");
        close(fd);
        return 1;
    }

    volatile uint32_t *reg =
        (volatile uint32_t *)((uint8_t *)mapping + page_offset);

    if (do_read) {
        printf("0x%08x\n", *reg);
    } else {
        *reg = (uint32_t)write_value;
        __sync_synchronize();
    }

    if (munmap(mapping, (size_t)page_size) != 0) {
        perror("munmap");
        close(fd);
        return 1;
    }

    close(fd);
    return 0;
}
