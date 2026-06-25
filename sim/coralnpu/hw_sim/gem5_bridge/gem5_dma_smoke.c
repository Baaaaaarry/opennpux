#include <stdint.h>

#define CORAL_EXTMEM_BASE 0x20000000u
#define CORAL_DMA_MAGIC 0x4e505544u

int
main(void)
{
    volatile uint32_t *shared =
        (volatile uint32_t *)CORAL_EXTMEM_BASE;

    const uint32_t sum = shared[0] + shared[1];
    shared[0] = sum;
    shared[2] = CORAL_DMA_MAGIC;
    return 0;
}
