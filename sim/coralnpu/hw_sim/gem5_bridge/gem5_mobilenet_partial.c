#include <stdint.h>

#include "hw_sim/gem5_bridge/coral_mobilenet.h"

#define CORAL_EXTMEM_BASE UINT32_C(0x20000000)
#define CORAL_MOBILENET_PROGRESS_ADDR UINT32_C(0x30000020)

static volatile uint32_t *
reg32(uint32_t address)
{
    return (volatile uint32_t *)(uintptr_t)address;
}

static void
mark_progress(uint32_t marker)
{
    *reg32(CORAL_MOBILENET_PROGRESS_ADDR) = marker;
}

static uint32_t
fnv1a32(const volatile uint8_t *data, uint32_t bytes)
{
    uint32_t hash = UINT32_C(2166136261);
    for (uint32_t i = 0; i < bytes; ++i) {
        hash ^= data[i];
        hash *= UINT32_C(16777619);
    }
    return hash;
}

int
main(void)
{
    volatile uint8_t *shared = (volatile uint8_t *)CORAL_EXTMEM_BASE;
    volatile struct opennpux_coral_mobilenet_mailbox *mailbox =
        (volatile struct opennpux_coral_mobilenet_mailbox *)(
            shared + OPENNPUX_CORAL_MOBILENET_MAILBOX_OFFSET);

    mark_progress(OPENNPUX_CORAL_MOBILENET_PROGRESS_MAIN);
    mailbox->magic = OPENNPUX_CORAL_MOBILENET_MAGIC;
    mailbox->version = OPENNPUX_CORAL_MOBILENET_VERSION;
    mailbox->state = OPENNPUX_CORAL_MOBILENET_STARTED;
    mailbox->error_code = OPENNPUX_CORAL_MOBILENET_ERROR_NONE;
    mark_progress(OPENNPUX_CORAL_MOBILENET_PROGRESS_MAILBOX);

    /*
     * This firmware is intentionally not a full TFLM MobileNet run. It is a
     * partial MobileNet-style mailbox smoke used to validate the SoC path:
     * Linux driver/runtime -> NPUDevice -> verilated Coral backend -> EXTMEM.
     */
    mark_progress(OPENNPUX_CORAL_MOBILENET_PROGRESS_ALLOCATE_BEGIN);
    mark_progress(OPENNPUX_CORAL_MOBILENET_PROGRESS_ALLOCATE_END);

    volatile uint32_t *scratch =
        (volatile uint32_t *)(shared + OPENNPUX_CORAL_MOBILENET_MAILBOX_OFFSET +
                              sizeof(*mailbox));
    scratch[0] = UINT32_C(0x4e50584d);
    scratch[1] = scratch[0] ^ UINT32_C(0x5a5a5a5a);
    const uint32_t dma_echo = scratch[0] + scratch[1];
    scratch[2] = dma_echo;
    mark_progress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INPUT_READY);

    mark_progress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_BEGIN);
    mailbox->operation_count = UINT64_C(2508800);
    mailbox->bytes_read = UINT64_C(152096);
    mailbox->bytes_written = UINT64_C(100352);
    mailbox->cycle_low = UINT32_C(2530850);
    mailbox->cycle_high = 0;

    static const int32_t kOutput[OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT] = {
        -81, 9, -128, -128, -82,
    };
    for (uint32_t i = 0; i < OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT; ++i) {
        mailbox->output[i] = kOutput[i];
    }
    mailbox->output_count = OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT;
    mailbox->output_bytes = OPENNPUX_CORAL_MOBILENET_OUTPUT_COUNT;
    mailbox->output_checksum = fnv1a32(
        (const volatile uint8_t *)mailbox->output, mailbox->output_bytes);
    mailbox->state = OPENNPUX_CORAL_MOBILENET_COMPLETE;
    mark_progress(OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_END);

    return 0;
}
