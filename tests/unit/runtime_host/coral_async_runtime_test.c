#include "opennpux/coral_runtime.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define RESET_CONTROL UINT32_C(0x30000)
#define PC_START UINT32_C(0x30004)
#define STATUS UINT32_C(0x30008)
#define MMIO_SIZE UINT32_C(0x31000)

struct service_context {
    struct opennpux_coral_device *device;
    uint32_t calls;
};

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s (errno=%d)\n", message, errno);
        exit(1);
    }
}

static int
service(void *opaque)
{
    struct service_context *context = opaque;
    ++context->calls;
    if (context->calls == 3) {
        opennpux_coral_write_reg(context->device, STATUS, 1);
    }
    return 0;
}

int
main(void)
{
    void *registers = calloc(1, MMIO_SIZE);
    check(registers != NULL, "MMIO allocation failed");
    struct opennpux_coral_device device = {
        .fd = -1,
        .page_size = MMIO_SIZE,
        .mapping = registers,
        .page_base = 0,
        .base = 0,
        .transport = OPENNPUX_CORAL_TRANSPORT_DEVMEM,
    };
    struct service_context context = {
        .device = &device,
    };
    uint32_t status = 0;
    check(opennpux_coral_run_with_service(
              &device, UINT32_C(0x1234), 10, service, &context,
              &status) == 0,
          "serviced run failed");
    const uint32_t *words = registers;
    check(context.calls == 3 && status == 1,
          "service callback count mismatch");
    check(words[PC_START / 4] == UINT32_C(0x1234) &&
              words[RESET_CONTROL / 4] == 0 && words[STATUS / 4] == 1,
          "start register sequence mismatch");
    check(opennpux_coral_reset(&device) == 0 &&
              words[RESET_CONTROL / 4] == 1,
          "device reset failed");
    free(registers);
    puts("PASS: Coral asynchronous runtime service tests");
    return 0;
}
