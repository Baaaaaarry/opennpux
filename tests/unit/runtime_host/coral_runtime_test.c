#include "opennpux/coral_runtime.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        _Exit(1);
    }
}

int
main(void)
{
    uint64_t value = 0;
    check(opennpux_coral_parse_u64("0x1d000000", &value) == 0,
          "hex parse failed");
    check(value == 0x1d000000, "hex parse returned wrong value");
    check(opennpux_coral_parse_u64("1234", &value) == 0,
          "decimal parse failed");
    check(value == 1234, "decimal parse returned wrong value");
    check(opennpux_coral_parse_u64("12bad", &value) != 0,
          "invalid parse succeeded");

    check(opennpux_coral_decode_backend(0x4e505501) ==
              OPENNPUX_CORAL_BACKEND_STAGE_A,
          "stage-a backend decode failed");
    check(opennpux_coral_decode_backend(0x4e505502) ==
              OPENNPUX_CORAL_BACKEND_VERILATED,
          "verilated backend decode failed");
    check(opennpux_coral_decode_backend(0) ==
              OPENNPUX_CORAL_BACKEND_UNKNOWN,
          "unknown backend decode failed");
    check(opennpux_coral_backend_name(OPENNPUX_CORAL_BACKEND_VERILATED)[0] ==
              'v',
          "backend name failed");
    check(opennpux_coral_transport_name(OPENNPUX_CORAL_TRANSPORT_DEVMEM)[0] ==
              'd',
          "devmem transport name failed");
    check(opennpux_coral_transport_name(OPENNPUX_CORAL_TRANSPORT_DRIVER)[0] ==
              'd',
          "driver transport name failed");

    check(OPENNPUX_CORAL_ABI_VERSION == 1, "unexpected driver ABI version");
    check(sizeof(struct opennpux_coral_ioc_caps) == 32,
          "driver capabilities ABI size changed");
    check(sizeof(struct opennpux_coral_ioc_start) == 8,
          "driver start ABI size changed");
    check((OPENNPUX_CORAL_FEATURE_SHARED_MMAP |
           OPENNPUX_CORAL_FEATURE_ASYNC_START |
           OPENNPUX_CORAL_FEATURE_POLL_COMPLETION |
           OPENNPUX_CORAL_FEATURE_RESET) == 0xf,
          "driver feature bits changed");

    check(opennpux_coral_check_shared_u32_access(16, 0) == 0,
          "offset 0 rejected");
    check(opennpux_coral_check_shared_u32_access(16, 12) == 0,
          "last aligned word rejected");
    check(opennpux_coral_check_shared_u32_access(16, 2) != 0 &&
              errno == EINVAL,
          "unaligned offset accepted");
    check(opennpux_coral_check_shared_u32_access(16, 16) != 0 &&
              errno == ERANGE,
          "out-of-range offset accepted");

    puts("PASS: coral runtime host unit tests");
    return 0;
}
