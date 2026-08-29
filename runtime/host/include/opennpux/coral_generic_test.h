#ifndef OPENNPUX_CORAL_GENERIC_TEST_H
#define OPENNPUX_CORAL_GENERIC_TEST_H

#include <stdint.h>

#define OPENNPUX_CORAL_GENERIC_TEST_MAGIC UINT32_C(0x4e504754)
#define OPENNPUX_CORAL_GENERIC_TEST_VERSION UINT32_C(2)
/* Keep control mailboxes in the low, directly mapped shared-window region. */
#define OPENNPUX_CORAL_GENERIC_TEST_MAILBOX_OFFSET UINT32_C(0x00008000)
#define OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT UINT32_C(4)

#define OPENNPUX_CORAL_GENERIC_TEST_STARTED UINT32_C(1)
#define OPENNPUX_CORAL_GENERIC_TEST_COMPLETE UINT32_C(2)
#define OPENNPUX_CORAL_GENERIC_TEST_ERROR UINT32_C(0x80000000)

#define OPENNPUX_CORAL_GENERIC_TEST_ERROR_NONE UINT32_C(0)
#define OPENNPUX_CORAL_GENERIC_TEST_ERROR_SUBMIT UINT32_C(1)
#define OPENNPUX_CORAL_GENERIC_TEST_ERROR_OUTPUT UINT32_C(2)

struct opennpux_coral_generic_test_mailbox {
    uint32_t magic;
    uint32_t version;
    uint32_t state;
    uint32_t error_code;
    uint32_t cycle_low;
    uint32_t cycle_high;
    uint32_t output_count;
    int32_t output[OPENNPUX_CORAL_GENERIC_TEST_OUTPUT_COUNT];
    uint32_t output_checksum;
    uint32_t output_bytes;
    uint64_t operation_count;
    uint64_t bytes_read;
    uint64_t bytes_written;
};

#endif
