#include <stddef.h>
#include <stdint.h>

#include "hw_sim/gem5_bridge/qwen_tcb.h"

#define CORAL_EXTMEM_BASE UINT32_C(0x20000000)

static uint32_t
fnv1a32_tcb(const volatile uint8_t *bytes, uint32_t size)
{
    uint32_t checksum = UINT32_C(2166136261);
    const uint32_t checksum_begin =
        (uint32_t)offsetof(struct opennpux_qwen_tcb_header, tcb_checksum);
    const uint32_t checksum_end = checksum_begin + sizeof(uint32_t);
    for (uint32_t index = 0; index < size; ++index) {
        uint8_t value = 0;
        if (index < checksum_begin || index >= checksum_end) {
            value = bytes[index];
        }
        checksum ^= value;
        checksum *= UINT32_C(16777619);
    }
    return checksum;
}

static void
finish(volatile struct opennpux_qwen_tcb_header *header, uint32_t state,
       uint32_t error)
{
    header->tcb_error = error;
    header->tcb_state = state;
}

static uint32_t
mix32(uint32_t hash, uint32_t value)
{
    hash ^= value;
    hash *= UINT32_C(16777619);
    return hash;
}

int
main(void)
{
    volatile uint8_t *shared = (volatile uint8_t *)CORAL_EXTMEM_BASE;
    volatile struct opennpux_qwen_tcb_header *header =
        (volatile struct opennpux_qwen_tcb_header *)shared;

    if (header->magic != OPENNPUX_QWEN_TCB_MAGIC ||
        header->version != OPENNPUX_QWEN_TCB_VERSION ||
        header->header_size != sizeof(struct opennpux_qwen_tcb_header) ||
        header->total_size < sizeof(struct opennpux_qwen_tcb_header) ||
        header->total_size > OPENNPUX_QWEN_TCB_MAX_SIZE ||
        header->op_count == 0 ||
        header->op_count > OPENNPUX_QWEN_MAX_OPS) {
        finish(header, OPENNPUX_QWEN_TCB_STATE_ERROR,
               OPENNPUX_QWEN_TCB_ERROR_ABI);
        return 1;
    }

    const uint32_t expected_size =
        sizeof(struct opennpux_qwen_tcb_header) +
        header->op_count * sizeof(struct opennpux_qwen_tcb_op);
    if (header->total_size != expected_size) {
        finish(header, OPENNPUX_QWEN_TCB_STATE_ERROR,
               OPENNPUX_QWEN_TCB_ERROR_BOUNDS);
        return 1;
    }

    const uint32_t checksum = fnv1a32_tcb(shared, header->total_size);
    if (checksum != header->tcb_checksum) {
        header->device_checksum = checksum;
        finish(header, OPENNPUX_QWEN_TCB_STATE_ERROR,
               OPENNPUX_QWEN_TCB_ERROR_CHECKSUM);
        return 1;
    }

    header->tcb_state = OPENNPUX_QWEN_TCB_STATE_RUNNING;
    header->tcb_error = OPENNPUX_QWEN_TCB_ERROR_NONE;

    volatile struct opennpux_qwen_tcb_op *ops =
        (volatile struct opennpux_qwen_tcb_op *)(
            shared + sizeof(struct opennpux_qwen_tcb_header));
    uint64_t modeled_cycles = 0;
    uint32_t op_mask = 0;
    uint32_t trace_checksum = UINT32_C(2166136261);
    for (uint32_t index = 0; index < header->op_count; ++index) {
        if (ops[index].index != index ||
            ops[index].kind >= OPENNPUX_QWEN_OP_KIND_COUNT ||
            ops[index].rank > OPENNPUX_QWEN_OP_MAX_DIMS) {
            header->device_completed_ops = index;
            finish(header, OPENNPUX_QWEN_TCB_STATE_ERROR,
                   OPENNPUX_QWEN_TCB_ERROR_OPERATOR);
            return 1;
        }
        op_mask |= UINT32_C(1) << ops[index].kind;
        trace_checksum = mix32(trace_checksum, ops[index].index);
        trace_checksum = mix32(trace_checksum, ops[index].kind);
        trace_checksum = mix32(trace_checksum, ops[index].layer);
        trace_checksum = mix32(trace_checksum, ops[index].rank);
        trace_checksum = mix32(trace_checksum, (uint32_t)ops[index].operations);
        trace_checksum = mix32(trace_checksum,
                               (uint32_t)(ops[index].operations >> 32));
        trace_checksum = mix32(trace_checksum,
                               (uint32_t)ops[index].modeled_cycles);
        trace_checksum = mix32(trace_checksum,
                               (uint32_t)(ops[index].modeled_cycles >> 32));
        modeled_cycles += ops[index].modeled_cycles;
        ops[index].reserved[0] = OPENNPUX_QWEN_TCB_OP_COMPLETE_MAGIC;
        ops[index].reserved[1] = ops[index].kind;
        header->device_completed_ops = index + 1;
    }

    header->device_checksum = checksum;
    header->device_modeled_cycles = modeled_cycles;
    header->device_op_mask = op_mask;
    header->device_trace_checksum = trace_checksum;
    finish(header, OPENNPUX_QWEN_TCB_STATE_COMPLETE,
           OPENNPUX_QWEN_TCB_ERROR_NONE);
    return 0;
}
