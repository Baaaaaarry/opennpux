#ifndef HW_SIM_GEM5_BRIDGE_NPU_WEIGHT_QUEUE_H_
#define HW_SIM_GEM5_BRIDGE_NPU_WEIGHT_QUEUE_H_

#include <stdint.h>

#define OPENNPUX_NPU_WEIGHT_QUEUE_MAGIC UINT32_C(0x5158504e)
#define OPENNPUX_NPU_WEIGHT_QUEUE_VERSION UINT32_C(4)
#define OPENNPUX_NPU_PAGE_FAULT_MAGIC UINT32_C(0x4658504e)
#define OPENNPUX_NPU_PAGE_FAULT_VERSION UINT32_C(4)
#define OPENNPUX_NPU_PAGE_FAULT_LAST UINT32_C(1)

#define OPENNPUX_NPU_PAGE_FAULT_EMPTY UINT32_C(0)
#define OPENNPUX_NPU_PAGE_FAULT_PENDING UINT32_C(1)
#define OPENNPUX_NPU_PAGE_FAULT_READY UINT32_C(2)
#define OPENNPUX_NPU_PAGE_FAULT_ERROR UINT32_C(3)

struct opennpux_npu_page_fault {
    uint32_t magic, version, struct_size, state;
    uint64_t sequence;
    uint32_t command_id, shard_index;
    uint64_t file_offset, expert_id;
    uint32_t role_id, component_id;
    uint64_t range_file_offset, range_size;
    uint32_t cache_slot, error_code, page_size, flags;
};

struct opennpux_npu_weight_queue_header {
    uint32_t magic, version, header_size, entry_size;
    uint32_t capacity, flags, reserved0, reserved1;
    uint32_t producer_index, service_index, retire_index;
    uint32_t backpressure_count;
    uint64_t reserved[2];
};

_Static_assert(sizeof(struct opennpux_npu_page_fault) == 88,
               "NPU page fault ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_weight_queue_header) == 64,
               "NPU weight queue header ABI size changed");

#endif
