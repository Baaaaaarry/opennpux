#ifndef HW_SIM_GEM5_BRIDGE_NPU_WEIGHT_RESIDENCY_H_
#define HW_SIM_GEM5_BRIDGE_NPU_WEIGHT_RESIDENCY_H_

#include <stdint.h>

#define OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC UINT32_C(0x5358504e)
#define OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION UINT32_C(1)
#define OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID UINT32_C(1)

struct opennpux_npu_weight_residency_header {
  uint32_t magic, version, header_size, record_size;
  uint32_t capacity, valid_records;
  uint64_t generation;
  uint64_t reserved[4];
};

struct opennpux_npu_weight_residency_record {
  uint32_t command_id, role_id, component_id, shard_index;
  uint64_t expert_id, range_file_offset, range_size, page_file_offset;
  uint32_t cache_slot, page_size, flags, reserved;
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_weight_residency_header) == 64);
static_assert(sizeof(struct opennpux_npu_weight_residency_record) == 64);
#else
_Static_assert(sizeof(struct opennpux_npu_weight_residency_header) == 64,
               "NPU residency header ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_weight_residency_record) == 64,
               "NPU residency record ABI size changed");
#endif

#endif
