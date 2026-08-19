#ifndef HW_SIM_GEM5_BRIDGE_NPU_PAGE_BUNDLE_H_
#define HW_SIM_GEM5_BRIDGE_NPU_PAGE_BUNDLE_H_

#include <stdint.h>

#define OPENNPUX_NPU_PAGE_BUNDLE_MAGIC UINT32_C(0x4258504e)
#define OPENNPUX_NPU_PAGE_BUNDLE_VERSION UINT32_C(1)
#define OPENNPUX_NPU_PAGE_BUNDLE_LAST UINT32_C(1)

struct opennpux_npu_page_bundle_header {
  uint32_t magic, version, header_size, record_size;
  uint32_t transfer_size, record_count, command_count, flags;
  uint64_t payload_bytes;
  uint32_t active_expert_count, max_pages_per_record;
  uint64_t reserved[2];
};

struct opennpux_npu_page_bundle_record {
  uint32_t command_id, shard_index, role_id, component_id;
  uint64_t expert_id, file_offset, range_file_offset, range_size;
  uint32_t page_size, flags;
  uint64_t reserved;
};

static_assert(sizeof(opennpux_npu_page_bundle_header) == 64);
static_assert(sizeof(opennpux_npu_page_bundle_record) == 64);

#endif
