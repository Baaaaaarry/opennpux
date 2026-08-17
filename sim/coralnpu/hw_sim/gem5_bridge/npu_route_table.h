#ifndef OPENNPUX_NPU_ROUTE_TABLE_H
#define OPENNPUX_NPU_ROUTE_TABLE_H

#include <stddef.h>
#include <stdint.h>

#define OPENNPUX_NPU_ROUTE_TABLE_MAGIC UINT32_C(0x5258504e)
#define OPENNPUX_NPU_ROUTE_TABLE_VERSION UINT32_C(1)
#define OPENNPUX_NPU_ROUTE_TABLE_HEADER_SIZE UINT32_C(32)
#define OPENNPUX_NPU_ROUTE_RECORD_SIZE UINT32_C(16)
#define OPENNPUX_NPU_MAX_ACTIVE_EXPERTS UINT32_C(64)

struct opennpux_npu_route_table_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t record_size;
    uint32_t record_count;
    uint32_t checksum;
    uint32_t total_size;
    uint32_t reserved;
};

struct opennpux_npu_route_record {
    uint32_t expert_id;
    uint32_t reserved;
    float logit;
    float weight;
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_route_table_header) ==
              OPENNPUX_NPU_ROUTE_TABLE_HEADER_SIZE);
static_assert(sizeof(struct opennpux_npu_route_record) ==
              OPENNPUX_NPU_ROUTE_RECORD_SIZE);
#else
_Static_assert(sizeof(struct opennpux_npu_route_table_header) ==
               OPENNPUX_NPU_ROUTE_TABLE_HEADER_SIZE,
               "NPU route table header ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_route_record) ==
               OPENNPUX_NPU_ROUTE_RECORD_SIZE,
               "NPU route record ABI size changed");
#endif

#endif
