#ifndef OPENNPUX_TVM_BYOC_MODULE_H
#define OPENNPUX_TVM_BYOC_MODULE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_TVM_MODULE_MAGIC UINT32_C(0x4d47584e)
#define OPENNPUX_TVM_MODULE_VERSION UINT32_C(1)
#define OPENNPUX_TVM_MODULE_HOST_RELU UINT32_C(1)

struct opennpux_tvm_module_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t total_size;
    uint32_t region_count;
    uint32_t edge_count;
    uint32_t host_binding_count;
    uint32_t host_operation_count;
    uint32_t output_count;
    uint32_t region_record_size;
    uint32_t edge_record_size;
    uint32_t host_binding_record_size;
    uint32_t host_operation_record_size;
    uint32_t output_record_size;
    uint32_t payload_offset;
    uint32_t reserved;
};

struct opennpux_tvm_module_region {
    uint32_t graph_offset;
    uint32_t graph_size;
    uint32_t arena_offset;
    uint32_t arena_size;
    uint32_t output_offset;
    uint32_t output_bytes;
    uint32_t flags;
    uint32_t reserved;
};

struct opennpux_tvm_module_edge {
    uint32_t from_region;
    uint32_t to_region;
    uint32_t source_offset;
    uint32_t target_offset;
    uint32_t bytes;
    uint32_t reserved;
};

struct opennpux_tvm_module_host_binding {
    uint32_t from_region;
    uint32_t to_region;
    uint32_t source_offset;
    uint32_t target_offset;
    uint32_t bytes;
    uint32_t first_operation;
    uint32_t operation_count;
};

struct opennpux_tvm_module_host_operation {
    uint32_t opcode;
    uint32_t flags;
};

struct opennpux_tvm_module_output {
    uint32_t region;
    uint32_t offset;
    uint32_t bytes;
    uint32_t name_checksum;
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_tvm_module_header) == 64,
              "TVM module header ABI changed");
static_assert(sizeof(struct opennpux_tvm_module_region) == 32,
              "TVM module region ABI changed");
static_assert(sizeof(struct opennpux_tvm_module_edge) == 24,
              "TVM module edge ABI changed");
static_assert(sizeof(struct opennpux_tvm_module_host_binding) == 28,
              "TVM module Host binding ABI changed");
static_assert(sizeof(struct opennpux_tvm_module_host_operation) == 8,
              "TVM module Host operation ABI changed");
static_assert(sizeof(struct opennpux_tvm_module_output) == 16,
              "TVM module output ABI changed");
#else
_Static_assert(sizeof(struct opennpux_tvm_module_header) == 64,
               "TVM module header ABI changed");
_Static_assert(sizeof(struct opennpux_tvm_module_region) == 32,
               "TVM module region ABI changed");
_Static_assert(sizeof(struct opennpux_tvm_module_edge) == 24,
               "TVM module edge ABI changed");
_Static_assert(sizeof(struct opennpux_tvm_module_host_binding) == 28,
               "TVM module Host binding ABI changed");
_Static_assert(sizeof(struct opennpux_tvm_module_host_operation) == 8,
               "TVM module Host operation ABI changed");
_Static_assert(sizeof(struct opennpux_tvm_module_output) == 16,
               "TVM module output ABI changed");
#endif

#ifdef __cplusplus
}
#endif

#endif
