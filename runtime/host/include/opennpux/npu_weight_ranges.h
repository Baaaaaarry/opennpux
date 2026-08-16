#ifndef OPENNPUX_NPU_WEIGHT_RANGES_H
#define OPENNPUX_NPU_WEIGHT_RANGES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_NPU_WEIGHT_RANGE_MAGIC UINT32_C(0x5258504e)
#define OPENNPUX_NPU_WEIGHT_RANGE_VERSION UINT32_C(1)
#define OPENNPUX_NPU_WEIGHT_RANGE_HEADER_SIZE UINT32_C(64)
#define OPENNPUX_NPU_WEIGHT_RANGE_RECORD_SIZE UINT32_C(64)
#define OPENNPUX_NPU_WEIGHT_EXPERT_NONE UINT64_MAX

/* Low 32 bits of the package compiler's SHA-256 semantic identifiers. */
#define OPENNPUX_NPU_WEIGHT_COMPONENT_WEIGHT UINT32_C(0x8fdf4408)
#define OPENNPUX_NPU_WEIGHT_COMPONENT_BIAS UINT32_C(0xe82448fa)
#define OPENNPUX_NPU_WEIGHT_COMPONENT_QWEIGHT UINT32_C(0x40406979)
#define OPENNPUX_NPU_WEIGHT_COMPONENT_QZEROS UINT32_C(0x5d11cedb)
#define OPENNPUX_NPU_WEIGHT_COMPONENT_SCALES UINT32_C(0x76c0006c)
#define OPENNPUX_NPU_WEIGHT_COMPONENT_G_IDX UINT32_C(0x1b2ede4b)
#define OPENNPUX_NPU_WEIGHT_COMPONENT_OTHER UINT32_C(0x108a29d9)

#define OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_PROJ UINT32_C(0xf91b83dd)
#define OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_O_PROJ UINT32_C(0xfa6afc40)
#define OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ UINT32_C(0xa3f281ba)
#define OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_V_PROJ UINT32_C(0x77364ab0)
#define OPENNPUX_NPU_WEIGHT_ROLE_EMBEDDING UINT32_C(0x560158aa)
#define OPENNPUX_NPU_WEIGHT_ROLE_LM_HEAD UINT32_C(0xe812c442)
#define OPENNPUX_NPU_WEIGHT_ROLE_ROUTED_EXPERT UINT32_C(0x676a07a0)
#define OPENNPUX_NPU_WEIGHT_ROLE_ROUTER UINT32_C(0x0456c974)
#define OPENNPUX_NPU_WEIGHT_ROLE_SHARED_EXPERT UINT32_C(0x06fc7f70)

struct opennpux_npu_weight_range_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t record_size;
    uint32_t command_count;
    uint32_t range_count;
    uint32_t shard_count;
    uint32_t checksum;
    uint64_t record_offset;
    uint64_t total_size;
    uint64_t executable_id;
    uint64_t reserved;
};

struct opennpux_npu_weight_range_record {
    uint32_t command_id;
    uint32_t shard_index;
    uint32_t role_id;
    uint32_t component_id;
    uint64_t file_offset;
    uint64_t byte_size;
    uint64_t tensor_id;
    uint64_t parameter_symbol;
    uint64_t expert_id;
    uint64_t flags;
};

struct opennpux_npu_weight_ranges {
    void *storage;
    size_t storage_size;
    const struct opennpux_npu_weight_range_header *header;
    const struct opennpux_npu_weight_range_record *records;
};

struct opennpux_npu_gptq_weight_ranges {
    const struct opennpux_npu_weight_range_record *qweight;
    const struct opennpux_npu_weight_range_record *qzeros;
    const struct opennpux_npu_weight_range_record *scales;
    const struct opennpux_npu_weight_range_record *g_idx;
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_weight_range_header) ==
              OPENNPUX_NPU_WEIGHT_RANGE_HEADER_SIZE);
static_assert(sizeof(struct opennpux_npu_weight_range_record) ==
              OPENNPUX_NPU_WEIGHT_RANGE_RECORD_SIZE);
#else
_Static_assert(sizeof(struct opennpux_npu_weight_range_header) ==
               OPENNPUX_NPU_WEIGHT_RANGE_HEADER_SIZE,
               "NPU weight range header ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_weight_range_record) ==
               OPENNPUX_NPU_WEIGHT_RANGE_RECORD_SIZE,
               "NPU weight range record ABI size changed");
#endif

int opennpux_npu_weight_ranges_load(
    const char *path, struct opennpux_npu_weight_ranges *ranges);
void opennpux_npu_weight_ranges_unload(
    struct opennpux_npu_weight_ranges *ranges);
int opennpux_npu_weight_ranges_for_command(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    const struct opennpux_npu_weight_range_record **records,
    uint32_t *record_count);
int opennpux_npu_weight_range_find(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    uint32_t role_id, uint32_t component_id, uint64_t expert_id,
    const struct opennpux_npu_weight_range_record **record);
int opennpux_npu_weight_ranges_find_gptq(
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    uint32_t role_id, uint64_t expert_id,
    struct opennpux_npu_gptq_weight_ranges *gptq);

#ifdef __cplusplus
}
#endif

#endif
