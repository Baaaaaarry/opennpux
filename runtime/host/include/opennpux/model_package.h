#ifndef OPENNPUX_MODEL_PACKAGE_H
#define OPENNPUX_MODEL_PACKAGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPENNPUX_MODEL_PACKAGE_FORMAT "OPENNPUX_MODEL_PACKAGE_V2"
#define OPENNPUX_MODEL_PACKAGE_VERSION UINT32_C(2)
#define OPENNPUX_MODEL_MAX_SHARDS UINT32_C(256)
#define OPENNPUX_MODEL_PATH_SIZE UINT32_C(256)
#define OPENNPUX_MODEL_TENSOR_INDEX_MAGIC UINT32_C(0x5458504e)
#define OPENNPUX_MODEL_TENSOR_INDEX_VERSION UINT32_C(1)
#define OPENNPUX_MODEL_TENSOR_NAME_SIZE UINT32_C(160)
#define OPENNPUX_MODEL_TENSOR_MAX_DIMS UINT32_C(8)

enum opennpux_model_architecture {
    OPENNPUX_MODEL_ARCH_UNKNOWN = 0,
    OPENNPUX_MODEL_ARCH_QWEN = 1,
};

enum opennpux_model_weight_format {
    OPENNPUX_MODEL_WEIGHTS_UNKNOWN = 0,
    OPENNPUX_MODEL_WEIGHTS_SAFETENSORS = 1,
};

struct opennpux_model_shard {
    char path[OPENNPUX_MODEL_PATH_SIZE];
    uint64_t size;
};

struct opennpux_model_package_info {
    char format[32];
    char name[96];
    char architecture_name[64];
    char dtype[24];
    char tensor_index_path[OPENNPUX_MODEL_PATH_SIZE];
    uint32_t version;
    uint32_t architecture;
    uint32_t weight_format;
    uint32_t layer_count;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t head_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t max_sequence_length;
    uint32_t expert_count;
    uint32_t experts_per_token;
    uint32_t moe_intermediate_size;
    uint32_t shared_expert_intermediate_size;
    uint32_t tensor_count;
    uint32_t shard_count;
    uint64_t total_weight_bytes;
    uint32_t required_op_mask;
    struct opennpux_model_shard shards[OPENNPUX_MODEL_MAX_SHARDS];
};

struct opennpux_model_tensor_index_header {
    uint32_t magic;
    uint32_t version;
    uint32_t header_size;
    uint32_t record_size;
    uint32_t tensor_count;
    uint32_t reserved[3];
};

struct opennpux_model_tensor_record {
    char name[OPENNPUX_MODEL_TENSOR_NAME_SIZE];
    uint32_t shard_index;
    uint32_t dtype;
    uint32_t rank;
    uint32_t reserved;
    uint64_t dims[OPENNPUX_MODEL_TENSOR_MAX_DIMS];
    uint64_t data_offset;
    uint64_t data_size;
};

int opennpux_model_package_load(
    const char *manifest_path, struct opennpux_model_package_info *info);
int opennpux_model_package_validate_shards(
    const char *manifest_path,
    const struct opennpux_model_package_info *info);
int opennpux_model_package_find_tensor(
    const char *manifest_path,
    const struct opennpux_model_package_info *info, const char *tensor_name,
    struct opennpux_model_tensor_record *tensor);
int opennpux_model_package_read_tensor(
    const char *manifest_path,
    const struct opennpux_model_package_info *info,
    const struct opennpux_model_tensor_record *tensor, uint64_t offset,
    void *buffer, uint64_t size);

#ifdef __cplusplus
}
#endif

#endif
