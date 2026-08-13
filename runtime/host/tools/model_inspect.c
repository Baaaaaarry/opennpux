#include "opennpux/model_package.h"

#include <stdio.h>

int
main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <model.npxm>\n", argv[0]);
        return 2;
    }
    struct opennpux_model_package_info info;
    if (opennpux_model_package_load(argv[1], &info) != 0) {
        perror("model package load");
        return 1;
    }
    printf("model_format=%s\n", info.format);
    printf("model_name=%s\n", info.name);
    printf("model_architecture=%s\n", info.architecture_name);
    printf("model_dtype=%s\n", info.dtype);
    printf("model_layers=%u\n", info.layer_count);
    printf("model_hidden=%u\n", info.hidden_size);
    printf("model_heads=%u\n", info.head_count);
    printf("model_kv_heads=%u\n", info.kv_head_count);
    printf("model_head_dim=%u\n", info.head_dim);
    printf("model_experts=%u\n", info.expert_count);
    printf("model_experts_per_token=%u\n", info.experts_per_token);
    printf("model_moe_intermediate=%u\n", info.moe_intermediate_size);
    printf("model_shared_expert_intermediate=%u\n",
           info.shared_expert_intermediate_size);
    printf("model_tensors=%u\n", info.tensor_count);
    printf("model_shards=%u\n", info.shard_count);
    printf("model_weight_bytes=%llu\n",
           (unsigned long long)info.total_weight_bytes);
    if (opennpux_model_package_validate_shards(argv[1], &info) != 0) {
        perror("model shard validation");
        return 1;
    }
    puts("model_package=PASS");
    return 0;
}
