#include "opennpux/model_package.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
check(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message);
        _Exit(1);
    }
}

int
main(int argc, char **argv)
{
    check(argc == 2, "usage: model_package_test <model.npxm>");
    struct opennpux_model_package_info info;
    check(opennpux_model_package_load(argv[1], &info) == 0,
          "failed to load v2 package");
    check(strcmp(info.format, OPENNPUX_MODEL_PACKAGE_FORMAT) == 0,
          "format mismatch");
    check(info.version == 2, "version mismatch");
    check(info.architecture == OPENNPUX_MODEL_ARCH_QWEN,
          "architecture mismatch");
    check(info.layer_count == 2 && info.hidden_size == 18,
          "model dimensions mismatch");
    check(info.head_count == 4 && info.kv_head_count == 2,
          "attention dimensions mismatch");
    check(info.head_dim == 6, "explicit head dimension mismatch");
    check(info.expert_count == 8 && info.experts_per_token == 2,
          "MoE topology mismatch");
    check(strcmp(info.quantization_method, "gptq") == 0 &&
              info.quantization_bits == 4 &&
              info.quantization_group_size == 128,
          "GPTQ configuration mismatch");
    check(info.tensor_count == 6 && info.shard_count == 2,
          "tensor/shard count mismatch");
    check(info.total_weight_bytes > 48, "weight byte count missing");
    check(opennpux_model_package_validate_shards(argv[1], &info) == 0,
          "shard validation failed");
    struct opennpux_model_tensor_record tensor;
    check(opennpux_model_package_find_tensor(
              argv[1], &info,
              "model.language_model.layers.1.mlp.router.weight",
                                             &tensor) == 0,
          "tensor lookup failed");
    check(tensor.shard_index == 1 && tensor.rank == 1 && tensor.dims[0] == 8,
          "tensor metadata mismatch");
    unsigned char bytes[4];
    check(opennpux_model_package_read_tensor(argv[1], &info, &tensor, 2,
                                             bytes, sizeof(bytes)) == 0,
          "tensor range read failed");
    check(bytes[0] == 22 && bytes[1] == 23 && bytes[2] == 24 && bytes[3] == 25,
          "tensor range data mismatch");
    puts("PASS: model package host unit tests");
    return 0;
}
