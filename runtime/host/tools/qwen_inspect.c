#include "opennpux/qwen_model.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

int
main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s <qwen-tiny.npxm>\n", argv[0]);
        return 2;
    }

    struct opennpux_qwen_model_info info;
    if (opennpux_qwen_load_model_info(argv[1], &info) != 0) {
        perror("qwen-load");
        return 1;
    }

    printf("qwen_model=%s\n", info.name);
    printf("qwen_format=%s\n", info.format);
    printf("qwen_version=%" PRIu32 "\n", info.version);
    printf("qwen_layers=%" PRIu32 "\n", info.layer_count);
    printf("qwen_hidden=%" PRIu32 "\n", info.hidden_size);
    printf("qwen_intermediate=%" PRIu32 "\n", info.intermediate_size);
    printf("qwen_heads=%" PRIu32 "\n", info.head_count);
    printf("qwen_head_dim=%" PRIu32 "\n", info.head_dim);
    printf("qwen_vocab=%" PRIu32 "\n", info.vocab_size);
    printf("qwen_prompt_tokens=%" PRIu32 "\n", info.prompt_token_count);
    printf("qwen_operator_count=%" PRIu32 "\n", info.operator_count);
    printf("qwen_ops=%s\n", opennpux_qwen_required_ops_string());
    printf("qwen_op_mask=0x%08" PRIx32 "\n", info.op_mask);
    printf("qwen_next_token=%" PRIu32 "\n", info.next_token);
    printf("qwen_logits_checksum=0x%08" PRIx32 "\n", info.logits_checksum);
    printf("qwen_weight_checksum=0x%08" PRIx32 "\n", info.weight_checksum);
    printf("qwen_inspect=PASS\n");
    return 0;
}
