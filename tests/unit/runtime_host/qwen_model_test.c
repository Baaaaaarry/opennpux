#include "opennpux/qwen_model.h"

#include <stdint.h>
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
    check(argc == 2, "usage: qwen_model_test <qwen-tiny.npxm>");

    struct opennpux_qwen_model_info info;
    check(opennpux_qwen_load_model_info(argv[1], &info) == 0,
          "failed to load qwen tiny package");
    check(strcmp(info.format, OPENNPUX_QWEN_TINY_FORMAT) == 0,
          "unexpected qwen format");
    check(strcmp(info.name, "qwen-tiny-synthetic") == 0,
          "unexpected qwen model name");
    check(info.version == 1, "unexpected qwen version");
    check(info.layer_count == 1, "unexpected qwen layer count");
    check(info.vocab_size == 16, "unexpected qwen vocab size");
    check(info.hidden_size == 8, "unexpected qwen hidden size");
    check(info.intermediate_size == 12, "unexpected qwen intermediate size");
    check(info.head_count == 2, "unexpected qwen head count");
    check(info.head_dim == 4, "unexpected qwen head dim");
    check(info.prompt_token_count == 4, "unexpected qwen prompt size");
    check(info.operator_count == 19, "unexpected qwen operator count");
    check(info.next_token == 7, "unexpected qwen next token");
    check(info.logits_checksum == 0x829e9f00,
          "unexpected qwen logits checksum");
    check(info.weight_checksum == 0x98a39dcc,
          "unexpected qwen weight checksum");
    check((info.op_mask & opennpux_qwen_required_op_mask()) ==
              opennpux_qwen_required_op_mask(),
          "missing required qwen op");

    struct opennpux_qwen_run_result result;
    check(opennpux_qwen_run_golden(argv[1], &result) == 0,
          "failed to run qwen golden path");
    check(result.prefill_pass == 1, "qwen prefill did not pass");
    check(result.decode_pass == 1, "qwen decode did not pass");
    check(result.completed_operators == 19,
          "unexpected qwen completed operator count");
    check(result.next_token == 7, "unexpected qwen run next token");
    check(result.output_checksum == 0x829e9f00,
          "unexpected qwen run output checksum");

    puts("PASS: qwen model host unit tests");
    puts("qwen_loader=PASS");
    return 0;
}
