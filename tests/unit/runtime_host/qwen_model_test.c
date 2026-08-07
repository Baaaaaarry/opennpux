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

    check(opennpux_qwen_run_hybrid_sim(argv[1], &result) == 0,
          "failed to run qwen hybrid simulation path");
    check(result.prefill_pass == 1, "qwen hybrid prefill did not pass");
    check(result.decode_pass == 1, "qwen hybrid decode did not pass");
    check(result.operation_count != 0, "qwen hybrid operation count missing");
    check(result.modeled_cycles != 0, "qwen hybrid modeled cycles missing");
    check(result.bytes_read != 0, "qwen hybrid read bytes missing");
    check(result.bytes_written != 0, "qwen hybrid write bytes missing");
    check(result.op_counts[1] == 8, "qwen hybrid matmul count mismatch");
    check(result.next_token == 7, "unexpected qwen hybrid next token");
    check(result.output_checksum == 0x829e9f00,
          "unexpected qwen hybrid output checksum");
    check(result.tcb_size != 0, "qwen tcb size missing");
    check(result.tcb_checksum != 0, "qwen tcb checksum missing");

    uint8_t tcb[OPENNPUX_QWEN_TCB_MAX_SIZE];
    uint32_t tcb_size = 0;
    uint32_t tcb_checksum = 0;
    check(opennpux_qwen_build_tcb(&result, tcb, sizeof(tcb),
                                  &tcb_size, &tcb_checksum) == 0,
          "failed to build qwen tcb");
    check(tcb_size == result.tcb_size, "qwen tcb size mismatch");
    check(tcb_checksum == result.tcb_checksum, "qwen tcb checksum mismatch");
    const struct opennpux_qwen_tcb_header *header =
        (const struct opennpux_qwen_tcb_header *)tcb;
    const struct opennpux_qwen_tcb_op *ops =
        (const struct opennpux_qwen_tcb_op *)(const void *)(
            tcb + sizeof(*header));
    check(header->magic == OPENNPUX_QWEN_TCB_MAGIC, "qwen tcb magic mismatch");
    check(header->version == OPENNPUX_QWEN_TCB_VERSION,
          "qwen tcb version mismatch");
    check(header->op_count == 19, "qwen tcb op count mismatch");
    check(header->logits_checksum == 0x829e9f00,
          "qwen tcb logits checksum mismatch");
    check(header->tcb_state == OPENNPUX_QWEN_TCB_STATE_PENDING,
          "qwen tcb initial state mismatch");
    check(header->tcb_error == OPENNPUX_QWEN_TCB_ERROR_NONE,
          "qwen tcb initial error mismatch");
    check(header->device_completed_ops == 0,
          "qwen tcb initial device completion mismatch");
    check(ops[2].kind == 1, "qwen tcb matmul kind mismatch");
    check(ops[2].rank == 3, "qwen tcb matmul rank mismatch");
    check(ops[2].dims[0] == 4 && ops[2].dims[1] == 8 &&
              ops[2].dims[2] == 8,
          "qwen tcb matmul dims mismatch");
    check(ops[2].input_offset >= OPENNPUX_QWEN_TCB_TENSOR_BASE,
          "qwen tcb tensor offset mismatch");

    puts("PASS: qwen model host unit tests");
    puts("qwen_loader=PASS");
    return 0;
}
