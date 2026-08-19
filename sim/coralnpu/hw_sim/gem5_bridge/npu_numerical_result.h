#ifndef HW_SIM_GEM5_BRIDGE_NPU_NUMERICAL_RESULT_H_
#define HW_SIM_GEM5_BRIDGE_NPU_NUMERICAL_RESULT_H_

#include <stdint.h>

#define OPENNPUX_NPU_NUMERICAL_RESULT_MAGIC UINT32_C(0x5258504e)
#define OPENNPUX_NPU_NUMERICAL_RESULT_VERSION UINT32_C(1)
#define OPENNPUX_NPU_NUMERICAL_RESULT_TEXT_BYTES UINT32_C(64)
#define OPENNPUX_NPU_INFERENCE_SOURCE_SIM_HOST UINT32_C(0x48534e55)

struct opennpux_npu_numerical_result {
  uint32_t magic, version, struct_size, flags;
  uint64_t executable_id;
  uint32_t prompt_checksum, next_token, vocabulary_size, logits_checksum;
  uint32_t input_token_count, token_text_size, model_checksum, logits_count;
  char token_text[OPENNPUX_NPU_NUMERICAL_RESULT_TEXT_BYTES];
  uint64_t reserved;
};

static_assert(sizeof(opennpux_npu_numerical_result) == 128);

#endif
