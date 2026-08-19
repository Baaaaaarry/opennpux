#ifndef HW_SIM_GEM5_BRIDGE_NPU_INFERENCE_IO_H_
#define HW_SIM_GEM5_BRIDGE_NPU_INFERENCE_IO_H_

#include <stdint.h>

#define OPENNPUX_NPU_INFERENCE_IO_MAGIC UINT32_C(0x4f49504e)
#define OPENNPUX_NPU_INFERENCE_IO_VERSION UINT32_C(1)
#define OPENNPUX_NPU_INFERENCE_PROMPT_BYTES UINT32_C(128)
#define OPENNPUX_NPU_INFERENCE_INLINE_TOKENS UINT32_C(12)
#define OPENNPUX_NPU_INFERENCE_REUSE_DECODE_WEIGHTS UINT32_C(1)

enum opennpux_npu_inference_state {
  OPENNPUX_NPU_INFERENCE_PENDING = 0,
  OPENNPUX_NPU_INFERENCE_RUNNING = 1,
  OPENNPUX_NPU_INFERENCE_COMPLETE = 2,
  OPENNPUX_NPU_INFERENCE_ERROR = 3,
};

enum opennpux_npu_inference_mode {
  OPENNPUX_NPU_INFERENCE_MODE_FUNCTIONAL = 1,
  OPENNPUX_NPU_INFERENCE_MODE_NUMERICAL = 2,
};

struct opennpux_npu_inference_io {
  uint32_t magic;
  uint32_t version;
  uint32_t struct_size;
  uint32_t state;
  uint32_t error;
  uint32_t mode;
  uint32_t prompt_size;
  uint32_t prompt_checksum;
  uint32_t vocabulary_size;
  uint32_t max_new_tokens;
  uint32_t input_token_count;
  uint32_t next_token;
  uint32_t completed_commands;
  uint32_t result_checksum;
  uint64_t modeled_cycles;
  char prompt[OPENNPUX_NPU_INFERENCE_PROMPT_BYTES];
  uint32_t reserved[16];
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_inference_io) == 256);
#else
_Static_assert(sizeof(struct opennpux_npu_inference_io) == 256,
               "NPU inference I/O ABI size changed");
#endif

#endif  // HW_SIM_GEM5_BRIDGE_NPU_INFERENCE_IO_H_
