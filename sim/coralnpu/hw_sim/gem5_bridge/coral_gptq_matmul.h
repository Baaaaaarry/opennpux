#ifndef HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_MATMUL_H_
#define HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_MATMUL_H_

#include <stdint.h>

#define CORAL_GPTQ_MATMUL_MAGIC UINT32_C(0x514d3447)
#define CORAL_GPTQ_MATMUL_VERSION UINT32_C(1)

enum coral_gptq_matmul_state {
  CORAL_GPTQ_MATMUL_PENDING = 0,
  CORAL_GPTQ_MATMUL_RUNNING = 1,
  CORAL_GPTQ_MATMUL_COMPLETE = 2,
  CORAL_GPTQ_MATMUL_ERROR = 3,
};

struct coral_gptq_matmul_request {
  uint32_t magic;
  uint32_t version;
  uint32_t struct_size;
  uint32_t state;
  uint32_t error;
  uint32_t rows;
  uint32_t input_columns;
  uint32_t output_columns;
  uint32_t group_size;
  uint32_t zero_bias;
  uint32_t input_address;
  uint32_t qweight_address;
  uint32_t qzeros_address;
  uint32_t scales_address;
  uint32_t g_idx_address;
  uint32_t output_address;
  uint64_t operations;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t modeled_cycles;
  uint32_t output_checksum;
  uint32_t reserved[7];
};

#ifdef __cplusplus
static_assert(sizeof(struct coral_gptq_matmul_request) == 128);
#else
_Static_assert(sizeof(struct coral_gptq_matmul_request) == 128,
               "GPTQ MatMul request ABI changed");
#endif

#endif  // HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_MATMUL_H_
