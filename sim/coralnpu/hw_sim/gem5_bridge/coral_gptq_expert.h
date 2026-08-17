#ifndef HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_EXPERT_H_
#define HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_EXPERT_H_

#include <stdint.h>

#define CORAL_GPTQ_EXPERT_MAGIC UINT32_C(0x45583447)
#define CORAL_GPTQ_EXPERT_VERSION UINT32_C(1)

enum coral_gptq_expert_state {
  CORAL_GPTQ_EXPERT_PENDING = 0,
  CORAL_GPTQ_EXPERT_RUNNING = 1,
  CORAL_GPTQ_EXPERT_COMPLETE = 2,
  CORAL_GPTQ_EXPERT_ERROR = 3,
};

struct coral_gptq_projection_weights {
  uint32_t qweight_address;
  uint32_t qzeros_address;
  uint32_t scales_address;
  uint32_t g_idx_address;
  uint32_t scale_data_type;
  uint32_t reserved;
};

/*
 * Model-independent gated MLP request:
 *   gate = GPTQ(input, gate_weights)
 *   up = GPTQ(input, up_weights)
 *   activated = SiLU(gate) * up
 *   output = GPTQ(activated, down_weights)
 */
struct coral_gptq_expert_request {
  uint32_t magic;
  uint32_t version;
  uint32_t struct_size;
  uint32_t state;
  uint32_t error;
  uint32_t rows;
  uint32_t hidden_columns;
  uint32_t intermediate_columns;
  uint32_t group_size;
  uint32_t zero_bias;
  uint32_t input_address;
  uint32_t gate_output_address;
  uint32_t up_output_address;
  uint32_t activated_address;
  uint32_t output_address;
  struct coral_gptq_projection_weights gate;
  struct coral_gptq_projection_weights up;
  struct coral_gptq_projection_weights down;
  uint32_t gate_checksum;
  uint32_t up_checksum;
  uint32_t activated_checksum;
  uint32_t output_checksum;
  uint32_t stats_alignment;
  uint64_t operations;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t modeled_cycles;
  uint64_t gate_cycles;
  uint64_t up_cycles;
  uint64_t activation_cycles;
  uint64_t down_cycles;
  uint32_t reserved[10];
};

#ifdef __cplusplus
static_assert(sizeof(struct coral_gptq_projection_weights) == 24);
static_assert(sizeof(struct coral_gptq_expert_request) == 256);
#else
_Static_assert(sizeof(struct coral_gptq_projection_weights) == 24,
               "GPTQ projection weights ABI changed");
_Static_assert(sizeof(struct coral_gptq_expert_request) == 256,
               "GPTQ expert request ABI changed");
#endif

#endif  // HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_EXPERT_H_
