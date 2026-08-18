#ifndef HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_PAGED_H_
#define HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_PAGED_H_

#include <stdint.h>

#include "hw_sim/gem5_bridge/npu_submission.h"

#define CORAL_GPTQ_PAGED_MAGIC UINT32_C(0x50475150)
#define CORAL_GPTQ_PAGED_VERSION UINT32_C(1)
#define CORAL_GPTQ_PAGED_MAX_SPANS UINT32_C(64)

enum coral_gptq_paged_state {
  CORAL_GPTQ_PAGED_PENDING = 0,
  CORAL_GPTQ_PAGED_RUNNING = 1,
  CORAL_GPTQ_PAGED_COMPLETE = 2,
  CORAL_GPTQ_PAGED_ERROR = 3,
};

// One command's weights are paged before this request is submitted. The
// bridge resolves semantic qweight/qzeros/scales/g_idx spans from residency
// records instead of requiring a contiguous multi-gigabyte weight buffer.
struct coral_gptq_paged_request {
  uint32_t magic;
  uint32_t version;
  uint32_t struct_size;
  uint32_t state;
  uint32_t error;
  uint32_t command_id;
  uint32_t role_id;
  uint32_t rows;
  uint64_t expert_id;
  struct opennpux_npu_operator_parameters parameters;
  uint32_t input_address;
  uint32_t input_size;
  uint32_t output_address;
  uint32_t output_size;
  uint32_t residency_address;
  uint32_t residency_size;
  uint32_t cache_address;
  uint32_t cache_size;
  uint32_t output_tile_columns;
  uint32_t span_count;
  uint32_t output_checksum;
  uint32_t reserved0;
  uint64_t operations;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t modeled_cycles;
};

#if defined(__cplusplus)
static_assert(sizeof(struct coral_gptq_paged_request) == 184);
#else
_Static_assert(sizeof(struct coral_gptq_paged_request) == 184,
               "paged GPTQ request ABI size changed");
#endif

#endif  // HW_SIM_GEM5_BRIDGE_CORAL_GPTQ_PAGED_H_
