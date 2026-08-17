#ifndef OPENNPUX_NPU_GPTQ_WEIGHTS_H
#define OPENNPUX_NPU_GPTQ_WEIGHTS_H

#include "opennpux/model_package.h"
#include "opennpux/npu_submission.h"
#include "opennpux/npu_weight_ranges.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct opennpux_npu_weight_blob {
    void *data;
    size_t size;
    uint32_t data_type;
};

struct opennpux_npu_gptq_weights {
    struct opennpux_npu_weight_blob qweight;
    struct opennpux_npu_weight_blob qzeros;
    struct opennpux_npu_weight_blob scales;
    struct opennpux_npu_weight_blob g_idx;
};

int opennpux_npu_gptq_weights_load(
    const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    uint32_t role_id, uint64_t expert_id, uint32_t slot_id,
    size_t max_component_size,
    struct opennpux_npu_gptq_weights *weights);
void opennpux_npu_gptq_weights_unload(
    struct opennpux_npu_gptq_weights *weights);

#ifdef __cplusplus
}
#endif

#endif
