#ifndef OPENNPUX_NPU_ROUTER_H
#define OPENNPUX_NPU_ROUTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct opennpux_npu_route {
    uint32_t expert_id;
    float logit;
    float weight;
};

/*
 * Select stable top-K experts and normalize their weights over the selected
 * set. Equal logits are ordered by ascending expert ID for reproducibility.
 */
int opennpux_npu_router_topk(
    const float *logits, uint32_t expert_count, uint32_t active_count,
    struct opennpux_npu_route *routes);

#ifdef __cplusplus
}
#endif

#endif
