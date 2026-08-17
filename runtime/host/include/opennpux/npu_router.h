#ifndef OPENNPUX_NPU_ROUTER_H
#define OPENNPUX_NPU_ROUTER_H

#include <stdint.h>

#include "opennpux/npu_route_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct opennpux_npu_route_record opennpux_npu_route;

/*
 * Select stable top-K experts and normalize their weights over the selected
 * set. Equal logits are ordered by ascending expert ID for reproducibility.
 */
int opennpux_npu_router_topk(
    const float *logits, uint32_t expert_count, uint32_t active_count,
    opennpux_npu_route *routes);

#ifdef __cplusplus
}
#endif

#endif
