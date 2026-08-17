#include "opennpux/npu_router.h"

#include <math.h>
#include <stddef.h>

static int
route_precedes(float logit, uint32_t expert_id,
               const struct opennpux_npu_route *route)
{
    return logit > route->logit ||
        (logit == route->logit && expert_id < route->expert_id);
}

int
opennpux_npu_router_topk(
    const float *logits, uint32_t expert_count, uint32_t active_count,
    struct opennpux_npu_route *routes)
{
    if (logits == NULL || routes == NULL || expert_count == 0 ||
        active_count == 0 || active_count > expert_count) {
        return -1;
    }
    for (uint32_t index = 0; index < active_count; ++index) {
        routes[index].expert_id = UINT32_MAX;
        routes[index].logit = -INFINITY;
        routes[index].weight = 0.0f;
    }

    for (uint32_t expert = 0; expert < expert_count; ++expert) {
        const float logit = logits[expert];
        if (!isfinite(logit)) {
            return -1;
        }
        uint32_t position = active_count;
        for (uint32_t index = 0; index < active_count; ++index) {
            if (route_precedes(logit, expert, &routes[index])) {
                position = index;
                break;
            }
        }
        if (position == active_count) {
            continue;
        }
        for (uint32_t index = active_count - 1; index > position; --index) {
            routes[index] = routes[index - 1];
        }
        routes[position].expert_id = expert;
        routes[position].logit = logit;
    }

    const float maximum = routes[0].logit;
    float sum = 0.0f;
    for (uint32_t index = 0; index < active_count; ++index) {
        routes[index].weight = expf(routes[index].logit - maximum);
        sum += routes[index].weight;
    }
    if (!isfinite(sum) || sum <= 0.0f) {
        return -1;
    }
    for (uint32_t index = 0; index < active_count; ++index) {
        routes[index].weight /= sum;
    }
    return 0;
}
