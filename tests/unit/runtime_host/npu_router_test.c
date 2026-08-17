#include "opennpux/npu_router.h"

#include <assert.h>
#include <math.h>

int
main(void)
{
    const float logits[] = {0.5f, 2.0f, -1.0f, 2.0f, 1.5f};
    struct opennpux_npu_route routes[3];
    assert(opennpux_npu_router_topk(logits, 5, 3, routes) == 0);
    assert(routes[0].expert_id == 1);
    assert(routes[1].expert_id == 3);
    assert(routes[2].expert_id == 4);
    assert(fabsf(routes[0].weight - routes[1].weight) < 1e-7f);
    assert(routes[0].weight > routes[2].weight);
    assert(fabsf(routes[0].weight + routes[1].weight + routes[2].weight -
                 1.0f) < 1e-6f);

    const float bad_logits[] = {0.0f, NAN};
    assert(opennpux_npu_router_topk(bad_logits, 2, 1, routes) != 0);
    assert(opennpux_npu_router_topk(logits, 5, 0, routes) != 0);
    assert(opennpux_npu_router_topk(logits, 5, 6, routes) != 0);
    return 0;
}
