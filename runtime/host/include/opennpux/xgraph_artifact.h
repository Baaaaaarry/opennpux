#ifndef OPENNPUX_XGRAPH_ARTIFACT_H
#define OPENNPUX_XGRAPH_ARTIFACT_H

#include <stddef.h>
#include <stdint.h>

#include "opennpux/xopennpux_graph.h"

#ifdef __cplusplus
extern "C" {
#endif

struct opennpux_xgraph_artifact {
    uint8_t *bytes;
    size_t size;
    const struct opennpux_xgraph_header *header;
    const struct opennpux_xgraph_command *commands;
};

int opennpux_xgraph_artifact_load(
    const char *path, struct opennpux_xgraph_artifact *artifact);
int opennpux_xgraph_artifact_validate_arena(
    const struct opennpux_xgraph_artifact *artifact, size_t arena_size);
void opennpux_xgraph_artifact_unload(
    struct opennpux_xgraph_artifact *artifact);

#ifdef __cplusplus
}
#endif

#endif
