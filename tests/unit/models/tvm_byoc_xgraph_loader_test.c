#include "opennpux/xgraph_artifact.h"

#include <assert.h>
#include <stdio.h>

int
main(int argc, char **argv)
{
    assert(argc == 3);
    struct opennpux_xgraph_artifact artifact;
    assert(opennpux_xgraph_artifact_load(argv[1], &artifact) == 0);
    FILE *arena = fopen(argv[2], "rb");
    assert(arena != NULL);
    assert(fseek(arena, 0, SEEK_END) == 0);
    const long arena_size = ftell(arena);
    assert(arena_size > 0);
    assert(fclose(arena) == 0);
    assert(opennpux_xgraph_artifact_validate_arena(
               &artifact, (size_t)arena_size) == 0);
    assert(artifact.header->command_count == 4);
    assert(artifact.header->output_bytes == 6 * sizeof(float));
    opennpux_xgraph_artifact_unload(&artifact);
    puts("XGraph artifact runtime loader test: PASS");
    return 0;
}
