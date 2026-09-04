#include "opennpux/xgraph_artifact.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char **argv)
{
    assert(argc == 5);
    const uint32_t expected_commands = (uint32_t)strtoul(argv[3], NULL, 0);
    const uint32_t expected_output_bytes = (uint32_t)strtoul(argv[4], NULL, 0);
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
    assert(artifact.header->command_count == expected_commands);
    assert(artifact.header->output_bytes == expected_output_bytes);
    opennpux_xgraph_artifact_unload(&artifact);
    puts("XGraph artifact runtime loader test: PASS");
    return 0;
}
