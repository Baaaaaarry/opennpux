#include "opennpux/xopennpux_graph.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int
main(int argc, char **argv)
{
    assert(argc == 2);
    FILE *source = fopen(argv[1], "rb");
    assert(source != NULL);
    struct opennpux_xgraph_header header;
    assert(fread(&header, sizeof(header), 1, source) == 1);
    assert(header.magic == OPENNPUX_XGRAPH_MAGIC);
    assert(header.version == OPENNPUX_XGRAPH_VERSION);
    assert(header.header_size == sizeof(header));
    assert(header.command_size == sizeof(struct opennpux_xgraph_command));
    assert(header.command_count == 5);
    assert(header.total_size == sizeof(header) + 5 * header.command_size);
    assert(header.state == OPENNPUX_XGRAPH_STATE_READY);

    struct opennpux_xgraph_command commands[5];
    assert(fread(commands, sizeof(commands), 1, source) == 1);
    assert(fgetc(source) == EOF);
    assert(fclose(source) == 0);
    const uint32_t expected_opcodes[] = {
        OPENNPUX_XGRAPH_OP_TMMA,
        OPENNPUX_XGRAPH_OP_TADD,
        OPENNPUX_XGRAPH_OP_TSILU,
        OPENNPUX_XGRAPH_OP_TSOFTMAX,
        OPENNPUX_XGRAPH_OP_TTOPK,
    };
    for (uint32_t index = 0; index < 5; ++index) {
        assert(commands[index].opcode == expected_opcodes[index]);
        assert(commands[index].command_id == index);
        assert(commands[index].data_type == OPENNPUX_XGRAPH_DTYPE_FP32);
    }
    assert(commands[0].dim0 == 2);
    assert(commands[0].dim1 == 3);
    assert(commands[0].dim2 == 4);
    assert(commands[4].flags == OPENNPUX_XGRAPH_TTOPK_SPLIT_OUTPUT);
    assert(commands[4].reserved[0] != 0);
    puts("TVM BYOC artifact C ABI test: PASS");
    return 0;
}
