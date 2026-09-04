#include "opennpux/xgraph_artifact.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
read_file(const char *path, uint8_t **bytes, size_t *size)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return -1;
    }
    const long length = ftell(file);
    if (length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        errno = EINVAL;
        return -1;
    }
    uint8_t *image = malloc((size_t)length);
    if (image == NULL) {
        fclose(file);
        return -1;
    }
    const size_t count = fread(image, 1, (size_t)length, file);
    const int failed = count != (size_t)length || ferror(file);
    fclose(file);
    if (failed) {
        free(image);
        errno = EIO;
        return -1;
    }
    *bytes = image;
    *size = (size_t)length;
    return 0;
}

int
opennpux_xgraph_artifact_load(
    const char *path, struct opennpux_xgraph_artifact *artifact)
{
    if (path == NULL || artifact == NULL) {
        errno = EINVAL;
        return -1;
    }
    memset(artifact, 0, sizeof(*artifact));
    if (read_file(path, &artifact->bytes, &artifact->size) != 0) {
        return -1;
    }
    if (artifact->size < sizeof(struct opennpux_xgraph_header)) {
        goto invalid;
    }
    artifact->header =
        (const struct opennpux_xgraph_header *)(const void *)artifact->bytes;
    const struct opennpux_xgraph_header *header = artifact->header;
    const uint64_t command_bytes =
        (uint64_t)header->command_count * sizeof(struct opennpux_xgraph_command);
    const uint64_t expected_size = sizeof(*header) + command_bytes;
    if (header->magic != OPENNPUX_XGRAPH_MAGIC ||
        header->version != OPENNPUX_XGRAPH_VERSION ||
        header->header_size != sizeof(*header) ||
        header->command_size != sizeof(struct opennpux_xgraph_command) ||
        header->command_count == 0 ||
        header->command_count > OPENNPUX_XGRAPH_MAX_COMMANDS ||
        expected_size != artifact->size || header->total_size != expected_size ||
        header->state != OPENNPUX_XGRAPH_STATE_READY ||
        OPENNPUX_XGRAPH_OFFSET + expected_size > OPENNPUX_XGRAPH_DATA_OFFSET) {
        goto invalid;
    }
    artifact->commands =
        (const struct opennpux_xgraph_command *)(const void *)(header + 1);
    for (uint32_t index = 0; index < header->command_count; ++index) {
        if (artifact->commands[index].command_id != index) {
            goto invalid;
        }
    }
    return 0;

invalid:
    opennpux_xgraph_artifact_unload(artifact);
    errno = EPROTO;
    return -1;
}

int
opennpux_xgraph_artifact_validate_arena(
    const struct opennpux_xgraph_artifact *artifact, size_t arena_size)
{
    if (artifact == NULL || artifact->header == NULL ||
        arena_size < OPENNPUX_XGRAPH_DATA_OFFSET || arena_size > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }
    const struct opennpux_xgraph_header *header = artifact->header;
    if (header->output_offset < OPENNPUX_XGRAPH_DATA_OFFSET ||
        header->output_offset > arena_size ||
        header->output_bytes > arena_size - header->output_offset) {
        errno = ERANGE;
        return -1;
    }
    for (uint32_t index = 0; index < header->command_count; ++index) {
        const struct opennpux_xgraph_command *command =
            &artifact->commands[index];
        if (command->destination_offset < OPENNPUX_XGRAPH_DATA_OFFSET ||
            command->destination_offset >= arena_size ||
            command->source0_offset < OPENNPUX_XGRAPH_DATA_OFFSET ||
            command->source0_offset >= arena_size ||
            (command->source1_offset != 0 &&
             (command->source1_offset < OPENNPUX_XGRAPH_DATA_OFFSET ||
              command->source1_offset >= arena_size))) {
            errno = ERANGE;
            return -1;
        }
    }
    return 0;
}

void
opennpux_xgraph_artifact_unload(struct opennpux_xgraph_artifact *artifact)
{
    if (artifact == NULL) {
        return;
    }
    free(artifact->bytes);
    memset(artifact, 0, sizeof(*artifact));
}
