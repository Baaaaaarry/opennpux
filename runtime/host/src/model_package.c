#include "opennpux/model_package.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int manifest_directory(const char *path, char *output, size_t size);

static char *
read_file(const char *path)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        return NULL;
    }
    const long size = ftell(file);
    if (size <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        errno = EINVAL;
        return NULL;
    }
    char *buffer = malloc((size_t)size + 1);
    if (buffer == NULL) {
        fclose(file);
        errno = ENOMEM;
        return NULL;
    }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        errno = EIO;
        return NULL;
    }
    fclose(file);
    buffer[size] = '\0';
    return buffer;
}

static const char *
find_value(const char *json, const char *key)
{
    char needle[96];
    const int count = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (count <= 0 || (size_t)count >= sizeof(needle)) {
        errno = EINVAL;
        return NULL;
    }
    const char *value = strstr(json, needle);
    if (value == NULL || (value = strchr(value + count, ':')) == NULL) {
        errno = ENOENT;
        return NULL;
    }
    do {
        ++value;
    } while (*value == ' ' || *value == '\n' || *value == '\t');
    return value;
}

static int
parse_string(const char *json, const char *key, char *output, size_t size)
{
    const char *value = find_value(json, key);
    if (value == NULL || *value != '"') {
        errno = EINVAL;
        return -1;
    }
    const char *end = strchr(++value, '"');
    if (end == NULL || (size_t)(end - value) >= size) {
        errno = EINVAL;
        return -1;
    }
    memcpy(output, value, (size_t)(end - value));
    output[end - value] = '\0';
    return 0;
}

static int
parse_u64(const char *json, const char *key, uint64_t *output)
{
    const char *value = find_value(json, key);
    if (value == NULL) {
        return -1;
    }
    char *end = NULL;
    const unsigned long long parsed = strtoull(value, &end, 0);
    if (end == value) {
        errno = EINVAL;
        return -1;
    }
    *output = (uint64_t)parsed;
    return 0;
}

static int
parse_u32(const char *json, const char *key, uint32_t *output)
{
    uint64_t value = 0;
    if (parse_u64(json, key, &value) != 0 || value > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }
    *output = (uint32_t)value;
    return 0;
}

static int
parse_shards(const char *json, struct opennpux_model_package_info *info)
{
    const char *cursor = find_value(json, "shards");
    if (cursor == NULL || *cursor != '[') {
        errno = EINVAL;
        return -1;
    }
    const char *end = strchr(cursor, ']');
    if (end == NULL) {
        errno = EINVAL;
        return -1;
    }

    uint32_t count = 0;
    while ((cursor = strstr(cursor, "\"path\"")) != NULL && cursor < end) {
        if (count >= OPENNPUX_MODEL_MAX_SHARDS ||
            parse_string(cursor, "path", info->shards[count].path,
                         sizeof(info->shards[count].path)) != 0 ||
            parse_u64(cursor, "size", &info->shards[count].size) != 0 ||
            info->shards[count].path[0] == '/' ||
            strstr(info->shards[count].path, "..") != NULL) {
            errno = EINVAL;
            return -1;
        }
        ++count;
        cursor += 6;
    }
    if (count != info->shard_count) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}

int
opennpux_model_package_load(
    const char *manifest_path, struct opennpux_model_package_info *info)
{
    if (manifest_path == NULL || info == NULL) {
        errno = EINVAL;
        return -1;
    }
    char *json = read_file(manifest_path);
    if (json == NULL) {
        return -1;
    }
    memset(info, 0, sizeof(*info));
    uint32_t declared_shards = 0;
    int result =
        parse_string(json, "format", info->format, sizeof(info->format)) ||
        parse_u32(json, "version", &info->version) ||
        parse_string(json, "name", info->name, sizeof(info->name)) ||
        parse_string(json, "architecture", info->architecture_name,
                     sizeof(info->architecture_name)) ||
        parse_string(json, "dtype", info->dtype, sizeof(info->dtype)) ||
        parse_string(json, "tensor_index", info->tensor_index_path,
                     sizeof(info->tensor_index_path)) ||
        parse_u32(json, "layer_count", &info->layer_count) ||
        parse_u32(json, "vocab_size", &info->vocab_size) ||
        parse_u32(json, "hidden_size", &info->hidden_size) ||
        parse_u32(json, "intermediate_size", &info->intermediate_size) ||
        parse_u32(json, "head_count", &info->head_count) ||
        parse_u32(json, "kv_head_count", &info->kv_head_count) ||
        parse_u32(json, "head_dim", &info->head_dim) ||
        parse_u32(json, "max_sequence_length", &info->max_sequence_length) ||
        parse_u32(json, "tensor_count", &info->tensor_count) ||
        parse_u32(json, "shard_count", &declared_shards) ||
        parse_u64(json, "total_weight_bytes", &info->total_weight_bytes) ||
        parse_u32(json, "required_op_mask", &info->required_op_mask);
    info->shard_count = declared_shards;
    if (result == 0) {
        result = parse_shards(json, info);
    }
    free(json);

    if (result != 0 ||
        strcmp(info->format, OPENNPUX_MODEL_PACKAGE_FORMAT) != 0 ||
        info->version != OPENNPUX_MODEL_PACKAGE_VERSION ||
        info->layer_count == 0 || info->hidden_size == 0 ||
        info->head_count == 0 || info->kv_head_count == 0 ||
        info->head_dim == 0 || info->tensor_count == 0 ||
        info->shard_count == 0 ||
        info->shard_count > OPENNPUX_MODEL_MAX_SHARDS ||
        info->tensor_index_path[0] == '/' ||
        strstr(info->tensor_index_path, "..") != NULL) {
        errno = EINVAL;
        return -1;
    }
    info->architecture = strstr(info->architecture_name, "Qwen") != NULL ?
        OPENNPUX_MODEL_ARCH_QWEN : OPENNPUX_MODEL_ARCH_UNKNOWN;
    info->weight_format = OPENNPUX_MODEL_WEIGHTS_SAFETENSORS;
    return 0;
}

static int
package_path(const char *manifest_path, const char *relative_path,
             char *output, size_t size)
{
    char directory[OPENNPUX_MODEL_PATH_SIZE];
    if (manifest_directory(manifest_path, directory, sizeof(directory)) != 0) {
        return -1;
    }
    const int count = snprintf(output, size, "%s/%s", directory, relative_path);
    if (count <= 0 || (size_t)count >= size) {
        errno = ENAMETOOLONG;
        return -1;
    }
    return 0;
}

int
opennpux_model_package_find_tensor(
    const char *manifest_path,
    const struct opennpux_model_package_info *info, const char *tensor_name,
    struct opennpux_model_tensor_record *tensor)
{
    if (manifest_path == NULL || info == NULL || tensor_name == NULL ||
        tensor == NULL) {
        errno = EINVAL;
        return -1;
    }
    char path[OPENNPUX_MODEL_PATH_SIZE * 2];
    if (package_path(manifest_path, info->tensor_index_path,
                     path, sizeof(path)) != 0) {
        return -1;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    struct opennpux_model_tensor_index_header header;
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        header.magic != OPENNPUX_MODEL_TENSOR_INDEX_MAGIC ||
        header.version != OPENNPUX_MODEL_TENSOR_INDEX_VERSION ||
        header.header_size != sizeof(header) ||
        header.record_size != sizeof(*tensor) ||
        header.tensor_count != info->tensor_count) {
        fclose(file);
        errno = EINVAL;
        return -1;
    }
    for (uint32_t index = 0; index < header.tensor_count; ++index) {
        if (fread(tensor, sizeof(*tensor), 1, file) != 1) {
            fclose(file);
            errno = EIO;
            return -1;
        }
        tensor->name[OPENNPUX_MODEL_TENSOR_NAME_SIZE - 1] = '\0';
        if (strcmp(tensor->name, tensor_name) == 0) {
            fclose(file);
            if (tensor->shard_index >= info->shard_count ||
                tensor->rank > OPENNPUX_MODEL_TENSOR_MAX_DIMS ||
                tensor->data_offset > info->shards[tensor->shard_index].size ||
                tensor->data_size >
                    info->shards[tensor->shard_index].size - tensor->data_offset) {
                errno = EINVAL;
                return -1;
            }
            return 0;
        }
    }
    fclose(file);
    errno = ENOENT;
    return -1;
}

int
opennpux_model_package_read_tensor(
    const char *manifest_path,
    const struct opennpux_model_package_info *info,
    const struct opennpux_model_tensor_record *tensor, uint64_t offset,
    void *buffer, uint64_t size)
{
    if (manifest_path == NULL || info == NULL || tensor == NULL ||
        buffer == NULL || tensor->shard_index >= info->shard_count ||
        offset > tensor->data_size || size > tensor->data_size - offset) {
        errno = EINVAL;
        return -1;
    }
    char path[OPENNPUX_MODEL_PATH_SIZE * 2];
    if (package_path(manifest_path, info->shards[tensor->shard_index].path,
                     path, sizeof(path)) != 0) {
        return -1;
    }
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return -1;
    }
    const uint64_t position = tensor->data_offset + offset;
    if (position > (uint64_t)LONG_MAX || fseek(file, (long)position, SEEK_SET) != 0) {
        fclose(file);
        errno = EOVERFLOW;
        return -1;
    }
    uint8_t *destination = buffer;
    uint64_t received = 0;
    while (received < size) {
        const size_t chunk = size - received > UINT32_MAX ?
            UINT32_MAX : (size_t)(size - received);
        const size_t count = fread(destination + received, 1, chunk, file);
        if (count == 0) {
            fclose(file);
            errno = EIO;
            return -1;
        }
        received += count;
    }
    fclose(file);
    return 0;
}

static int
manifest_directory(const char *path, char *output, size_t size)
{
    const char *slash = strrchr(path, '/');
    const size_t length = slash == NULL ? 1 : (size_t)(slash - path);
    if (length >= size) {
        errno = ENAMETOOLONG;
        return -1;
    }
    if (slash == NULL) {
        output[0] = '.';
        output[1] = '\0';
    } else {
        memcpy(output, path, length);
        output[length] = '\0';
    }
    return 0;
}

int
opennpux_model_package_validate_shards(
    const char *manifest_path, const struct opennpux_model_package_info *info)
{
    if (manifest_path == NULL || info == NULL) {
        errno = EINVAL;
        return -1;
    }
    char directory[OPENNPUX_MODEL_PATH_SIZE];
    if (manifest_directory(manifest_path, directory, sizeof(directory)) != 0) {
        return -1;
    }
    uint64_t total = 0;
    for (uint32_t index = 0; index < info->shard_count; ++index) {
        char path[OPENNPUX_MODEL_PATH_SIZE * 2];
        const int count = snprintf(path, sizeof(path), "%s/%s", directory,
                                   info->shards[index].path);
        struct stat st;
        if (count <= 0 || (size_t)count >= sizeof(path) || stat(path, &st) != 0 ||
            st.st_size < 0 || (uint64_t)st.st_size != info->shards[index].size) {
            errno = EINVAL;
            return -1;
        }
        total += info->shards[index].size;
    }
    if (total != info->total_weight_bytes) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}
