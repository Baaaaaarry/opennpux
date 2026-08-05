#include "opennpux/qwen_model.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t kRequiredOps =
    OPENNPUX_QWEN_OP_EMBED |
    OPENNPUX_QWEN_OP_MATMUL |
    OPENNPUX_QWEN_OP_ADD |
    OPENNPUX_QWEN_OP_MUL |
    OPENNPUX_QWEN_OP_RMS_NORM |
    OPENNPUX_QWEN_OP_ROPE |
    OPENNPUX_QWEN_OP_SILU |
    OPENNPUX_QWEN_OP_SOFTMAX |
    OPENNPUX_QWEN_OP_TOPK;

const char *
opennpux_qwen_required_ops_string(void)
{
    return "ADD,EMBED,MATMUL,MUL,RMS_NORM,ROPE,SILU,SOFTMAX,TOPK";
}

uint32_t
opennpux_qwen_required_op_mask(void)
{
    return kRequiredOps;
}

static char *
read_file(const char *path, size_t *size_out)
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
    if (size < 0) {
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    char *buffer = (char *)malloc((size_t)size + 1);
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
    if (size_out != NULL) {
        *size_out = (size_t)size;
    }
    return buffer;
}

static const char *
find_key(const char *json, const char *key)
{
    char needle[96];
    const int written = snprintf(needle, sizeof(needle), "\"%s\"", key);
    if (written <= 0 || (size_t)written >= sizeof(needle)) {
        errno = EINVAL;
        return NULL;
    }
    const char *position = strstr(json, needle);
    if (position == NULL) {
        errno = ENOENT;
        return NULL;
    }
    position = strchr(position + written, ':');
    if (position == NULL) {
        errno = EINVAL;
        return NULL;
    }
    return position + 1;
}

static int
parse_u32_key(const char *json, const char *key, uint32_t *value)
{
    const char *position = find_key(json, key);
    if (position == NULL) {
        return -1;
    }
    while (*position == ' ' || *position == '\n') {
        ++position;
    }
    char *end = NULL;
    const unsigned long parsed = strtoul(position, &end, 0);
    if (end == position || parsed > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static int
parse_hex_key(const char *json, const char *key, uint32_t *value)
{
    const char *position = find_key(json, key);
    if (position == NULL) {
        return -1;
    }
    while (*position == ' ' || *position == '\n') {
        ++position;
    }
    if (*position != '"') {
        errno = EINVAL;
        return -1;
    }
    ++position;
    char *end = NULL;
    const unsigned long parsed = strtoul(position, &end, 16);
    if (end == position || parsed > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }
    *value = (uint32_t)parsed;
    return 0;
}

static int
parse_string_key(const char *json, const char *key, char *value, size_t size)
{
    const char *position = find_key(json, key);
    if (position == NULL) {
        return -1;
    }
    while (*position == ' ' || *position == '\n') {
        ++position;
    }
    if (*position != '"') {
        errno = EINVAL;
        return -1;
    }
    ++position;
    const char *end = strchr(position, '"');
    if (end == NULL || (size_t)(end - position) >= size) {
        errno = EINVAL;
        return -1;
    }
    memcpy(value, position, (size_t)(end - position));
    value[end - position] = '\0';
    return 0;
}

static uint32_t
operator_mask_from_json(const char *json)
{
    uint32_t mask = 0;
    if (strstr(json, "\"op\": \"EMBED\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_EMBED;
    }
    if (strstr(json, "\"op\": \"MATMUL\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_MATMUL;
    }
    if (strstr(json, "\"op\": \"ADD\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_ADD;
    }
    if (strstr(json, "\"op\": \"MUL\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_MUL;
    }
    if (strstr(json, "\"op\": \"RMS_NORM\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_RMS_NORM;
    }
    if (strstr(json, "\"op\": \"ROPE\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_ROPE;
    }
    if (strstr(json, "\"op\": \"SILU\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_SILU;
    }
    if (strstr(json, "\"op\": \"SOFTMAX\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_SOFTMAX;
    }
    if (strstr(json, "\"op\": \"TOPK\"") != NULL) {
        mask |= OPENNPUX_QWEN_OP_TOPK;
    }
    return mask;
}

static uint32_t
count_operators(const char *json)
{
    uint32_t count = 0;
    const char *position = json;
    while ((position = strstr(position, "\"op\":")) != NULL) {
        ++count;
        position += 5;
    }
    return count;
}

static uint32_t
count_prompt_tokens(const char *json)
{
    const char *position = find_key(json, "input_ids");
    if (position == NULL) {
        return 0;
    }
    const char *begin = strchr(position, '[');
    const char *end = strchr(position, ']');
    if (begin == NULL || end == NULL || begin > end) {
        errno = EINVAL;
        return 0;
    }
    uint32_t count = 0;
    int in_number = 0;
    for (const char *cursor = begin + 1; cursor < end; ++cursor) {
        if (*cursor >= '0' && *cursor <= '9') {
            if (!in_number) {
                ++count;
                in_number = 1;
            }
        } else {
            in_number = 0;
        }
    }
    return count;
}

int
opennpux_qwen_load_model_info(const char *path,
                              struct opennpux_qwen_model_info *info)
{
    if (path == NULL || info == NULL) {
        errno = EINVAL;
        return -1;
    }
    size_t size = 0;
    char *json = read_file(path, &size);
    if (json == NULL) {
        return -1;
    }
    if (size == 0) {
        free(json);
        errno = EINVAL;
        return -1;
    }

    memset(info, 0, sizeof(*info));
    int result =
        parse_string_key(json, "format", info->format, sizeof(info->format)) ||
        parse_u32_key(json, "version", &info->version) ||
        parse_string_key(json, "name", info->name, sizeof(info->name)) ||
        parse_u32_key(json, "layer_count", &info->layer_count) ||
        parse_u32_key(json, "vocab_size", &info->vocab_size) ||
        parse_u32_key(json, "hidden_size", &info->hidden_size) ||
        parse_u32_key(json, "intermediate_size", &info->intermediate_size) ||
        parse_u32_key(json, "head_count", &info->head_count) ||
        parse_u32_key(json, "head_dim", &info->head_dim) ||
        parse_u32_key(json, "next_token", &info->next_token) ||
        parse_hex_key(json, "logits_checksum", &info->logits_checksum) ||
        parse_hex_key(json, "weight_checksum", &info->weight_checksum);
    if (result != 0) {
        free(json);
        return -1;
    }

    info->prompt_token_count = count_prompt_tokens(json);
    info->operator_count = count_operators(json);
    info->op_mask = operator_mask_from_json(json);
    free(json);

    if (strcmp(info->format, OPENNPUX_QWEN_TINY_FORMAT) != 0 ||
        info->version != 1 ||
        info->prompt_token_count == 0 ||
        info->operator_count == 0 ||
        (info->op_mask & kRequiredOps) != kRequiredOps ||
        info->next_token >= info->vocab_size) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}

