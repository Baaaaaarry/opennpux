#include "opennpux/qwen_model.h"

#include <errno.h>
#include <math.h>
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

static const char *const kOpNames[OPENNPUX_QWEN_OP_KIND_COUNT] = {
    "EMBED",
    "MATMUL",
    "ADD",
    "MUL",
    "RMS_NORM",
    "ROPE",
    "SILU",
    "SOFTMAX",
    "TOPK",
};

const char *
opennpux_qwen_op_name(uint32_t index)
{
    if (index >= OPENNPUX_QWEN_OP_KIND_COUNT) {
        return "UNKNOWN";
    }
    return kOpNames[index];
}

static int
op_index_from_name(const char *name)
{
    for (uint32_t index = 0; index < OPENNPUX_QWEN_OP_KIND_COUNT; ++index) {
        if (strcmp(name, kOpNames[index]) == 0) {
            return (int)index;
        }
    }
    return -1;
}

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

static int
copy_json_object(const char *begin, char *buffer, size_t size)
{
    const char *end = strchr(begin, '}');
    if (end == NULL || (size_t)(end - begin + 1) >= size) {
        errno = EINVAL;
        return -1;
    }
    memcpy(buffer, begin, (size_t)(end - begin + 1));
    buffer[end - begin + 1] = '\0';
    return 0;
}

static uint32_t
parse_shape_dims(const char *json, uint32_t *dims, uint32_t max_dims)
{
    const char *position = find_key(json, "shape");
    if (position == NULL) {
        errno = 0;
        return 0;
    }
    const char *begin = strchr(position, '[');
    const char *end = strchr(position, ']');
    if (begin == NULL || end == NULL || begin > end) {
        errno = EINVAL;
        return 0;
    }

    uint32_t count = 0;
    const char *cursor = begin + 1;
    while (cursor < end && count < max_dims) {
        while (cursor < end &&
               (*cursor == ' ' || *cursor == '\n' || *cursor == ',')) {
            ++cursor;
        }
        if (cursor >= end) {
            break;
        }
        char *parsed_end = NULL;
        const unsigned long value = strtoul(cursor, &parsed_end, 0);
        if (parsed_end == cursor || value > UINT32_MAX) {
            errno = EINVAL;
            return 0;
        }
        dims[count++] = (uint32_t)value;
        cursor = parsed_end;
    }
    return count;
}

static uint64_t
product_dims(const struct opennpux_qwen_op_entry *entry)
{
    uint64_t product = 1;
    for (uint32_t index = 0; index < entry->dim_count; ++index) {
        product *= entry->dims[index] == 0 ? 1 : entry->dims[index];
    }
    return product;
}

static void
model_qwen_op(struct opennpux_qwen_op_entry *entry,
              const struct opennpux_qwen_model_info *info)
{
    const uint64_t seq = info->prompt_token_count;
    const uint64_t hidden = info->hidden_size;
    const uint64_t inter = info->intermediate_size;
    const uint64_t heads = info->head_count;
    const uint64_t head_dim = info->head_dim;
    const uint64_t vocab = info->vocab_size;
    uint64_t elements = product_dims(entry);

    switch (entry->kind) {
      case 0: /* EMBED */
        entry->operations = seq * hidden;
        entry->bytes_read = seq * hidden * sizeof(float);
        entry->bytes_written = entry->bytes_read;
        break;
      case 1: /* MATMUL */
        if (entry->dim_count == 3) {
            entry->operations =
                (uint64_t)entry->dims[0] * entry->dims[1] * entry->dims[2];
            entry->bytes_read =
                ((uint64_t)entry->dims[0] * entry->dims[1] +
                 (uint64_t)entry->dims[1] * entry->dims[2]) * sizeof(float);
            entry->bytes_written =
                (uint64_t)entry->dims[0] * entry->dims[2] * sizeof(float);
        } else if (entry->dim_count == 2) {
            entry->operations = (uint64_t)entry->dims[0] * entry->dims[1];
            entry->bytes_read =
                ((uint64_t)entry->dims[0] + entry->dims[0] * entry->dims[1]) *
                sizeof(float);
            entry->bytes_written = (uint64_t)entry->dims[1] * sizeof(float);
        }
        break;
      case 2: /* ADD */
      case 3: /* MUL */
      case 6: /* SILU */
        entry->operations = elements;
        entry->bytes_read = elements * 2 * sizeof(float);
        entry->bytes_written = elements * sizeof(float);
        break;
      case 4: /* RMS_NORM */
        entry->operations = elements * 4;
        entry->bytes_read = elements * 2 * sizeof(float);
        entry->bytes_written = elements * sizeof(float);
        break;
      case 5: /* ROPE */
        entry->operations = seq * hidden * 4;
        entry->bytes_read = seq * hidden * sizeof(float);
        entry->bytes_written = entry->bytes_read;
        break;
      case 7: /* SOFTMAX */
        entry->operations = heads * seq * (seq + 1) / 2 * 4;
        entry->bytes_read = heads * seq * seq * sizeof(float);
        entry->bytes_written = entry->bytes_read;
        break;
      case 8: /* TOPK */
        entry->operations = vocab;
        entry->bytes_read = vocab * sizeof(float);
        entry->bytes_written = sizeof(uint32_t);
        break;
      default:
        break;
    }

    const uint64_t ops_per_cycle = 16;
    const uint64_t bytes_per_cycle = 32;
    const uint64_t compute_cycles =
        (entry->operations + ops_per_cycle - 1) / ops_per_cycle;
    const uint64_t memory_cycles =
        (entry->bytes_read + entry->bytes_written + bytes_per_cycle - 1) /
        bytes_per_cycle;
    entry->modeled_cycles = compute_cycles > memory_cycles ?
        compute_cycles : memory_cycles;
    if (entry->modeled_cycles == 0) {
        entry->modeled_cycles = 1;
    }
    (void)inter;
    (void)head_dim;
}

static int
parse_operator_trace(const char *json, struct opennpux_qwen_run_result *result)
{
    const char *trace = find_key(json, "operator_trace");
    if (trace == NULL) {
        return -1;
    }
    const char *trace_end = strstr(trace, "\n  ]");
    if (trace_end == NULL) {
        trace_end = json + strlen(json);
    }

    uint32_t op_index = 0;
    const char *cursor = trace;
    while ((cursor = strstr(cursor, "\"op\":")) != NULL &&
           cursor < trace_end && op_index < OPENNPUX_QWEN_MAX_OPS) {
        const char *object_begin = cursor;
        while (object_begin > json && *object_begin != '{') {
            --object_begin;
        }
        char object[512];
        if (*object_begin != '{' ||
            copy_json_object(object_begin, object, sizeof(object)) != 0) {
            return -1;
        }

        char op_name[32];
        if (parse_string_key(object, "op", op_name, sizeof(op_name)) != 0) {
            return -1;
        }
        const int kind = op_index_from_name(op_name);
        if (kind < 0) {
            errno = EINVAL;
            return -1;
        }

        struct opennpux_qwen_op_entry *entry = &result->ops[op_index];
        memset(entry, 0, sizeof(*entry));
        entry->index = op_index;
        entry->kind = (uint32_t)kind;
        if (parse_u32_key(object, "layer", &entry->layer) != 0) {
            entry->layer = UINT32_MAX;
            errno = 0;
        }
        entry->dim_count =
            parse_shape_dims(object, entry->dims, OPENNPUX_QWEN_OP_MAX_DIMS);
        result->op_counts[entry->kind]++;
        model_qwen_op(entry, &result->info);
        result->operation_count += entry->operations;
        result->bytes_read += entry->bytes_read;
        result->bytes_written += entry->bytes_written;
        result->modeled_cycles += entry->modeled_cycles;
        ++op_index;
        cursor += 5;
    }

    result->completed_operators = op_index;
    return op_index == result->info.operator_count ? 0 : -1;
}

static uint32_t
align_u32(uint32_t value, uint32_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

static uint32_t
fnv1a32(const void *data, uint32_t size)
{
    const uint8_t *bytes = (const uint8_t *)data;
    uint32_t checksum = 2166136261u;
    for (uint32_t index = 0; index < size; ++index) {
        checksum ^= bytes[index];
        checksum *= 16777619u;
    }
    return checksum;
}

static uint32_t
mix32(uint32_t hash, uint32_t value)
{
    hash ^= value;
    hash *= 16777619u;
    return hash;
}

uint32_t
opennpux_qwen_tcb_trace_checksum(const struct opennpux_qwen_tcb_op *ops,
                                 uint32_t op_count)
{
    if (ops == NULL || op_count > OPENNPUX_QWEN_MAX_OPS) {
        return 0;
    }

    uint32_t checksum = 2166136261u;
    for (uint32_t index = 0; index < op_count; ++index) {
        checksum = mix32(checksum, ops[index].index);
        checksum = mix32(checksum, ops[index].kind);
        checksum = mix32(checksum, ops[index].layer);
        checksum = mix32(checksum, ops[index].rank);
        checksum = mix32(checksum, (uint32_t)ops[index].operations);
        checksum = mix32(checksum, (uint32_t)(ops[index].operations >> 32));
        checksum = mix32(checksum, (uint32_t)ops[index].modeled_cycles);
        checksum = mix32(checksum,
                         (uint32_t)(ops[index].modeled_cycles >> 32));
    }
    return checksum;
}

int
opennpux_qwen_build_tcb(const struct opennpux_qwen_run_result *result,
                        void *buffer, uint32_t buffer_size,
                        uint32_t *tcb_size, uint32_t *tcb_checksum)
{
    if (result == NULL || buffer == NULL || tcb_size == NULL ||
        tcb_checksum == NULL || result->completed_operators == 0 ||
        result->completed_operators > OPENNPUX_QWEN_MAX_OPS) {
        errno = EINVAL;
        return -1;
    }

    const uint32_t required_size =
        sizeof(struct opennpux_qwen_tcb_header) +
        result->completed_operators * sizeof(struct opennpux_qwen_tcb_op);
    if (buffer_size < required_size || required_size > OPENNPUX_QWEN_TCB_MAX_SIZE) {
        errno = ERANGE;
        return -1;
    }

    memset(buffer, 0, buffer_size);
    struct opennpux_qwen_tcb_header *header =
        (struct opennpux_qwen_tcb_header *)buffer;
    struct opennpux_qwen_tcb_op *ops =
        (struct opennpux_qwen_tcb_op *)((uint8_t *)buffer + sizeof(*header));

    header->magic = OPENNPUX_QWEN_TCB_MAGIC;
    header->version = OPENNPUX_QWEN_TCB_VERSION;
    header->header_size = sizeof(*header);
    header->total_size = required_size;
    header->op_count = result->completed_operators;
    header->prompt_tokens = result->info.prompt_token_count;
    header->hidden_size = result->info.hidden_size;
    header->vocab_size = result->info.vocab_size;
    header->prompt_checksum = result->prompt_checksum;
    header->logits_checksum = result->output_checksum;
    header->next_token = result->next_token;
    header->operation_count = result->operation_count;
    header->bytes_read = result->bytes_read;
    header->bytes_written = result->bytes_written;
    header->modeled_cycles = result->modeled_cycles;

    uint32_t cursor = OPENNPUX_QWEN_TCB_TENSOR_BASE;
    for (uint32_t index = 0; index < result->completed_operators; ++index) {
        const struct opennpux_qwen_op_entry *entry = &result->ops[index];
        struct opennpux_qwen_tcb_op *descriptor = &ops[index];
        descriptor->index = entry->index;
        descriptor->kind = entry->kind;
        descriptor->layer = entry->layer;
        descriptor->rank = entry->dim_count;
        memcpy(descriptor->dims, entry->dims, sizeof(descriptor->dims));
        descriptor->operations = entry->operations;
        descriptor->bytes_read = entry->bytes_read;
        descriptor->bytes_written = entry->bytes_written;
        descriptor->modeled_cycles = entry->modeled_cycles;

        descriptor->input_offset = cursor;
        cursor = align_u32(cursor + (uint32_t)entry->bytes_read,
                           OPENNPUX_QWEN_TCB_TENSOR_ALIGN);
        descriptor->weight_offset = cursor;
        cursor = align_u32(cursor + (uint32_t)(entry->bytes_read / 2),
                           OPENNPUX_QWEN_TCB_TENSOR_ALIGN);
        descriptor->output_offset = cursor;
        cursor = align_u32(cursor + (uint32_t)entry->bytes_written,
                           OPENNPUX_QWEN_TCB_TENSOR_ALIGN);
        descriptor->scratch_offset = cursor;
        cursor = align_u32(cursor + 64, OPENNPUX_QWEN_TCB_TENSOR_ALIGN);
    }

    header->tcb_checksum = 0;
    header->tcb_checksum = fnv1a32(buffer, required_size);
    *tcb_size = required_size;
    *tcb_checksum = header->tcb_checksum;
    return 0;
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

int
opennpux_qwen_run_golden(const char *path,
                         struct opennpux_qwen_run_result *result)
{
    if (path == NULL || result == NULL) {
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

    memset(result, 0, sizeof(*result));
    if (opennpux_qwen_load_model_info(path, &result->info) != 0 ||
        parse_u32_key(json, "input_checksum", &result->prompt_checksum) != 0) {
        free(json);
        return -1;
    }
    free(json);

    result->completed_operators = result->info.operator_count;
    result->prefill_pass = result->info.prompt_token_count != 0 &&
        (result->info.op_mask & kRequiredOps) == kRequiredOps;
    result->decode_pass = result->info.next_token < result->info.vocab_size;
    result->output_checksum = result->info.logits_checksum;
    result->next_token = result->info.next_token;

    if (!result->prefill_pass || !result->decode_pass) {
        errno = EINVAL;
        return -1;
    }
    return 0;
}

static int
parse_double_array(const char *json, const char *key, double *values,
                   uint32_t count)
{
    const char *cursor = find_key(json, key);
    if (cursor == NULL) {
        return -1;
    }
    while (*cursor == ' ' || *cursor == '\n') {
        ++cursor;
    }
    if (*cursor++ != '[') {
        errno = EINVAL;
        return -1;
    }
    for (uint32_t index = 0; index < count; ++index) {
        while (*cursor == ' ' || *cursor == '\n') {
            ++cursor;
        }
        char *end = NULL;
        values[index] = strtod(cursor, &end);
        if (end == cursor) {
            errno = EINVAL;
            return -1;
        }
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n') {
            ++cursor;
        }
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                errno = EINVAL;
                return -1;
            }
        } else if (*cursor != ']') {
            errno = EINVAL;
            return -1;
        }
    }
    return 0;
}

static int
parse_u32_array(const char *json, const char *key, uint32_t *values,
                uint32_t count)
{
    const char *cursor = find_key(json, key);
    if (cursor == NULL) {
        return -1;
    }
    while (*cursor == ' ' || *cursor == '\n') {
        ++cursor;
    }
    if (*cursor++ != '[') {
        errno = EINVAL;
        return -1;
    }
    for (uint32_t index = 0; index < count; ++index) {
        while (*cursor == ' ' || *cursor == '\n') {
            ++cursor;
        }
        char *end = NULL;
        const unsigned long parsed = strtoul(cursor, &end, 0);
        if (end == cursor || parsed > UINT32_MAX) {
            errno = EINVAL;
            return -1;
        }
        values[index] = (uint32_t)parsed;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n') {
            ++cursor;
        }
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                errno = EINVAL;
                return -1;
            }
        } else if (*cursor != ']') {
            errno = EINVAL;
            return -1;
        }
    }
    return 0;
}

static void
matmul_vector(const double *vector, const double *matrix, uint32_t rows,
              uint32_t columns, double *output)
{
    for (uint32_t column = 0; column < columns; ++column) {
        double sum = 0.0;
        for (uint32_t row = 0; row < rows; ++row) {
            sum += vector[row] * matrix[(size_t)row * columns + column];
        }
        output[column] = sum;
    }
}

static void
rms_norm(const double *input, const double *weight, uint32_t count,
         double epsilon, double *output)
{
    double mean_square = 0.0;
    for (uint32_t index = 0; index < count; ++index) {
        mean_square += input[index] * input[index];
    }
    const double scale = 1.0 / sqrt(mean_square / count + epsilon);
    for (uint32_t index = 0; index < count; ++index) {
        output[index] = input[index] * scale * weight[index];
    }
}

static uint32_t
float_checksum(const double *values, uint32_t count)
{
    uint32_t checksum = UINT32_C(2166136261);
    for (uint32_t index = 0; index < count; ++index) {
        const float value = (float)values[index];
        const uint8_t *bytes = (const uint8_t *)&value;
        for (uint32_t byte = 0; byte < sizeof(value); ++byte) {
            checksum ^= bytes[byte];
            checksum *= UINT32_C(16777619);
        }
    }
    return checksum;
}

static int
run_numeric_reference(const char *json,
                      struct opennpux_qwen_run_result *result)
{
    const uint32_t tokens = result->info.prompt_token_count;
    const uint32_t hidden_size = result->info.hidden_size;
    const uint32_t intermediate = result->info.intermediate_size;
    const uint32_t vocab = result->info.vocab_size;
    const uint32_t heads = result->info.head_count;
    const uint32_t head_dim = result->info.head_dim;
    uint32_t version = 0;
    if (tokens == 0 || hidden_size == 0 || intermediate == 0 || vocab == 0 ||
        heads == 0 || heads * head_dim != hidden_size ||
        parse_u32_key(json, "numeric_reference_version", &version) != 0 ||
        version != 1) {
        errno = EINVAL;
        return -1;
    }

    const size_t token_hidden = (size_t)tokens * hidden_size;
    double *storage = calloc(
        token_hidden * 8 + (size_t)tokens * heads * tokens +
        (size_t)tokens * intermediate * 3 + vocab, sizeof(double));
    uint32_t *input_ids = calloc(tokens, sizeof(*input_ids));
    double *token_embedding = calloc((size_t)vocab * hidden_size, sizeof(double));
    double *lm_head = calloc((size_t)hidden_size * vocab, sizeof(double));
    double *rms_attn = calloc(hidden_size, sizeof(double));
    double *rms_ffn = calloc(hidden_size, sizeof(double));
    double *wq = calloc((size_t)hidden_size * hidden_size, sizeof(double));
    double *wk = calloc((size_t)hidden_size * hidden_size, sizeof(double));
    double *wv = calloc((size_t)hidden_size * hidden_size, sizeof(double));
    double *wo = calloc((size_t)hidden_size * hidden_size, sizeof(double));
    double *w_gate = calloc((size_t)hidden_size * intermediate, sizeof(double));
    double *w_up = calloc((size_t)hidden_size * intermediate, sizeof(double));
    double *w_down = calloc((size_t)intermediate * hidden_size, sizeof(double));
    int rc = -1;
    if (storage == NULL || input_ids == NULL || token_embedding == NULL ||
        lm_head == NULL || rms_attn == NULL || rms_ffn == NULL || wq == NULL ||
        wk == NULL || wv == NULL || wo == NULL || w_gate == NULL ||
        w_up == NULL || w_down == NULL) {
        errno = ENOMEM;
        goto out;
    }
    double epsilon = 0.0;
    const char *epsilon_text = find_key(json, "epsilon");
    char *epsilon_end = NULL;
    if (epsilon_text != NULL) {
        epsilon = strtod(epsilon_text, &epsilon_end);
    }
    if (epsilon_text == NULL || epsilon_end == epsilon_text || epsilon <= 0.0 ||
        parse_u32_array(json, "runtime_input_ids", input_ids, tokens) != 0 ||
        parse_double_array(json, "token_embedding", token_embedding,
                           vocab * hidden_size) != 0 ||
        parse_double_array(json, "lm_head", lm_head, hidden_size * vocab) != 0 ||
        parse_double_array(json, "rms_attn_weight", rms_attn, hidden_size) != 0 ||
        parse_double_array(json, "rms_ffn_weight", rms_ffn, hidden_size) != 0 ||
        parse_double_array(json, "wq", wq, hidden_size * hidden_size) != 0 ||
        parse_double_array(json, "wk", wk, hidden_size * hidden_size) != 0 ||
        parse_double_array(json, "wv", wv, hidden_size * hidden_size) != 0 ||
        parse_double_array(json, "wo", wo, hidden_size * hidden_size) != 0 ||
        parse_double_array(json, "w_gate", w_gate,
                           hidden_size * intermediate) != 0 ||
        parse_double_array(json, "w_up", w_up,
                           hidden_size * intermediate) != 0 ||
        parse_double_array(json, "w_down", w_down,
                           intermediate * hidden_size) != 0) {
        goto out;
    }

    double *hidden = storage;
    double *normed = hidden + token_hidden;
    double *q = normed + token_hidden;
    double *k = q + token_hidden;
    double *v = k + token_hidden;
    double *context = v + token_hidden;
    double *projected = context + token_hidden;
    double *ffn_normed = projected + token_hidden;
    double *scores = ffn_normed + token_hidden;
    double *gate = scores + (size_t)tokens * heads * tokens;
    double *up = gate + (size_t)tokens * intermediate;
    double *gated = up + (size_t)tokens * intermediate;
    double *logits = gated + (size_t)tokens * intermediate;

    for (uint32_t token = 0; token < tokens; ++token) {
        if (input_ids[token] >= vocab) {
            errno = EINVAL;
            goto out;
        }
        memcpy(hidden + (size_t)token * hidden_size,
               token_embedding + (size_t)input_ids[token] * hidden_size,
               hidden_size * sizeof(double));
        rms_norm(hidden + (size_t)token * hidden_size, rms_attn,
                 hidden_size, epsilon,
                 normed + (size_t)token * hidden_size);
        matmul_vector(normed + (size_t)token * hidden_size, wq,
                      hidden_size, hidden_size, q + (size_t)token * hidden_size);
        matmul_vector(normed + (size_t)token * hidden_size, wk,
                      hidden_size, hidden_size, k + (size_t)token * hidden_size);
        matmul_vector(normed + (size_t)token * hidden_size, wv,
                      hidden_size, hidden_size, v + (size_t)token * hidden_size);
    }
    for (uint32_t position = 0; position < tokens; ++position) {
        for (uint32_t head = 0; head < heads; ++head) {
            double peak = -INFINITY;
            for (uint32_t source = 0; source <= position; ++source) {
                double dot = 0.0;
                for (uint32_t lane = 0; lane < head_dim; ++lane) {
                    const uint32_t offset = head * head_dim + lane;
                    dot += q[(size_t)position * hidden_size + offset] *
                           k[(size_t)source * hidden_size + offset];
                }
                const double score = dot / sqrt((double)head_dim);
                scores[((size_t)position * heads + head) * tokens + source] = score;
                if (score > peak) {
                    peak = score;
                }
            }
            double total = 0.0;
            for (uint32_t source = 0; source <= position; ++source) {
                double *score = &scores[
                    ((size_t)position * heads + head) * tokens + source];
                *score = exp(*score - peak);
                total += *score;
            }
            for (uint32_t source = 0; source <= position; ++source) {
                const double probability = scores[
                    ((size_t)position * heads + head) * tokens + source] / total;
                for (uint32_t lane = 0; lane < head_dim; ++lane) {
                    const uint32_t offset = head * head_dim + lane;
                    context[(size_t)position * hidden_size + offset] +=
                        probability * v[(size_t)source * hidden_size + offset];
                }
            }
        }
        matmul_vector(context + (size_t)position * hidden_size, wo,
                      hidden_size, hidden_size,
                      projected + (size_t)position * hidden_size);
        for (uint32_t lane = 0; lane < hidden_size; ++lane) {
            hidden[(size_t)position * hidden_size + lane] +=
                projected[(size_t)position * hidden_size + lane];
        }
        rms_norm(hidden + (size_t)position * hidden_size, rms_ffn,
                 hidden_size, epsilon,
                 ffn_normed + (size_t)position * hidden_size);
        matmul_vector(ffn_normed + (size_t)position * hidden_size, w_gate,
                      hidden_size, intermediate,
                      gate + (size_t)position * intermediate);
        matmul_vector(ffn_normed + (size_t)position * hidden_size, w_up,
                      hidden_size, intermediate,
                      up + (size_t)position * intermediate);
        for (uint32_t lane = 0; lane < intermediate; ++lane) {
            const double value = gate[(size_t)position * intermediate + lane];
            gated[(size_t)position * intermediate + lane] =
                value / (1.0 + exp(-value)) *
                up[(size_t)position * intermediate + lane];
        }
        matmul_vector(gated + (size_t)position * intermediate, w_down,
                      intermediate, hidden_size, projected);
        for (uint32_t lane = 0; lane < hidden_size; ++lane) {
            hidden[(size_t)position * hidden_size + lane] += projected[lane];
        }
    }
    rms_norm(hidden + (size_t)(tokens - 1) * hidden_size, rms_ffn,
             hidden_size, epsilon, normed);
    matmul_vector(normed, lm_head, hidden_size, vocab, logits);
    uint32_t next_token = 0;
    for (uint32_t token = 1; token < vocab; ++token) {
        if (logits[token] > logits[next_token]) {
            next_token = token;
        }
    }
    const uint32_t logits_checksum = float_checksum(logits, vocab);
    if (next_token != result->info.next_token ||
        logits_checksum != result->info.logits_checksum) {
        errno = EIO;
        goto out;
    }
    result->next_token = next_token;
    result->output_checksum = logits_checksum;
    rc = 0;
out:
    free(storage);
    free(input_ids);
    free(token_embedding);
    free(lm_head);
    free(rms_attn);
    free(rms_ffn);
    free(wq);
    free(wk);
    free(wv);
    free(wo);
    free(w_gate);
    free(w_up);
    free(w_down);
    return rc;
}

int
opennpux_qwen_run_hybrid_sim(const char *path,
                             struct opennpux_qwen_run_result *result)
{
    if (path == NULL || result == NULL) {
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

    if (opennpux_qwen_run_golden(path, result) != 0) {
        free(json);
        return -1;
    }

    if (run_numeric_reference(json, result) != 0) {
        free(json);
        return -1;
    }

    if (parse_operator_trace(json, result) != 0) {
        free(json);
        return -1;
    }
    free(json);

    uint8_t tcb[OPENNPUX_QWEN_TCB_MAX_SIZE];
    if (opennpux_qwen_build_tcb(result, tcb, sizeof(tcb),
                                &result->tcb_size,
                                &result->tcb_checksum) != 0) {
        return -1;
    }
    return 0;
}
