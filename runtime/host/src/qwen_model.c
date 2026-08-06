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
