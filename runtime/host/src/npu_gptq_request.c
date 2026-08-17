#include "opennpux/npu_gptq_request.h"

#include "opennpux/npu_gptq_weights.h"
#include "opennpux/npu_submission.h"

#include <errno.h>
#include <string.h>

static uint32_t
ceil_div(uint32_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

/* Bytes for `count` elements of `element_size`, or 0 on overflow. */
static uint64_t
byte_size(uint64_t count, uint32_t element_size)
{
    return count > UINT64_MAX / element_size ? 0 : count * element_size;
}

static uint64_t
element_count(uint32_t rows, uint32_t columns)
{
    return (uint64_t)rows * columns;
}

/* Place one operand and advance the cursor to the next aligned boundary. */
static int
place(uint64_t bytes, uint32_t *offset, uint64_t *cursor)
{
    const uint32_t alignment = OPENNPUX_NPU_GPTQ_REQUEST_ALIGNMENT;
    if (bytes == 0 || *cursor > UINT32_MAX) {
        return -1;
    }
    *offset = (uint32_t)*cursor;
    const uint64_t end = *cursor + bytes;
    if (end > UINT32_MAX - (alignment - 1)) {
        return -1;
    }
    *cursor = (end + alignment - 1) & ~(uint64_t)(alignment - 1);
    return 0;
}

static uint32_t
scale_element_size_for(uint32_t data_type)
{
    if (data_type == OPENNPUX_NPU_DTYPE_FLOAT16 ||
        data_type == OPENNPUX_NPU_DTYPE_BFLOAT16) {
        return 2;
    }
    return data_type == OPENNPUX_NPU_DTYPE_FLOAT32 ? 4 : 0;
}

int
opennpux_npu_gptq_request_layout(
    const struct opennpux_npu_gptq_projection_shape *shape,
    uint32_t scale_element_size, uint32_t has_g_idx,
    struct opennpux_npu_gptq_request_layout *layout)
{
    if (shape == NULL || layout == NULL || shape->rows == 0 ||
        shape->input_columns == 0 || shape->output_columns == 0 ||
        shape->group_size == 0 || shape->zero_bias > 15 ||
        (scale_element_size != 2 && scale_element_size != 4)) {
        errno = EINVAL;
        return -1;
    }
    memset(layout, 0, sizeof(*layout));

    const uint32_t groups = ceil_div(shape->input_columns, shape->group_size);
    const uint64_t input_bytes = byte_size(
        element_count(shape->rows, shape->input_columns), sizeof(float));
    const uint64_t qweight_bytes = byte_size(
        element_count(ceil_div(shape->input_columns, 8), shape->output_columns),
        sizeof(uint32_t));
    const uint64_t qzeros_bytes = byte_size(
        element_count(groups, ceil_div(shape->output_columns, 8)),
        sizeof(uint32_t));
    const uint64_t scales_bytes = byte_size(
        element_count(groups, shape->output_columns), scale_element_size);
    const uint64_t g_idx_bytes = byte_size(shape->input_columns,
                                           sizeof(uint32_t));
    const uint64_t output_bytes = byte_size(
        element_count(shape->rows, shape->output_columns), sizeof(float));
    uint64_t cursor = sizeof(struct coral_gptq_matmul_request);
    if (input_bytes == 0 || qweight_bytes == 0 || qzeros_bytes == 0 ||
        scales_bytes == 0 || g_idx_bytes == 0 || output_bytes == 0 ||
        output_bytes > UINT32_MAX ||
        place(input_bytes, &layout->input_offset, &cursor) != 0 ||
        place(qweight_bytes, &layout->qweight_offset, &cursor) != 0 ||
        place(qzeros_bytes, &layout->qzeros_offset, &cursor) != 0 ||
        place(scales_bytes, &layout->scales_offset, &cursor) != 0 ||
        (has_g_idx && place(g_idx_bytes, &layout->g_idx_offset, &cursor) != 0) ||
        place(output_bytes, &layout->output_offset, &cursor) != 0 ||
        cursor > UINT32_MAX) {
        errno = EOVERFLOW;
        return -1;
    }
    layout->output_bytes = (uint32_t)output_bytes;
    layout->total_size = (uint32_t)cursor;
    return 0;
}

/* Copy one resolved component after checking it against the shape. */
static int
copy_component(const struct opennpux_npu_weight_blob *blob, uint64_t expected,
               uint8_t *image, uint32_t offset)
{
    if (blob->data == NULL || blob->size != expected) {
        errno = EPROTO;
        return -1;
    }
    memcpy(image + offset, blob->data, blob->size);
    return 0;
}

int
opennpux_npu_gptq_request_stage(
    const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_ranges *ranges,
    const struct opennpux_npu_gptq_projection_selector *selector,
    const struct opennpux_npu_gptq_projection_shape *shape,
    const float *input, size_t input_floats, uint64_t device_base,
    void *image, size_t image_size,
    struct opennpux_npu_gptq_request_layout *layout)
{
    if (manifest_path == NULL || model == NULL || ranges == NULL ||
        selector == NULL || shape == NULL || input == NULL || image == NULL ||
        layout == NULL || device_base > UINT32_MAX) {
        errno = EINVAL;
        return -1;
    }
    if (input_floats != element_count(shape->rows, shape->input_columns)) {
        errno = EINVAL;
        return -1;
    }

    struct opennpux_npu_gptq_weights weights;
    if (opennpux_npu_gptq_weights_load(
            manifest_path, model, ranges, selector->command_id,
            selector->role_id, selector->expert_id, selector->slot_id,
            image_size, &weights) != 0) {
        return -1;
    }
    const uint32_t scale_element_size =
        scale_element_size_for(weights.scales.data_type);
    if (scale_element_size == 0) {
        opennpux_npu_gptq_weights_unload(&weights);
        errno = EPROTO;
        return -1;
    }
    if (opennpux_npu_gptq_request_layout(
            shape, scale_element_size, weights.g_idx.data != NULL,
            layout) != 0) {
        const int saved = errno;
        opennpux_npu_gptq_weights_unload(&weights);
        errno = saved;
        return -1;
    }
    if (layout->total_size > image_size) {
        opennpux_npu_gptq_weights_unload(&weights);
        errno = ENOSPC;
        return -1;
    }
    layout->scale_data_type = weights.scales.data_type;

    uint8_t *bytes = image;
    memset(bytes, 0, layout->total_size);
    const uint32_t groups = ceil_div(shape->input_columns, shape->group_size);
    if (copy_component(&weights.qweight,
                       byte_size(element_count(
                           ceil_div(shape->input_columns, 8),
                           shape->output_columns), sizeof(uint32_t)),
                       bytes, layout->qweight_offset) != 0 ||
        copy_component(&weights.qzeros,
                       byte_size(element_count(
                           groups, ceil_div(shape->output_columns, 8)),
                           sizeof(uint32_t)),
                       bytes, layout->qzeros_offset) != 0 ||
        copy_component(&weights.scales,
                       byte_size(element_count(
                           groups, shape->output_columns),
                           scale_element_size),
                       bytes, layout->scales_offset) != 0 ||
        (weights.g_idx.data != NULL &&
         copy_component(&weights.g_idx,
                        byte_size(shape->input_columns, sizeof(uint32_t)),
                        bytes, layout->g_idx_offset) != 0)) {
        const int saved = errno;
        opennpux_npu_gptq_weights_unload(&weights);
        errno = saved;
        return -1;
    }
    opennpux_npu_gptq_weights_unload(&weights);
    memcpy(bytes + layout->input_offset, input,
           input_floats * sizeof(*input));

    struct coral_gptq_matmul_request request;
    memset(&request, 0, sizeof(request));
    request.magic = CORAL_GPTQ_MATMUL_MAGIC;
    request.version = CORAL_GPTQ_MATMUL_VERSION;
    request.struct_size = sizeof(request);
    request.state = CORAL_GPTQ_MATMUL_PENDING;
    request.rows = shape->rows;
    request.input_columns = shape->input_columns;
    request.output_columns = shape->output_columns;
    request.group_size = shape->group_size;
    request.zero_bias = shape->zero_bias;
    request.input_address = (uint32_t)device_base + layout->input_offset;
    request.qweight_address = (uint32_t)device_base + layout->qweight_offset;
    request.qzeros_address = (uint32_t)device_base + layout->qzeros_offset;
    request.scales_address = (uint32_t)device_base + layout->scales_offset;
    request.g_idx_address = layout->g_idx_offset == 0 ? 0 :
        (uint32_t)device_base + layout->g_idx_offset;
    request.output_address = (uint32_t)device_base + layout->output_offset;
    request.scale_data_type = layout->scale_data_type;
    memcpy(bytes, &request, sizeof(request));
    return 0;
}

static int
place_weight_set(
    uint32_t input_columns, uint32_t output_columns, uint32_t group_size,
    const struct opennpux_npu_gptq_weights *weights,
    struct opennpux_npu_gptq_weight_layout *layout, uint64_t *cursor)
{
    const uint32_t scale_size =
        scale_element_size_for(weights->scales.data_type);
    const uint32_t groups = ceil_div(input_columns, group_size);
    const uint64_t qweight_bytes = byte_size(
        element_count(ceil_div(input_columns, 8), output_columns),
        sizeof(uint32_t));
    const uint64_t qzeros_bytes = byte_size(
        element_count(groups, ceil_div(output_columns, 8)), sizeof(uint32_t));
    const uint64_t scales_bytes = byte_size(
        element_count(groups, output_columns), scale_size);
    const uint64_t g_idx_bytes = byte_size(input_columns, sizeof(uint32_t));
    if (scale_size == 0 || qweight_bytes == 0 || qzeros_bytes == 0 ||
        scales_bytes == 0 ||
        weights->qweight.size != qweight_bytes ||
        weights->qzeros.size != qzeros_bytes ||
        weights->scales.size != scales_bytes ||
        (weights->g_idx.data != NULL && weights->g_idx.size != g_idx_bytes) ||
        place(qweight_bytes, &layout->qweight_offset, cursor) != 0 ||
        place(qzeros_bytes, &layout->qzeros_offset, cursor) != 0 ||
        place(scales_bytes, &layout->scales_offset, cursor) != 0 ||
        (weights->g_idx.data != NULL &&
         place(g_idx_bytes, &layout->g_idx_offset, cursor) != 0)) {
        errno = EPROTO;
        return -1;
    }
    layout->scale_data_type = weights->scales.data_type;
    return 0;
}

static int
copy_weight_set(
    const struct opennpux_npu_gptq_weights *weights,
    const struct opennpux_npu_gptq_weight_layout *layout, uint8_t *image)
{
    if (copy_component(&weights->qweight, weights->qweight.size, image,
                       layout->qweight_offset) != 0 ||
        copy_component(&weights->qzeros, weights->qzeros.size, image,
                       layout->qzeros_offset) != 0 ||
        copy_component(&weights->scales, weights->scales.size, image,
                       layout->scales_offset) != 0 ||
        (weights->g_idx.data != NULL &&
         copy_component(&weights->g_idx, weights->g_idx.size, image,
                        layout->g_idx_offset) != 0)) {
        return -1;
    }
    return 0;
}

static struct coral_gptq_projection_weights
device_weight_set(
    uint32_t device_base,
    const struct opennpux_npu_gptq_weight_layout *layout)
{
    const struct coral_gptq_projection_weights result = {
        device_base + layout->qweight_offset,
        device_base + layout->qzeros_offset,
        device_base + layout->scales_offset,
        layout->g_idx_offset == 0 ? 0 : device_base + layout->g_idx_offset,
        layout->scale_data_type,
        0,
    };
    return result;
}

int
opennpux_npu_gptq_expert_stage(
    const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_ranges *ranges,
    const struct opennpux_npu_gptq_expert_selector *selector,
    const struct opennpux_npu_gptq_expert_shape *shape,
    const float *input, size_t input_floats, uint64_t device_base,
    void *image, size_t image_size,
    struct opennpux_npu_gptq_expert_layout *layout)
{
    if (manifest_path == NULL || model == NULL || ranges == NULL ||
        selector == NULL || shape == NULL || input == NULL || image == NULL ||
        layout == NULL || device_base > UINT32_MAX || shape->rows == 0 ||
        shape->hidden_columns == 0 || shape->intermediate_columns == 0 ||
        shape->group_size == 0 || shape->zero_bias > 15 ||
        input_floats != element_count(shape->rows, shape->hidden_columns)) {
        errno = EINVAL;
        return -1;
    }

    struct opennpux_npu_gptq_weights gate;
    struct opennpux_npu_gptq_weights up;
    struct opennpux_npu_gptq_weights down;
    memset(&gate, 0, sizeof(gate));
    memset(&up, 0, sizeof(up));
    memset(&down, 0, sizeof(down));
    if (opennpux_npu_gptq_weights_load(
            manifest_path, model, ranges, selector->command_id,
            selector->role_id, selector->expert_id,
            OPENNPUX_NPU_WEIGHT_SLOT_GATE_PROJ, image_size, &gate) != 0 ||
        opennpux_npu_gptq_weights_load(
            manifest_path, model, ranges, selector->command_id,
            selector->role_id, selector->expert_id,
            OPENNPUX_NPU_WEIGHT_SLOT_UP_PROJ, image_size, &up) != 0 ||
        opennpux_npu_gptq_weights_load(
            manifest_path, model, ranges, selector->command_id,
            selector->role_id, selector->expert_id,
            OPENNPUX_NPU_WEIGHT_SLOT_DOWN_PROJ, image_size, &down) != 0) {
        const int saved = errno;
        opennpux_npu_gptq_weights_unload(&down);
        opennpux_npu_gptq_weights_unload(&up);
        opennpux_npu_gptq_weights_unload(&gate);
        errno = saved;
        return -1;
    }

    memset(layout, 0, sizeof(*layout));
    errno = 0;
    uint64_t cursor = sizeof(struct coral_gptq_expert_request);
    const uint64_t input_bytes = byte_size(
        element_count(shape->rows, shape->hidden_columns), sizeof(float));
    const uint64_t intermediate_bytes = byte_size(
        element_count(shape->rows, shape->intermediate_columns),
        sizeof(float));
    const uint64_t output_bytes = input_bytes;
    int status = 0;
    if (gate.scales.data_type != up.scales.data_type ||
        gate.scales.data_type != down.scales.data_type ||
        input_bytes == 0 || intermediate_bytes == 0 ||
        output_bytes > UINT32_MAX ||
        place(input_bytes, &layout->input_offset, &cursor) != 0 ||
        place(intermediate_bytes, &layout->gate_output_offset, &cursor) != 0 ||
        place(intermediate_bytes, &layout->up_output_offset, &cursor) != 0 ||
        place(intermediate_bytes, &layout->activated_offset, &cursor) != 0 ||
        place(output_bytes, &layout->output_offset, &cursor) != 0 ||
        place_weight_set(shape->hidden_columns, shape->intermediate_columns,
                         shape->group_size, &gate, &layout->gate,
                         &cursor) != 0 ||
        place_weight_set(shape->hidden_columns, shape->intermediate_columns,
                         shape->group_size, &up, &layout->up, &cursor) != 0 ||
        place_weight_set(shape->intermediate_columns, shape->hidden_columns,
                         shape->group_size, &down, &layout->down,
                         &cursor) != 0 ||
        cursor > UINT32_MAX || cursor > image_size ||
        device_base + cursor > UINT32_MAX) {
        if (errno == 0) {
            errno = cursor > image_size ? ENOSPC : EOVERFLOW;
        }
        status = -1;
        goto out;
    }
    layout->output_bytes = (uint32_t)output_bytes;
    layout->total_size = (uint32_t)cursor;

    uint8_t *bytes = image;
    memset(bytes, 0, layout->total_size);
    memcpy(bytes + layout->input_offset, input, input_bytes);
    if (copy_weight_set(&gate, &layout->gate, bytes) != 0 ||
        copy_weight_set(&up, &layout->up, bytes) != 0 ||
        copy_weight_set(&down, &layout->down, bytes) != 0) {
        status = -1;
        goto out;
    }

    struct coral_gptq_expert_request request;
    memset(&request, 0, sizeof(request));
    request.magic = CORAL_GPTQ_EXPERT_MAGIC;
    request.version = CORAL_GPTQ_EXPERT_VERSION;
    request.struct_size = sizeof(request);
    request.state = CORAL_GPTQ_EXPERT_PENDING;
    request.rows = shape->rows;
    request.hidden_columns = shape->hidden_columns;
    request.intermediate_columns = shape->intermediate_columns;
    request.group_size = shape->group_size;
    request.zero_bias = shape->zero_bias;
    request.input_address = (uint32_t)device_base + layout->input_offset;
    request.gate_output_address =
        (uint32_t)device_base + layout->gate_output_offset;
    request.up_output_address =
        (uint32_t)device_base + layout->up_output_offset;
    request.activated_address =
        (uint32_t)device_base + layout->activated_offset;
    request.output_address = (uint32_t)device_base + layout->output_offset;
    request.gate = device_weight_set((uint32_t)device_base, &layout->gate);
    request.up = device_weight_set((uint32_t)device_base, &layout->up);
    request.down = device_weight_set((uint32_t)device_base, &layout->down);
    memcpy(bytes, &request, sizeof(request));

out:
    {
        const int saved = errno;
        opennpux_npu_gptq_weights_unload(&down);
        opennpux_npu_gptq_weights_unload(&up);
        opennpux_npu_gptq_weights_unload(&gate);
        errno = saved;
    }
    return status;
}
