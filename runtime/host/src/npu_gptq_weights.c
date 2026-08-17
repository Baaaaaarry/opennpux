#include "opennpux/npu_gptq_weights.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static int
load_blob(const char *manifest_path,
          const struct opennpux_model_package_info *model,
          const struct opennpux_npu_weight_range_record *range,
          size_t max_component_size, struct opennpux_npu_weight_blob *blob)
{
    if (range == NULL) {
        return 0;
    }
    if (range->byte_size == 0 || range->byte_size > max_component_size) {
        errno = EFBIG;
        return -1;
    }
    blob->data = malloc((size_t)range->byte_size);
    if (blob->data == NULL) {
        return -1;
    }
    blob->size = (size_t)range->byte_size;
    blob->data_type = (uint32_t)(
        (range->flags & OPENNPUX_NPU_WEIGHT_DTYPE_MASK) >>
        OPENNPUX_NPU_WEIGHT_DTYPE_SHIFT);
    if (opennpux_model_package_read_shard_range(
            manifest_path, model, range->shard_index, range->file_offset,
            blob->data, range->byte_size) != 0) {
        free(blob->data);
        memset(blob, 0, sizeof(*blob));
        return -1;
    }
    return 0;
}

void
opennpux_npu_gptq_weights_unload(
    struct opennpux_npu_gptq_weights *weights)
{
    if (weights == NULL) {
        return;
    }
    free(weights->qweight.data);
    free(weights->qzeros.data);
    free(weights->scales.data);
    free(weights->g_idx.data);
    memset(weights, 0, sizeof(*weights));
}

int
opennpux_npu_gptq_weights_load(
    const char *manifest_path,
    const struct opennpux_model_package_info *model,
    const struct opennpux_npu_weight_ranges *ranges, uint32_t command_id,
    uint32_t role_id, uint64_t expert_id, uint32_t slot_id,
    size_t max_component_size,
    struct opennpux_npu_gptq_weights *weights)
{
    if (manifest_path == NULL || model == NULL || ranges == NULL ||
        max_component_size == 0 || weights == NULL) {
        errno = EINVAL;
        return -1;
    }
    memset(weights, 0, sizeof(*weights));
    struct opennpux_npu_gptq_weight_ranges components;
    if (opennpux_npu_weight_ranges_find_gptq(
            ranges, command_id, role_id, expert_id, slot_id,
            &components) != 0 ||
        load_blob(manifest_path, model, components.qweight,
                  max_component_size, &weights->qweight) != 0 ||
        load_blob(manifest_path, model, components.qzeros,
                  max_component_size, &weights->qzeros) != 0 ||
        load_blob(manifest_path, model, components.scales,
                  max_component_size, &weights->scales) != 0 ||
        load_blob(manifest_path, model, components.g_idx,
                  max_component_size, &weights->g_idx) != 0) {
        opennpux_npu_gptq_weights_unload(weights);
        return -1;
    }
    return 0;
}
