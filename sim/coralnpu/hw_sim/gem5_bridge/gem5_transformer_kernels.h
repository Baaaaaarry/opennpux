#ifndef HW_SIM_GEM5_BRIDGE_GEM5_TRANSFORMER_KERNELS_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_TRANSFORMER_KERNELS_H_

#include <cstddef>
#include <cstdint>

struct Gem5TransformerKernelStats {
  uint64_t operations;
  uint64_t bytes_read;
  uint64_t bytes_written;
  uint64_t modeled_cycles;
};

bool RunGem5EmbeddingF32(const uint32_t* token_ids, size_t token_count,
                         const float* table, size_t vocabulary_size,
                         size_t features, float* output,
                         Gem5TransformerKernelStats* stats);

bool RunGem5MatMulF32(const float* input, const float* weight, size_t rows,
                      size_t input_features, size_t output_features,
                      float* output, Gem5TransformerKernelStats* stats);

bool RunGem5AddF32(const float* lhs, const float* rhs, size_t count,
                   float* output, Gem5TransformerKernelStats* stats);

bool RunGem5MulF32(const float* lhs, const float* rhs, size_t count,
                   float* output, Gem5TransformerKernelStats* stats);

bool RunGem5SiluF32(const float* input, size_t count, float* output,
                    Gem5TransformerKernelStats* stats);

bool RunGem5RmsNormF32(const float* input, const float* weight, size_t rows,
                       size_t features, float epsilon, float* output,
                       Gem5TransformerKernelStats* stats);

bool RunGem5SoftmaxF32(const float* input, size_t rows, size_t features,
                       float* output, Gem5TransformerKernelStats* stats);

// Applies the standard adjacent-pair rotary embedding independently to each
// [row, head]. Positions has one entry per row.
bool RunGem5RopeF32(const float* input, const uint32_t* positions, size_t rows,
                    size_t heads, size_t head_dim, float theta, float* output,
                    Gem5TransformerKernelStats* stats);

// Returns values in descending order. Equal values retain the lower source
// index first so functional-model results are deterministic across hosts.
bool RunGem5TopKF32(const float* input, size_t count, size_t k,
                    float* output_values, uint32_t* output_indices,
                    Gem5TransformerKernelStats* stats);

bool RunGem5KvCacheUpdateF32(
    const float* key, const float* value, size_t token_count,
    size_t kv_heads, size_t head_dim, size_t kv_length, float* state,
    Gem5TransformerKernelStats* stats);

bool RunGem5AttentionF32(
    const float* query, const float* state, size_t query_rows,
    size_t heads, size_t kv_heads, size_t head_dim, size_t kv_length,
    float* output, Gem5TransformerKernelStats* stats);

bool RunGem5RecurrentUpdateF32(
    const float* input, size_t count, float* output, float* state,
    Gem5TransformerKernelStats* stats);

bool RunGem5CombineF32(
    const float* routed, const float* shared, size_t count, float* output,
    Gem5TransformerKernelStats* stats);

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_TRANSFORMER_KERNELS_H_
