#include "hw_sim/gem5_bridge/gem5_transformer_kernels.h"

#include <cassert>
#include <cmath>
#include <cstdio>

namespace {

bool Near(float actual, float expected, float tolerance = 1.0e-5f) {
  return std::fabs(actual - expected) <= tolerance;
}

}  // namespace

int main() {
  Gem5TransformerKernelStats stats = {};

  const uint32_t token_ids[] = {2, 0};
  const float embedding_table[] = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  float embeddings[6] = {};
  assert(RunGem5EmbeddingF32(token_ids, 2, embedding_table, 3, 3,
                             embeddings, &stats));
  assert(embeddings[0] == 7.0f && embeddings[2] == 9.0f &&
         embeddings[3] == 1.0f && embeddings[5] == 3.0f);
  assert(stats.operations == 6 && stats.bytes_written == sizeof(embeddings));

  const float matmul_input[] = {1.0f, 2.0f, 3.0f,
                                4.0f, 5.0f, 6.0f};
  const float matmul_weight[] = {1.0f, 3.0f, 5.0f,
                                 2.0f, 4.0f, 6.0f};
  float matmul_output[4] = {};
  assert(RunGem5MatMulF32(matmul_input, matmul_weight, 2, 3, 2,
                          matmul_output, &stats));
  assert(matmul_output[0] == 22.0f && matmul_output[1] == 28.0f &&
         matmul_output[2] == 49.0f && matmul_output[3] == 64.0f);
  assert(stats.operations == 24 &&
         stats.bytes_written == sizeof(matmul_output));

  const float lhs[] = {1.0f, -2.0f, 3.0f};
  const float rhs[] = {4.0f, 5.0f, -6.0f};
  float elementwise[3] = {};
  assert(RunGem5AddF32(lhs, rhs, 3, elementwise, &stats));
  assert(elementwise[0] == 5.0f && elementwise[1] == 3.0f &&
         elementwise[2] == -3.0f);
  assert(stats.operations == 3 && stats.bytes_written == sizeof(elementwise));
  assert(RunGem5MulF32(lhs, rhs, 3, elementwise, &stats));
  assert(elementwise[0] == 4.0f && elementwise[1] == -10.0f &&
         elementwise[2] == -18.0f);

  const float silu_input[] = {0.0f, 1.0f, -1.0f};
  float silu_output[3] = {};
  assert(RunGem5SiluF32(silu_input, 3, silu_output, &stats));
  assert(Near(silu_output[0], 0.0f));
  assert(Near(silu_output[1], 0.7310586f));
  assert(Near(silu_output[2], -0.2689414f));

  const float norm_input[] = {3.0f, 4.0f};
  const float norm_weight[] = {1.0f, 2.0f};
  float norm_output[2] = {};
  assert(RunGem5RmsNormF32(norm_input, norm_weight, 1, 2, 0.0f,
                           norm_output, &stats));
  assert(Near(norm_output[0], 0.8485281f));
  assert(Near(norm_output[1], 2.2627417f));
  const float zero_norm_weight[] = {0.0f, 0.0f};
  assert(RunGem5RmsNormF32(norm_input, zero_norm_weight, 1, 2, 0.0f,
                           norm_output, &stats, true));
  assert(Near(norm_output[0], 0.8485281f));
  assert(Near(norm_output[1], 1.1313708f));

  const float softmax_input[] = {0.0f, 0.0f, 0.0f, 0.0f,
                                 1000.0f, 999.0f, 998.0f, 997.0f};
  float softmax_output[8] = {};
  assert(RunGem5SoftmaxF32(softmax_input, 2, 4, softmax_output, &stats));
  for (size_t index = 0; index < 4; ++index) {
    assert(Near(softmax_output[index], 0.25f));
  }
  assert(Near(softmax_output[4] + softmax_output[5] + softmax_output[6] +
                  softmax_output[7],
              1.0f));

  const float rope_input[] = {1.0f, 0.0f, 2.0f, 0.0f};
  const uint32_t positions[] = {0, 1};
  float rope_output[4] = {};
  assert(RunGem5RopeF32(rope_input, positions, 2, 1, 2, 2, 10000.0f,
                        rope_output, &stats));
  assert(rope_output[0] == 1.0f && rope_output[1] == 0.0f);
  assert(Near(rope_output[2], 2.0f * std::cos(1.0f)));
  assert(Near(rope_output[3], 2.0f * std::sin(1.0f)));
  const float partial_rope_input[] = {2.0f, 3.0f, 5.0f, 7.0f};
  float partial_rope_output[4] = {};
  const uint32_t partial_position[] = {1};
  assert(RunGem5RopeF32(partial_rope_input, partial_position, 1, 1, 4, 2,
                        10000.0f, partial_rope_output, &stats));
  assert(Near(partial_rope_output[0],
              2.0f * std::cos(1.0f) - 3.0f * std::sin(1.0f)));
  assert(Near(partial_rope_output[1],
              3.0f * std::cos(1.0f) + 2.0f * std::sin(1.0f)));
  assert(partial_rope_output[2] == 5.0f && partial_rope_output[3] == 7.0f);

  const float conv_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float conv_weight[] = {0.25f, 0.5f, 1.0f,
                               1.0f, -0.5f, 0.25f};
  float conv_full[6] = {};
  float conv_prefill[6] = {};
  float conv_prefill_state[4] = {};
  const float conv_zero_state[4] = {};
  assert(RunGem5CausalDepthwiseConvF32(
      conv_input, conv_weight, 3, 2, 3, conv_full, &stats));
  assert(RunGem5CausalDepthwiseConvStatefulF32(
      conv_input, conv_weight, 3, 2, 3, conv_zero_state,
      conv_prefill_state, conv_prefill, &stats));
  float conv_decode[6] = {};
  float conv_decode_state[4] = {};
  for (size_t row = 0; row < 3; ++row) {
    float next_state[4] = {};
    assert(RunGem5CausalDepthwiseConvStatefulF32(
        conv_input + row * 2, conv_weight, 1, 2, 3, conv_decode_state,
        next_state, conv_decode + row * 2, &stats));
    std::copy(std::begin(next_state), std::end(next_state),
              std::begin(conv_decode_state));
  }
  for (size_t index = 0; index < 6; ++index) {
    assert(Near(conv_full[index], conv_prefill[index]));
    assert(Near(conv_full[index], conv_decode[index]));
  }

  const float logits[] = {1.0f, 3.0f, 3.0f, 2.0f};
  float top_values[2] = {};
  uint32_t top_indices[2] = {};
  assert(RunGem5TopKF32(logits, 4, 2, top_values, top_indices, &stats));
  assert(top_indices[0] == 1 && top_indices[1] == 2);
  assert(top_values[0] == 3.0f && top_values[1] == 3.0f);

  const float key[] = {1.0f, 0.0f, 0.0f, 1.0f};
  const float value[] = {2.0f, 0.0f, 0.0f, 4.0f};
  float kv_state[8] = {};
  assert(RunGem5KvCacheUpdateF32(key, value, 2, 1, 2, 2, kv_state,
                                 &stats));
  assert(kv_state[0] == 1.0f && kv_state[3] == 1.0f &&
         kv_state[4] == 2.0f && kv_state[7] == 4.0f);
  const float query[] = {1.0f, 0.0f};
  float attention_output[2] = {};
  assert(RunGem5AttentionF32(query, kv_state, 1, 1, 1, 2, 2,
                             attention_output, &stats));
  assert(attention_output[0] > 1.0f && attention_output[0] < 2.0f);
  assert(attention_output[1] > 0.0f && attention_output[1] < 2.0f);

  const float causal_query[] = {1.0f, 0.0f, 1.0f, 0.0f};
  float causal_output[4] = {};
  assert(RunGem5AttentionF32(causal_query, kv_state, 2, 1, 1, 2, 2,
                             causal_output, &stats));
  assert(causal_output[0] == 2.0f && causal_output[1] == 0.0f);
  assert(causal_output[2] > 1.0f && causal_output[2] < 2.0f);
  assert(causal_output[3] > 0.0f && causal_output[3] < 2.0f);

  const float grouped_query[] = {1.0f, 0.0f, 1.0f, 0.0f,
                                 1.0f, 0.0f, 1.0f, 0.0f};
  const float grouped_state[] = {
      1.0f, 0.0f, 0.0f, 1.0f,
      10.0f, 0.0f, 20.0f, 0.0f,
  };
  float grouped_output[8] = {};
  assert(RunGem5AttentionF32(grouped_query, grouped_state, 1, 4, 2, 2, 1,
                             grouped_output, &stats));
  assert(grouped_output[0] == 10.0f && grouped_output[2] == 10.0f);
  assert(grouped_output[4] == 20.0f && grouped_output[6] == 20.0f);

  float recurrent_output[3] = {};
  float recurrent_state[3] = {};
  assert(RunGem5RecurrentUpdateF32(lhs, 1, 3, recurrent_output,
                                   recurrent_state, &stats));
  assert(recurrent_output[2] == 3.0f && recurrent_state[1] == -2.0f);
  const float delta_qkv[] = {1.0f, 1.0f, 2.0f};
  const float delta_alpha[] = {0.0f};
  const float delta_beta[] = {0.0f};
  const float delta_a_log[] = {0.0f};
  const float delta_dt_bias[] = {0.0f};
  float delta_output[] = {0.0f};
  float delta_state[] = {0.0f};
  assert(RunGem5GatedDeltaNetF32(
      delta_qkv, delta_alpha, delta_beta, delta_a_log, delta_dt_bias, 1, 1,
      1, 1, 1, delta_output, delta_state, &stats));
  assert(Near(delta_state[0], 1.0f, 1.0e-4f));
  assert(Near(delta_output[0], 1.0f, 1.0e-4f));
  assert(RunGem5CombineF32(lhs, rhs, 3, recurrent_output, &stats));
  assert(recurrent_output[0] == 5.0f && recurrent_output[2] == -3.0f);

  const float shared_input[] = {1.0f, 2.0f};
  const float shared_gate_weight[] = {1.0f, 0.0f, 0.0f, 1.0f};
  const float shared_up_weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
  const float shared_down_weight[] = {1.0f, 0.0f, 0.0f, 1.0f};
  const float shared_router_weight[] = {0.0f, 0.0f};
  float shared_output[2] = {};
  assert(RunGem5SharedExpertF32(
      shared_input, shared_gate_weight, shared_up_weight, shared_down_weight,
      shared_router_weight, 1, 2, 2, 2, shared_output, &stats));
  assert(Near(shared_output[0], 1.5f * 0.7310586f));
  assert(Near(shared_output[1], 3.0f * 0.8807971f));
  assert(stats.operations != 0 && stats.modeled_cycles != 0);

  const float gated_q_weight[] = {1.0f, 0.0f, 0.0f, 1.0f,
                                  10.0f, 10.0f, 10.0f, 10.0f};
  const float identity_weight[] = {1.0f, 0.0f, 0.0f, 1.0f};
  const float head_norm[] = {1.0f, 1.0f};
  float query_output[2] = {};
  float key_output[2] = {};
  float value_output[2] = {};
  float attention_gate[2] = {};
  assert(RunGem5FloatQkvF32(
      shared_input, gated_q_weight, identity_weight, identity_weight,
      head_norm, head_norm, 1, 2, 1, 1, 2, 4, 0.0f, false, query_output,
      key_output, value_output, attention_gate, &stats));
  assert(Near(query_output[0], 0.6324555f));
  assert(Near(query_output[1], 1.2649110f));
  assert(Near(key_output[0], 0.6324555f));
  assert(Near(key_output[1], 1.2649110f));
  assert(value_output[0] == 1.0f && value_output[1] == 2.0f);
  assert(attention_gate[0] == 30.0f && attention_gate[1] == 30.0f);

  const float multihead_q_weight[] = {
      1.0f, 0.0f, 10.0f, 10.0f,
      0.0f, 1.0f, 20.0f, 20.0f};
  float multihead_query[2] = {};
  float multihead_gate[2] = {};
  assert(RunGem5FloatQkvF32(
      shared_input, multihead_q_weight, identity_weight, identity_weight,
      head_norm, head_norm, 1, 2, 2, 2, 1, 4, 0.0f, false,
      multihead_query, key_output, value_output, multihead_gate, &stats));
  assert(Near(multihead_query[0], 1.0f));
  assert(Near(multihead_query[1], 1.0f));
  assert(multihead_gate[0] == 30.0f && multihead_gate[1] == 60.0f);

  assert(!RunGem5RmsNormF32(norm_input, norm_weight, 0, 2, 0.0f,
                            norm_output, &stats));
  assert(!RunGem5RopeF32(rope_input, positions, 2, 1, 3, 3, 10000.0f,
                         rope_output, &stats));
  assert(!RunGem5TopKF32(logits, 4, 5, top_values, top_indices, &stats));
  const uint32_t invalid_token[] = {3};
  assert(!RunGem5EmbeddingF32(invalid_token, 1, embedding_table, 3, 3,
                              embeddings, &stats));
  assert(!RunGem5MatMulF32(matmul_input, matmul_weight, 0, 3, 2,
                           matmul_output, &stats));

  std::puts("gem5_transformer_kernels=PASS");
  return 0;
}
