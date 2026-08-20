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
  assert(RunGem5RopeF32(rope_input, positions, 2, 1, 2, 10000.0f,
                        rope_output, &stats));
  assert(rope_output[0] == 1.0f && rope_output[1] == 0.0f);
  assert(Near(rope_output[2], 2.0f * std::cos(1.0f)));
  assert(Near(rope_output[3], 2.0f * std::sin(1.0f)));

  const float logits[] = {1.0f, 3.0f, 3.0f, 2.0f};
  float top_values[2] = {};
  uint32_t top_indices[2] = {};
  assert(RunGem5TopKF32(logits, 4, 2, top_values, top_indices, &stats));
  assert(top_indices[0] == 1 && top_indices[1] == 2);
  assert(top_values[0] == 3.0f && top_values[1] == 3.0f);

  assert(!RunGem5RmsNormF32(norm_input, norm_weight, 0, 2, 0.0f,
                            norm_output, &stats));
  assert(!RunGem5RopeF32(rope_input, positions, 2, 1, 3, 10000.0f,
                         rope_output, &stats));
  assert(!RunGem5TopKF32(logits, 4, 5, top_values, top_indices, &stats));

  std::puts("gem5_transformer_kernels=PASS");
  return 0;
}
