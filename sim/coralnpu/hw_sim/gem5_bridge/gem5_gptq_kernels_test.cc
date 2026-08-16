#include "hw_sim/gem5_bridge/gem5_gptq_kernels.h"

#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

void TestGroupedMatMul() {
  const Gem5GptqMatMulConfig config = {1, 4, 2, 2, 1};
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const uint32_t qweight[] = {UINT32_C(0x00005432),
                              UINT32_C(0x00001234)};
  const uint32_t qzeros[] = {UINT32_C(0x00000011),
                             UINT32_C(0x00000022)};
  const float scales[] = {0.5f, 1.0f, 0.25f, 0.5f};
  float output[2] = {};
  Gem5GptqKernelStats stats = {};

  assert(RunGem5GptqInt4MatMul(config, input, qweight, qzeros, scales,
                               nullptr, output, &stats));
  assert(std::fabs(output[0] - 3.75f) < 1.0e-6f);
  assert(std::fabs(output[1] + 1.5f) < 1.0e-6f);
  assert(stats.operations == 16);
  assert(stats.bytes_written == sizeof(output));
  assert(stats.modeled_cycles > 8);
}

void TestExplicitGroupIndex() {
  const Gem5GptqMatMulConfig config = {1, 2, 1, 1, 0};
  const float input[] = {1.0f, 1.0f};
  const uint32_t qweight[] = {UINT32_C(0x00000021)};
  const uint32_t qzeros[] = {0, 0};
  const float scales[] = {2.0f, 3.0f};
  const uint32_t g_idx[] = {1, 0};
  float output[1] = {};
  Gem5GptqKernelStats stats = {};

  assert(RunGem5GptqInt4MatMul(config, input, qweight, qzeros, scales,
                               g_idx, output, &stats));
  assert(std::fabs(output[0] - 7.0f) < 1.0e-6f);
}

void TestRejectsInvalidGroupIndex() {
  const Gem5GptqMatMulConfig config = {1, 1, 1, 1, 0};
  const float input[] = {1.0f};
  const uint32_t packed[] = {1};
  const float scales[] = {1.0f};
  const uint32_t g_idx[] = {1};
  float output[1] = {};
  Gem5GptqKernelStats stats = {};
  assert(!RunGem5GptqInt4MatMul(config, input, packed, packed, scales,
                                g_idx, output, &stats));
}

}  // namespace

int main() {
  TestGroupedMatMul();
  TestExplicitGroupIndex();
  TestRejectsInvalidGroupIndex();
  return 0;
}
