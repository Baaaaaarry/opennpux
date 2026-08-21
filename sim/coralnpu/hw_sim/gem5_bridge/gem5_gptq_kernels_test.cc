#include "hw_sim/gem5_bridge/gem5_gptq_kernels.h"

#include <cassert>
#include <cstring>

namespace {

struct ReaderBuffers {
  const void* components[4];
  size_t sizes[4];
};

bool ReadComponent(void* opaque, Gem5GptqComponent component,
                   uint64_t offset, void* destination, size_t size) {
  auto* buffers = static_cast<ReaderBuffers*>(opaque);
  const size_t index = static_cast<size_t>(component);
  if (index >= 4 || offset > buffers->sizes[index] ||
      size > buffers->sizes[index] - offset) {
    return false;
  }
  std::memcpy(destination,
              static_cast<const uint8_t*>(buffers->components[index]) + offset,
              size);
  return true;
}

}  // namespace
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

void TestGroupedMatMul() {
  const Gem5GptqMatMulConfig config = {
      1, 4, 2, 2, 1, kGem5GptqScaleFloat32};
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
  const Gem5GptqMatMulConfig config = {
      1, 2, 1, 1, 0, kGem5GptqScaleFloat32};
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
  const Gem5GptqMatMulConfig config = {
      1, 1, 1, 1, 0, kGem5GptqScaleFloat32};
  const float input[] = {1.0f};
  const uint32_t packed[] = {1};
  const float scales[] = {1.0f};
  const uint32_t g_idx[] = {1};
  float output[1] = {};
  Gem5GptqKernelStats stats = {};
  assert(!RunGem5GptqInt4MatMul(config, input, packed, packed, scales,
                                g_idx, output, &stats));
}

void TestFloat16Scales() {
  const Gem5GptqMatMulConfig config = {
      1, 2, 1, 1, 0, kGem5GptqScaleFloat16};
  const float input[] = {1.0f, 1.0f};
  const uint32_t qweight[] = {UINT32_C(0x21)};
  const uint32_t qzeros[] = {0, 0};
  const uint16_t scales[] = {UINT16_C(0x4000), UINT16_C(0x4200)};
  float output[1] = {};
  Gem5GptqKernelStats stats = {};
  assert(RunGem5GptqInt4MatMul(config, input, qweight, qzeros, scales,
                               nullptr, output, &stats));
  assert(std::fabs(output[0] - 8.0f) < 1.0e-6f);
  assert(stats.bytes_read == 24);
}

void TestBfloat16Scales() {
  const Gem5GptqMatMulConfig config = {
      1, 2, 1, 1, 0, kGem5GptqScaleBfloat16};
  const float input[] = {1.0f, 1.0f};
  const uint32_t qweight[] = {UINT32_C(0x21)};
  const uint32_t qzeros[] = {0, 0};
  const uint16_t scales[] = {UINT16_C(0x4000), UINT16_C(0x4040)};
  float output[1] = {};
  Gem5GptqKernelStats stats = {};
  assert(RunGem5GptqInt4MatMul(config, input, qweight, qzeros, scales,
                               nullptr, output, &stats));
  assert(std::fabs(output[0] - 8.0f) < 1.0e-6f);
}

void TestStoredMaximumZeroWithBias() {
  const Gem5GptqMatMulConfig config = {
      1, 1, 1, 1, 1, kGem5GptqScaleFloat32};
  const float input[] = {2.0f};
  const uint32_t qweight[] = {UINT32_C(0x0000000f)};
  const uint32_t qzeros[] = {UINT32_C(0x0000000f)};
  const float scales[] = {0.5f};
  float output[1] = {};
  Gem5GptqKernelStats stats = {};

  assert(RunGem5GptqInt4MatMul(config, input, qweight, qzeros, scales,
                               nullptr, output, &stats));
  assert(output[0] == -1.0f);
}

// Shared vector with the host reference in
// tests/unit/runtime_host/npu_gptq_reference_test.c. Both implementations must
// produce identical float32 values and the same FNV-1a output checksum,
// because the full-system projection test compares the device checksum against
// the host expectation.
void TestHostReferenceVector() {
  const Gem5GptqMatMulConfig config = {
      1, 8, 8, 8, 1, kGem5GptqScaleFloat32};
  float input[8];
  uint32_t qweight[8];
  float scales[8];
  for (unsigned index = 0; index < 8; ++index) {
    input[index] = 1.0f;
    qweight[index] = UINT32_C(0x76543210);
    scales[index] = 1.0f;
  }
  const uint32_t qzeros[] = {0};
  float output[8] = {};
  Gem5GptqKernelStats stats = {};

  assert(RunGem5GptqInt4MatMul(config, input, qweight, qzeros, scales,
                               nullptr, output, &stats));
  for (unsigned index = 0; index < 8; ++index) {
    assert(output[index] == 20.0f);
  }
  assert(stats.operations == 128);
  assert(stats.bytes_read == 100);
  assert(stats.bytes_written == 32);
  assert(stats.modeled_cycles == 73);

  uint32_t checksum = UINT32_C(2166136261);
  const auto* bytes = reinterpret_cast<const uint8_t*>(output);
  for (std::size_t index = 0; index < sizeof(output); ++index) {
    checksum ^= bytes[index];
    checksum *= UINT32_C(16777619);
  }
  assert(checksum == UINT32_C(0x5bea2f85));
}

void TestStreamedOutputTilesMatchContiguousKernel() {
  const Gem5GptqMatMulConfig config = {
      2, 8, 16, 8, 1, kGem5GptqScaleFloat32};
  float input[16];
  uint32_t qweight[16];
  uint32_t qzeros[2] = {};
  float scales[16];
  for (size_t index = 0; index < 16; ++index) {
    input[index] = 1.0f;
    qweight[index] = UINT32_C(0x76543210);
    scales[index] = 1.0f;
  }
  float expected[32] = {};
  float streamed[32] = {};
  Gem5GptqKernelStats expected_stats = {};
  Gem5GptqKernelStats streamed_stats = {};
  assert(RunGem5GptqInt4MatMul(config, input, qweight, qzeros, scales,
                               nullptr, expected, &expected_stats));
  ReaderBuffers reader = {
      {qweight, qzeros, scales, nullptr},
      {sizeof(qweight), sizeof(qzeros), sizeof(scales), 0}};
  assert(RunGem5GptqInt4MatMulStreamed(
      config, input, ReadComponent, &reader, 8, false, streamed,
      &streamed_stats));
  for (size_t index = 0; index < 32; ++index) {
    assert(streamed[index] == expected[index]);
  }
  assert(streamed_stats.operations == expected_stats.operations);
  assert(!RunGem5GptqInt4MatMulStreamed(
      config, input, ReadComponent, &reader, 7, false, streamed,
      &streamed_stats));
}

}  // namespace

int main() {
  TestGroupedMatMul();
  TestExplicitGroupIndex();
  TestRejectsInvalidGroupIndex();
  TestFloat16Scales();
  TestBfloat16Scales();
  TestStoredMaximumZeroWithBias();
  TestHostReferenceVector();
  TestStreamedOutputTilesMatchContiguousKernel();
  return 0;
}
