#include "hw_sim/gem5_bridge/gem5_tmma_coprocessor.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr uint32_t kMemoryBase = 0x20000000;

void WriteFloat(std::vector<uint8_t>* memory, size_t offset, float value) {
  std::memcpy(memory->data() + offset, &value, sizeof(value));
}

void WriteUint32(std::vector<uint8_t>* memory, size_t offset,
                 uint32_t value) {
  std::memcpy(memory->data() + offset, &value, sizeof(value));
}

float ReadFloat(const std::vector<uint8_t>& memory, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, memory.data() + offset, sizeof(value));
  return value;
}

uint32_t ReadUint32(const std::vector<uint8_t>& memory, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, memory.data() + offset, sizeof(value));
  return value;
}

uint32_t Fnv1a(const std::vector<uint8_t>& memory, size_t offset,
               size_t size) {
  uint32_t hash = 2166136261u;
  for (size_t index = 0; index < size; ++index) {
    hash = (hash ^ memory[offset + index]) * 16777619u;
  }
  return hash;
}

Gem5TmmaDispatchPacket Packet(uint32_t sequence_id) {
  Gem5TmmaDispatchPacket packet;
  packet.instruction = xopennpux::EncodeTmma(12, 10, 11);
  packet.pc = 0x100;
  packet.rs1_value = kMemoryBase;
  packet.rs2_value = kMemoryBase + 0x100;
  packet.rd_value = kMemoryBase + 0x200;
  packet.sequence_id = sequence_id;
  return packet;
}

void ConfigureFp32(Gem5TmmaCoprocessor* coprocessor, uint32_t m, uint32_t n,
                   uint32_t k) {
  assert(coprocessor->WriteCsr(xopennpux::kCsrMmaShape,
                               xopennpux::EncodeMmaShape(m, n, k)));
  assert(coprocessor->WriteCsr(
      xopennpux::kCsrMmaDataType,
      xopennpux::EncodeMmaDataTypes(xopennpux::DataType::kFp32,
                                    xopennpux::DataType::kFp32,
                                    xopennpux::DataType::kFp32)));
}

void ConfigureTensorFp32(Gem5XOpenNpuFunctionalCoprocessor* coprocessor,
                         uint32_t rows, uint32_t features) {
  assert(coprocessor->WriteCsr(xopennpux::kCsrTensorShape,
                               xopennpux::EncodeTensorShape(rows, features)));
  assert(coprocessor->WriteCsr(
      xopennpux::kCsrTensorDataType,
      xopennpux::EncodeMmaDataTypes(xopennpux::DataType::kFp32,
                                    xopennpux::DataType::kFp32,
                                    xopennpux::DataType::kFp32)));
}

void TestEncodingAndSecondLevelDecode() {
  const uint32_t instruction = xopennpux::EncodeTmma(12, 10, 11);
  assert((instruction & 0x7f) == 0x7b);
  assert(((instruction >> 7) & 0x1f) == 12);
  assert(((instruction >> 15) & 0x1f) == 10);
  assert(((instruction >> 20) & 0x1f) == 11);
  assert(xopennpux::IsTmma(instruction));
  assert(!xopennpux::IsTmma(instruction | (1u << 25)));
  assert(xopennpux::IsTfence(xopennpux::EncodeTfence()));
  assert(!xopennpux::IsTmma(xopennpux::EncodeTfence()));
  const uint32_t tadd = xopennpux::EncodeTadd(12, 10, 11);
  assert(xopennpux::IsTadd(tadd));
  assert(!xopennpux::IsTmma(tadd));
  assert(xopennpux::DecodeOperation(tadd) ==
         xopennpux::Operation::kTadd);
  const uint32_t row_scale = xopennpux::EncodeTrowScale(12, 10, 11);
  assert(xopennpux::IsTrowScale(row_scale));
  assert(xopennpux::DecodeOperation(row_scale) ==
         xopennpux::Operation::kTrowScale);
  const uint32_t rmsnorm = xopennpux::EncodeTrmsnorm(12, 10, 11);
  assert(xopennpux::IsTrmsnorm(rmsnorm));
  assert(xopennpux::DecodeOperation(rmsnorm) ==
         xopennpux::Operation::kTrmsnorm);
  const uint32_t softmax = xopennpux::EncodeTsoftmax(12, 10);
  assert(xopennpux::IsTsoftmax(softmax));
  assert(xopennpux::DecodeOperation(softmax) ==
         xopennpux::Operation::kTsoftmax);
  assert(!xopennpux::IsTsoftmax(softmax | (1u << 20)));
  const uint32_t rope = xopennpux::EncodeTrope(12, 10, 11);
  assert(xopennpux::IsTrope(rope));
  assert(xopennpux::DecodeOperation(rope) ==
         xopennpux::Operation::kTrope);
  const uint32_t silu = xopennpux::EncodeTsilu(12, 10);
  assert(xopennpux::IsTsilu(silu));
  assert(((silu >> 20) & 0x1f) == 0);
  assert(xopennpux::DecodeOperation(silu) ==
         xopennpux::Operation::kTsilu);
  assert(!xopennpux::IsTsilu(silu | (1u << 20)));
  const uint32_t sigmoid = xopennpux::EncodeTsigmoid(12, 10);
  assert(xopennpux::IsTsigmoid(sigmoid));
  assert(xopennpux::DecodeOperation(sigmoid) ==
         xopennpux::Operation::kTsigmoid);
  assert(!xopennpux::IsTsigmoid(sigmoid | (1u << 20)));
  const uint32_t gather = xopennpux::EncodeTgather(12, 10, 11);
  assert(xopennpux::IsTgather(gather));
  assert(xopennpux::DecodeOperation(gather) ==
         xopennpux::Operation::kTgather);
  const uint32_t topk = xopennpux::EncodeTtopk(12, 10);
  assert(xopennpux::IsTtopk(topk));
  assert(xopennpux::DecodeOperation(topk) ==
         xopennpux::Operation::kTtopk);
  assert(!xopennpux::IsTtopk(topk | (1u << 20)));
  const uint32_t dequant = xopennpux::EncodeTdequant(12, 10);
  assert(xopennpux::IsTdequant(dequant));
  assert(((dequant >> 20) & 0x1f) == 0);
  assert(xopennpux::DecodeOperation(dequant) ==
         xopennpux::Operation::kTdequant);
  assert(!xopennpux::IsTdequant(dequant | (1u << 20)));
  const uint32_t dma = xopennpux::EncodeTdma(12, 10);
  assert(xopennpux::IsTdma(dma));
  assert(xopennpux::DecodeOperation(dma) == xopennpux::Operation::kTdma);
  assert(!xopennpux::IsTdma(dma | (1u << 20)));
  const uint32_t causal_conv =
      xopennpux::EncodeTcausalconv(12, 10, 11);
  assert(xopennpux::IsTcausalconv(causal_conv));
  assert(xopennpux::DecodeOperation(causal_conv) ==
         xopennpux::Operation::kTcausalconv);
  const uint32_t attention = xopennpux::EncodeTattention(12, 10, 11);
  assert(xopennpux::IsTattention(attention));
  assert(xopennpux::DecodeOperation(attention) ==
         xopennpux::Operation::kTattention);
  const uint32_t recurrent = xopennpux::EncodeTrecurrent(12, 10, 11);
  assert(xopennpux::IsTrecurrent(recurrent));
  assert(xopennpux::DecodeOperation(recurrent) ==
         xopennpux::Operation::kTrecurrent);
  const uint32_t conv = xopennpux::EncodeTconv(12, 10, 11);
  assert(xopennpux::IsTconv(conv));
  assert(xopennpux::DecodeOperation(conv) ==
         xopennpux::Operation::kTconv);
}

void TestFp32GroupedConv2d() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 2);
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvInputHw,
                              3u | (3u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvOutputHw,
                              2u | (2u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvChannelsGroups,
                              2u | (2u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvKernelHw,
                              2u | (2u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvStrideHw,
                              1u | (1u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvPaddingTl, 0));
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvPaddingBr, 0));
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvDilationHw,
                              1u | (1u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrConvBiasAddress,
                              kMemoryBase + 0x300));

  Gem5TmmaDispatchPacket packet = Packet(34);
  packet.instruction = xopennpux::EncodeTconv(12, 10, 11);
  packet.rs1_value = kMemoryBase + 0x100;
  packet.rs2_value = kMemoryBase + 0x200;
  packet.rd_value = kMemoryBase + 0x400;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  for (uint32_t pixel = 0; pixel < 9; ++pixel) {
    WriteFloat(&memory, 0x100 + (pixel * 2) * 4,
               static_cast<float>(pixel + 1));
    WriteFloat(&memory, 0x100 + (pixel * 2 + 1) * 4,
               static_cast<float>((pixel + 1) * 10));
  }
  for (uint32_t index = 0; index < 8; ++index) {
    WriteFloat(&memory, 0x200 + index * 4, 1.0f);
  }
  WriteFloat(&memory, 0x300, 1.0f);
  WriteFloat(&memory, 0x304, -1.0f);

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTconv);
  assert(completion.mac_operations == 32);
  assert(completion.modeled_cycles == 32);
  const float expected[] = {13, 119, 17, 159, 25, 239, 29, 279};
  for (size_t index = 0; index < 8; ++index) {
    assert(ReadFloat(memory, 0x400 + index * 4) == expected[index]);
  }
}

void TestGatedRecurrentUpdate() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 1);
  assert(coprocessor.WriteCsr(xopennpux::kCsrRecurrentHeads,
                              1u | (1u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrRecurrentDims,
                              1u | (1u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrRecurrentBetaAddress,
                              kMemoryBase + 0x300));
  assert(coprocessor.WriteCsr(xopennpux::kCsrTensorAuxDestinationAddress,
                              kMemoryBase + 0x400));
  assert(coprocessor.WriteCsr(xopennpux::kCsrRecurrentALogAddress,
                              kMemoryBase + 0x500));
  assert(coprocessor.WriteCsr(xopennpux::kCsrRecurrentDtBiasAddress,
                              kMemoryBase + 0x600));

  Gem5TmmaDispatchPacket packet = Packet(33);
  packet.instruction = xopennpux::EncodeTrecurrent(12, 10, 11);
  packet.rs1_value = kMemoryBase + 0x100;
  packet.rs2_value = kMemoryBase + 0x200;
  packet.rd_value = kMemoryBase + 0x700;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  WriteFloat(&memory, 0x100, 1.0f);  // Q
  WriteFloat(&memory, 0x104, 1.0f);  // K
  WriteFloat(&memory, 0x108, 2.0f);  // V
  WriteFloat(&memory, 0x200, 0.0f);  // alpha
  WriteFloat(&memory, 0x300, 0.0f);  // beta
  WriteFloat(&memory, 0x400, 0.0f);  // state
  WriteFloat(&memory, 0x500, 0.0f);  // A-log
  WriteFloat(&memory, 0x600, 0.0f);  // dt-bias

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTrecurrent);
  assert(completion.element_operations == 21);
  assert(completion.modeled_cycles == 21);
  assert(std::fabs(ReadFloat(memory, 0x400) - 1.0f) < 2.0e-6f);
  assert(std::fabs(ReadFloat(memory, 0x700) - 1.0f) < 3.0e-6f);
}

void TestCausalGqaAttention() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 4);
  assert(coprocessor.WriteCsr(xopennpux::kCsrAttentionHeads,
                              2u | (1u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrAttentionHeadDimFlags, 2));
  assert(coprocessor.WriteCsr(xopennpux::kCsrAttentionKvLength, 3));

  Gem5TmmaDispatchPacket packet = Packet(32);
  packet.instruction = xopennpux::EncodeTattention(12, 10, 11);
  packet.rs1_value = kMemoryBase + 0x100;
  packet.rs2_value = kMemoryBase + 0x200;
  packet.rd_value = kMemoryBase + 0x400;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float query[] = {1, 0, 0, 1, 1, 1, 1, -1};
  const float state[] = {
      1, 0, 0, 1, 1, 1,  // K plane
      1, 2, 3, 4, 5, 6,  // V plane
  };
  for (size_t index = 0; index < 8; ++index) {
    WriteFloat(&memory, 0x100 + index * 4, query[index]);
  }
  for (size_t index = 0; index < 12; ++index) {
    WriteFloat(&memory, 0x200 + index * 4, state[index]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTattention);
  assert(completion.element_operations == 80);
  assert(completion.modeled_cycles == 80);
  const float scale = 1.0f / std::sqrt(2.0f);
  const float exp_scale = std::exp(scale);
  const float first0 = (exp_scale * 1.0f + 3.0f) / (exp_scale + 1.0f);
  const float first1 = (exp_scale * 2.0f + 4.0f) / (exp_scale + 1.0f);
  assert(std::fabs(ReadFloat(memory, 0x400) - first0) < 1.0e-5f);
  assert(std::fabs(ReadFloat(memory, 0x404) - first1) < 1.0e-5f);
  assert(std::fabs(ReadFloat(memory, 0x408) -
                   (1.0f + exp_scale * 3.0f) / (1.0f + exp_scale)) <
         1.0e-5f);
  assert(std::isfinite(ReadFloat(memory, 0x41c)));
}

void TestGatedAttention() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 1);
  assert(coprocessor.WriteCsr(xopennpux::kCsrAttentionHeads,
                              1u | (1u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrAttentionHeadDimFlags,
                              1u | (1u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrAttentionKvLength, 1));
  assert(coprocessor.WriteCsr(xopennpux::kCsrTensorAuxSourceAddress,
                              kMemoryBase + 0x300));

  Gem5TmmaDispatchPacket packet = Packet(34);
  packet.instruction = xopennpux::EncodeTattention(12, 10, 11);
  packet.rs1_value = kMemoryBase + 0x100;
  packet.rs2_value = kMemoryBase + 0x200;
  packet.rd_value = kMemoryBase + 0x400;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  WriteFloat(&memory, 0x100, 2.0f);
  WriteFloat(&memory, 0x200, 2.0f);  // K
  WriteFloat(&memory, 0x204, 3.0f);  // V
  WriteFloat(&memory, 0x300, 0.0f);  // sigmoid gate = 0.5
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.element_operations == 8);
  assert(completion.modeled_cycles == 8);
  assert(std::fabs(ReadFloat(memory, 0x400) - 1.5f) < 1.0e-6f);
}

void TestStatefulCausalDepthwiseConv() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 2);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0,
                              3u | (1u << 16)));
  assert(coprocessor.WriteCsr(xopennpux::kCsrTensorAuxSourceAddress,
                              kMemoryBase + 0x300));
  assert(coprocessor.WriteCsr(xopennpux::kCsrTensorAuxDestinationAddress,
                              kMemoryBase + 0x400));

  Gem5TmmaDispatchPacket packet = Packet(31);
  packet.instruction = xopennpux::EncodeTcausalconv(12, 10, 11);
  packet.rs1_value = kMemoryBase + 0x100;
  packet.rs2_value = kMemoryBase + 0x200;
  packet.rd_value = kMemoryBase + 0x500;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float input[] = {3.0f, 30.0f, 4.0f, 40.0f};
  const float weight[] = {1.0f, 2.0f, 3.0f, 0.1f, 0.2f, 0.3f};
  const float state[] = {1.0f, 10.0f, 2.0f, 20.0f};
  for (size_t index = 0; index < 4; ++index) {
    WriteFloat(&memory, 0x100 + index * sizeof(float), input[index]);
    WriteFloat(&memory, 0x300 + index * sizeof(float), state[index]);
  }
  for (size_t index = 0; index < 6; ++index) {
    WriteFloat(&memory, 0x200 + index * sizeof(float), weight[index]);
  }
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTcausalconv);
  assert(completion.element_operations == 24);
  assert(completion.modeled_cycles == 24);
  assert(std::fabs(ReadFloat(memory, 0x500) - 14.0f) < 1.0e-5f);
  assert(std::fabs(ReadFloat(memory, 0x504) - 14.0f) < 1.0e-5f);
  assert(std::fabs(ReadFloat(memory, 0x508) - 20.0f) < 1.0e-5f);
  assert(std::fabs(ReadFloat(memory, 0x50c) - 20.0f) < 1.0e-5f);
  for (size_t index = 0; index < 4; ++index) {
    assert(ReadFloat(memory, 0x400 + index * sizeof(float)) == input[index]);
  }
}

void ConfigureInt4Dequant(Gem5XOpenNpuFunctionalCoprocessor* coprocessor,
                          uint32_t n, uint32_t k, bool has_g_idx) {
  assert(coprocessor->WriteCsr(xopennpux::kCsrMmaShape,
                               xopennpux::EncodeMmaShape(1, n, k)));
  assert(coprocessor->WriteCsr(
      xopennpux::kCsrMmaDataType,
      xopennpux::EncodeMmaDataTypes(xopennpux::DataType::kInt4,
                                    xopennpux::DataType::kFp32,
                                    xopennpux::DataType::kFp32)));
  assert(coprocessor->WriteCsr(xopennpux::kCsrQuantQzerosAddress,
                               kMemoryBase + 0x200));
  assert(coprocessor->WriteCsr(xopennpux::kCsrQuantScalesAddress,
                               kMemoryBase + 0x300));
  assert(coprocessor->WriteCsr(
      xopennpux::kCsrQuantGIdxAddress,
      has_g_idx ? kMemoryBase + 0x380 : 0));
  assert(coprocessor->WriteCsr(
      xopennpux::kCsrQuantConfig,
      xopennpux::EncodeQuantConfig(4, 1, xopennpux::DataType::kFp32,
                                   has_g_idx)));
  assert(coprocessor->WriteCsr(xopennpux::kCsrQuantQweightStride, 20));
  assert(coprocessor->WriteCsr(xopennpux::kCsrQuantQzerosStride, 8));
  assert(coprocessor->WriteCsr(xopennpux::kCsrQuantScalesStride, 20));
  assert(coprocessor->WriteCsr(xopennpux::kCsrQuantGroupRange, 2u << 16));
}

void TestInt4DequantStridedTailTile() {
  constexpr uint32_t kColumns = 3;
  constexpr uint32_t kRows = 8;
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureInt4Dequant(&coprocessor, kColumns, kRows, true);

  Gem5TmmaDispatchPacket packet = Packet(19);
  packet.instruction = xopennpux::EncodeTdequant(12, 10);
  packet.rs1_value = kMemoryBase + 0x100;
  packet.rs2_value = 0;
  packet.rd_value = kMemoryBase + 0x400;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  for (uint32_t column = 0; column < kColumns; ++column) {
    uint32_t packed = 0;
    for (uint32_t row = 0; row < kRows; ++row) {
      packed |= ((row + column + 1) & 0xf) << (row * 4);
    }
    WriteUint32(&memory, 0x100 + column * 4, packed);
  }
  WriteUint32(&memory, 0x200, 0x00000321);
  WriteUint32(&memory, 0x208, 0x00000654);
  const float scales[] = {0.5f, 1.0f, 1.5f, 2.0f, 0.25f, 0.75f};
  for (uint32_t group = 0; group < 2; ++group) {
    for (uint32_t column = 0; column < kColumns; ++column) {
      WriteFloat(&memory, 0x300 + group * 20 + column * 4,
                 scales[group * kColumns + column]);
    }
  }
  const uint32_t g_idx[] = {0, 0, 1, 1, 0, 0, 1, 1};
  for (uint32_t row = 0; row < kRows; ++row) {
    WriteUint32(&memory, 0x380 + row * 4, g_idx[row]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTdequant);
  assert(completion.element_operations == kRows * kColumns);
  assert(completion.modeled_cycles == kRows * kColumns);
  assert(completion.destination_bytes == kRows * kColumns * sizeof(float));
  for (uint32_t row = 0; row < kRows; ++row) {
    const uint32_t group = g_idx[row];
    for (uint32_t column = 0; column < kColumns; ++column) {
      const int32_t quantized = (row + column + 1) & 0xf;
      const int32_t zero = static_cast<int32_t>(
          (group == 0 ? column + 1 : column + 4) + 1);
      const float expected =
          static_cast<float>(quantized - zero) *
          scales[group * kColumns + column];
      const float actual = ReadFloat(
          memory, 0x400 + (row * kColumns + column) * sizeof(float));
      assert(actual == expected);
    }
  }
}

void TestInt4DequantGlobalGroupBase() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureInt4Dequant(&coprocessor, 1, 4, false);
  assert(coprocessor.WriteCsr(xopennpux::kCsrQuantGroupRange,
                              (2u << 16) | 1u));

  Gem5TmmaDispatchPacket packet = Packet(20);
  packet.instruction = xopennpux::EncodeTdequant(12, 10);
  packet.rs1_value = kMemoryBase + 0x100;
  packet.rs2_value = 0;
  packet.rd_value = kMemoryBase + 0x400;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  WriteUint32(&memory, 0x100, 0x9999u);
  WriteUint32(&memory, 0x200, 0u);
  WriteUint32(&memory, 0x208, 2u);
  WriteFloat(&memory, 0x300, 1.0f);
  WriteFloat(&memory, 0x314, 2.0f);
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  for (size_t index = 0; index < 4; ++index) {
    assert(ReadFloat(memory, 0x400 + index * sizeof(float)) == 12.0f);
  }
}

void TestInt4DequantRejectsInvalidStride() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureInt4Dequant(&coprocessor, 3, 8, false);
  assert(coprocessor.WriteCsr(xopennpux::kCsrQuantQweightStride, 8));
  Gem5TmmaDispatchPacket packet = Packet(20);
  packet.instruction = xopennpux::EncodeTdequant(12, 10);
  assert(coprocessor.Submit(packet) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);
}

void TestInt4DequantThenFp32Matmul() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureInt4Dequant(&coprocessor, 2, 8, false);
  Gem5TmmaDispatchPacket dequant = Packet(21);
  dequant.instruction = xopennpux::EncodeTdequant(12, 10);
  dequant.rs1_value = kMemoryBase + 0x100;
  dequant.rs2_value = 0;
  dequant.rd_value = kMemoryBase + 0x400;
  assert(coprocessor.Submit(dequant) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  WriteUint32(&memory, 0x100, UINT32_C(0x99999999));
  WriteUint32(&memory, 0x104, UINT32_C(0xaaaaaaaa));
  WriteUint32(&memory, 0x200, UINT32_C(0x00000077));
  WriteUint32(&memory, 0x208, UINT32_C(0x00000077));
  for (uint32_t group = 0; group < 2; ++group) {
    WriteFloat(&memory, 0x300 + group * 20, 0.5f);
    WriteFloat(&memory, 0x304 + group * 20, 1.0f);
  }
  Gem5TmmaCompletion dequant_completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase,
                                 &dequant_completion));
  assert(dequant_completion.error == Gem5TmmaExecutionError::kNone);

  ConfigureFp32(&coprocessor, 2, 2, 8);
  Gem5TmmaDispatchPacket mma = Packet(22);
  mma.rs1_value = kMemoryBase + 0x500;
  mma.rs2_value = kMemoryBase + 0x400;
  mma.rd_value = kMemoryBase + 0x600;
  for (uint32_t inner = 0; inner < 8; ++inner) {
    WriteFloat(&memory, 0x500 + inner * sizeof(float),
               static_cast<float>(inner + 1));
    WriteFloat(&memory, 0x500 + (8 + inner) * sizeof(float), 1.0f);
  }
  assert(coprocessor.Submit(mma) == Gem5TmmaSubmitResult::kAccepted);
  Gem5TmmaCompletion mma_completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &mma_completion));
  assert(mma_completion.error == Gem5TmmaExecutionError::kNone);
  assert(mma_completion.mac_operations == 32);
  assert(ReadFloat(memory, 0x600) == 18.0f);
  assert(ReadFloat(memory, 0x604) == 72.0f);
  assert(ReadFloat(memory, 0x608) == 4.0f);
  assert(ReadFloat(memory, 0x60c) == 16.0f);
}

void TestFp32TensorAdd() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 4);
  Gem5TmmaDispatchPacket packet = Packet(8);
  packet.instruction = xopennpux::EncodeTadd(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  for (size_t index = 0; index < 8; ++index) {
    WriteFloat(&memory, index * sizeof(float),
               static_cast<float>(index + 1));
    WriteFloat(&memory, 0x100 + index * sizeof(float),
               static_cast<float>(8 - index));
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTadd);
  assert(completion.mac_operations == 0);
  assert(completion.element_operations == 8);
  assert(completion.modeled_cycles == 8);
  assert(completion.destination_bytes == 8 * sizeof(float));
  for (size_t index = 0; index < 8; ++index) {
    assert(ReadFloat(memory, 0x200 + index * sizeof(float)) == 9.0f);
  }
}

void TestFp32TensorMul() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 6);
  Gem5TmmaDispatchPacket packet = Packet(9);
  packet.instruction = xopennpux::EncodeTmul(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  for (size_t index = 0; index < 6; ++index) {
    WriteFloat(&memory, index * sizeof(float),
               static_cast<float>(index + 1));
    WriteFloat(&memory, 0x100 + index * sizeof(float), 2.0f);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTmul);
  assert(completion.element_operations == 6);
  assert(completion.modeled_cycles == 6);
  for (size_t index = 0; index < 6; ++index) {
    assert(ReadFloat(memory, 0x200 + index * sizeof(float)) ==
           static_cast<float>((index + 1) * 2));
  }
}

void TestFp32Sigmoid() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 3);
  Gem5TmmaDispatchPacket packet = Packet(10);
  packet.instruction = xopennpux::EncodeTsigmoid(12, 10);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  WriteFloat(&memory, 0, 0.0f);
  WriteFloat(&memory, 4, 1.0f);
  WriteFloat(&memory, 8, -1.0f);
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTsigmoid);
  assert(completion.element_operations == 9);
  assert(std::abs(ReadFloat(memory, 0x200) - 0.5f) < 1.0e-6f);
  assert(std::abs(ReadFloat(memory, 0x204) - 0.7310586f) < 1.0e-6f);
  assert(std::abs(ReadFloat(memory, 0x208) - 0.2689414f) < 1.0e-6f);
}

void TestFp32RowScale() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 3);
  Gem5TmmaDispatchPacket packet = Packet(11);
  packet.instruction = xopennpux::EncodeTrowScale(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  for (size_t index = 0; index < 6; ++index) {
    WriteFloat(&memory, index * sizeof(float),
               static_cast<float>(index + 1));
  }
  WriteFloat(&memory, 0x100, 2.0f);
  WriteFloat(&memory, 0x104, -1.0f);
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTrowScale);
  assert(completion.element_operations == 6);
  assert(ReadFloat(memory, 0x200) == 2.0f);
  assert(ReadFloat(memory, 0x208) == 6.0f);
  assert(ReadFloat(memory, 0x20c) == -4.0f);
  assert(ReadFloat(memory, 0x214) == -6.0f);
}

void TestFp32Dma() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 3);
  Gem5TmmaDispatchPacket packet = Packet(10);
  packet.instruction = xopennpux::EncodeTdma(12, 10);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  for (size_t index = 0; index < 6; ++index) {
    WriteFloat(&memory, index * sizeof(float),
               static_cast<float>(index + 1));
  }
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTdma);
  assert(completion.element_operations == 6);
  assert(completion.modeled_cycles == 6);
  for (size_t index = 0; index < 6; ++index) {
    assert(ReadFloat(memory, 0x200 + index * sizeof(float)) ==
           static_cast<float>(index + 1));
  }
}

void TestFp32RmsNorm() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 4);
  float epsilon = 1.0e-6f;
  uint32_t epsilon_bits = 0;
  std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon_bits));
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, epsilon_bits));

  Gem5TmmaDispatchPacket packet = Packet(10);
  packet.instruction = xopennpux::EncodeTrmsnorm(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                         -1.0f, 1.0f, -1.0f, 1.0f};
  const float weight[] = {1.0f, 0.5f, 2.0f, 1.0f};
  for (size_t index = 0; index < 8; ++index) {
    WriteFloat(&memory, index * sizeof(float), input[index]);
  }
  for (size_t index = 0; index < 4; ++index) {
    WriteFloat(&memory, 0x100 + index * sizeof(float), weight[index]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTrmsnorm);
  assert(completion.element_operations == 32);
  assert(completion.modeled_cycles == 32);
  const float inverse_rms0 = 1.0f / std::sqrt(7.5f + epsilon);
  for (size_t index = 0; index < 4; ++index) {
    const float expected = input[index] * inverse_rms0 * weight[index];
    assert(std::fabs(ReadFloat(memory, 0x200 + index * sizeof(float)) -
                     expected) < 1.0e-6f);
  }
  const float inverse_rms1 = 1.0f / std::sqrt(1.0f + epsilon);
  for (size_t index = 0; index < 4; ++index) {
    const float expected = input[index + 4] * inverse_rms1 * weight[index];
    assert(std::fabs(ReadFloat(memory, 0x210 + index * sizeof(float)) -
                     expected) < 1.0e-6f);
  }
}

void TestRmsNormFlags() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 2);
  float epsilon = 1.0e-6f;
  uint32_t epsilon_bits = 0;
  std::memcpy(&epsilon_bits, &epsilon, sizeof(epsilon_bits));
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, epsilon_bits));
  assert(coprocessor.WriteCsr(
      xopennpux::kCsrTensorFlags,
      xopennpux::kTensorFlagNormWeightOffset |
          xopennpux::kTensorFlagBfloat16Input));

  Gem5TmmaDispatchPacket packet = Packet(20);
  packet.instruction = xopennpux::EncodeTrmsnorm(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  WriteFloat(&memory, 0, 1.001f);
  WriteFloat(&memory, sizeof(float), 2.001f);
  WriteFloat(&memory, 0x100, 0.0f);
  WriteFloat(&memory, 0x100 + sizeof(float), 0.0f);
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  const float inverse_rms = 1.0f / std::sqrt(2.5f + epsilon);
  assert(std::fabs(ReadFloat(memory, 0x200) - inverse_rms) < 1.0e-6f);
  assert(std::fabs(ReadFloat(memory, 0x204) - 2.0f * inverse_rms) < 1.0e-6f);
}

void TestRmsNormRejectsInvalidEpsilon() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 4);
  Gem5TmmaDispatchPacket packet = Packet(11);
  packet.instruction = xopennpux::EncodeTrmsnorm(12, 10, 11);
  assert(coprocessor.Submit(packet) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);
}

void TestFp32Silu() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 4);
  Gem5TmmaDispatchPacket packet = Packet(12);
  packet.instruction = xopennpux::EncodeTsilu(12, 10);
  packet.rs2_value = 0;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float input[] = {-2.0f, -1.0f, 0.0f, 1.0f,
                         2.0f, 4.0f, 8.0f, 16.0f};
  for (size_t index = 0; index < 8; ++index) {
    WriteFloat(&memory, index * sizeof(float), input[index]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTsilu);
  assert(completion.element_operations == 24);
  assert(completion.modeled_cycles == 24);
  assert(completion.destination_bytes == 8 * sizeof(float));
  for (size_t index = 0; index < 8; ++index) {
    const float expected = input[index] / (1.0f + std::exp(-input[index]));
    assert(std::fabs(ReadFloat(memory, 0x200 + index * sizeof(float)) -
                     expected) < 1.0e-6f);
  }
}

void TestFp32Softmax() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 4);
  Gem5TmmaDispatchPacket packet = Packet(13);
  packet.instruction = xopennpux::EncodeTsoftmax(12, 10);
  packet.rs2_value = 0;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                         -4.0f, -2.0f, 0.0f, 2.0f};
  for (size_t index = 0; index < 8; ++index) {
    WriteFloat(&memory, index * sizeof(float), input[index]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTsoftmax);
  assert(completion.element_operations == 32);
  assert(completion.modeled_cycles == 32);
  for (size_t row = 0; row < 2; ++row) {
    float expected_sum = 0.0f;
    for (size_t feature = 0; feature < 4; ++feature) {
      expected_sum += std::exp(input[row * 4 + feature] -
                               input[row * 4 + 3]);
    }
    float actual_sum = 0.0f;
    for (size_t feature = 0; feature < 4; ++feature) {
      const float expected =
          std::exp(input[row * 4 + feature] - input[row * 4 + 3]) /
          expected_sum;
      const float actual =
          ReadFloat(memory, 0x200 + (row * 4 + feature) * sizeof(float));
      assert(std::fabs(actual - expected) < 1.0e-6f);
      actual_sum += actual;
    }
    assert(std::fabs(actual_sum - 1.0f) < 1.0e-6f);
  }
}

void TestFp32Gather() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 3);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 4));
  Gem5TmmaDispatchPacket packet = Packet(14);
  packet.instruction = xopennpux::EncodeTgather(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  for (size_t index = 0; index < 12; ++index) {
    WriteFloat(&memory, index * sizeof(float),
               static_cast<float>(index + 1));
  }
  const uint32_t indices[] = {2, 0};
  std::memcpy(memory.data() + 0x100, indices, sizeof(indices));

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTgather);
  assert(completion.element_operations == 6);
  assert(completion.modeled_cycles == 6);
  const float expected[] = {7, 8, 9, 1, 2, 3};
  for (size_t index = 0; index < 6; ++index) {
    assert(ReadFloat(memory, 0x200 + index * sizeof(float)) ==
           expected[index]);
  }
}

void TestFp32RopeHalfSplit() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 4);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 1));
  Gem5TmmaDispatchPacket packet = Packet(15);
  packet.instruction = xopennpux::EncodeTrope(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float input[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float cosine[] = {1, 1, 1, 1, 0, 0, 0, 0};
  const float sine[] = {0, 0, 0, 0, 1, 1, 1, 1};
  for (size_t index = 0; index < 8; ++index) {
    WriteFloat(&memory, index * sizeof(float), input[index]);
    WriteFloat(&memory, 0x100 + index * sizeof(float), cosine[index]);
    WriteFloat(&memory, 0x120 + index * sizeof(float), sine[index]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTrope);
  assert(completion.element_operations == 24);
  assert(completion.modeled_cycles == 24);
  const float expected[] = {1, 2, 3, 4, -7, -8, 5, 6};
  for (size_t index = 0; index < 8; ++index) {
    assert(ReadFloat(memory, 0x200 + index * sizeof(float)) ==
           expected[index]);
  }
}

void TestFp32RopeAdjacent() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 4);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 0));
  Gem5TmmaDispatchPacket packet = Packet(17);
  packet.instruction = xopennpux::EncodeTrope(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float input[] = {1, 2, 3, 4};
  for (size_t index = 0; index < 4; ++index) {
    WriteFloat(&memory, index * sizeof(float), input[index]);
    WriteFloat(&memory, 0x100 + index * sizeof(float), 0.0f);
    WriteFloat(&memory, 0x110 + index * sizeof(float), 1.0f);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  const float expected[] = {-2, 1, -4, 3};
  for (size_t index = 0; index < 4; ++index) {
    assert(ReadFloat(memory, 0x200 + index * sizeof(float)) ==
           expected[index]);
  }
}

void TestRemainingOperatorValidation() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  Gem5TmmaDispatchPacket packet = Packet(18);

  ConfigureTensorFp32(&coprocessor, 1, 3);
  packet.instruction = xopennpux::EncodeTrope(12, 10, 11);
  assert(coprocessor.Submit(packet) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);

  ConfigureTensorFp32(&coprocessor, 1, 4);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 2));
  assert(coprocessor.Submit(packet) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);

  packet.instruction = xopennpux::EncodeTgather(12, 10, 11);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 0));
  assert(coprocessor.Submit(packet) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);

  packet.instruction = xopennpux::EncodeTtopk(12, 10);
  packet.rs2_value = 0;
  assert(coprocessor.Submit(packet) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 5));
  assert(coprocessor.Submit(packet) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);
}

void TestGatherIndexFault() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 1, 2);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 2));
  Gem5TmmaDispatchPacket packet = Packet(19);
  packet.instruction = xopennpux::EncodeTgather(12, 10, 11);
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const uint32_t invalid_index = 2;
  std::memcpy(memory.data() + 0x100, &invalid_index, sizeof(invalid_index));
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kAddress);
  assert(completion.faulting_address == kMemoryBase + 0x100);
}

void TestFp32TopK() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 5);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 2));
  Gem5TmmaDispatchPacket packet = Packet(16);
  packet.instruction = xopennpux::EncodeTtopk(12, 10);
  packet.rs2_value = 0;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float input[] = {1, 5, 3, 5, 2, -1, 0, 4, 2, 3};
  for (size_t index = 0; index < 10; ++index) {
    WriteFloat(&memory, index * sizeof(float), input[index]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.operation == xopennpux::Operation::kTtopk);
  assert(completion.element_operations == 20);
  assert(completion.modeled_cycles == 20);
  assert(completion.destination_bytes == 8 * sizeof(uint32_t));
  const float expected_values[] = {5, 5, 4, 3};
  const uint32_t expected_indices[] = {1, 3, 2, 4};
  for (size_t index = 0; index < 4; ++index) {
    assert(ReadFloat(memory, 0x200 + index * sizeof(float)) ==
           expected_values[index]);
    assert(ReadUint32(memory, 0x210 + index * sizeof(uint32_t)) ==
           expected_indices[index]);
  }
}

void TestFp32TopKSplitOutput() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureTensorFp32(&coprocessor, 2, 5);
  assert(coprocessor.WriteCsr(xopennpux::kCsrScalarParam0, 2));
  assert(coprocessor.WriteCsr(xopennpux::kCsrTensorAuxDestinationAddress,
                              kMemoryBase + 0x300));
  Gem5TmmaDispatchPacket packet = Packet(17);
  packet.instruction = xopennpux::EncodeTtopk(12, 10);
  packet.rs2_value = 0;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float input[] = {1, 5, 3, 5, 2, -1, 0, 4, 2, 3};
  for (size_t index = 0; index < 10; ++index) {
    WriteFloat(&memory, index * sizeof(float), input[index]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.destination_bytes == 4 * sizeof(float));
  const float expected_values[] = {5, 5, 4, 3};
  const uint32_t expected_indices[] = {1, 3, 2, 4};
  for (size_t index = 0; index < 4; ++index) {
    assert(ReadFloat(memory, 0x200 + index * sizeof(float)) ==
           expected_values[index]);
    assert(ReadUint32(memory, 0x300 + index * sizeof(uint32_t)) ==
           expected_indices[index]);
  }
}

void TestFp32MatmulAndSnapshot() {
  Gem5TmmaCoprocessor coprocessor;
  ConfigureFp32(&coprocessor, 2, 2, 3);
  const uint32_t accepted_epoch = coprocessor.csr_epoch();
  assert(coprocessor.Submit(Packet(7)) == Gem5TmmaSubmitResult::kAccepted);

  // A younger CSR write must not mutate the accepted command.
  assert(coprocessor.WriteCsr(xopennpux::kCsrMmaShape,
                              xopennpux::EncodeMmaShape(1, 1, 1)));

  std::vector<uint8_t> memory(4096, 0);
  const float lhs[] = {1, 2, 3, 4, 5, 6};
  const float rhs[] = {7, 8, 9, 10, 11, 12};
  for (size_t i = 0; i < 6; ++i) {
    WriteFloat(&memory, i * sizeof(float), lhs[i]);
    WriteFloat(&memory, 0x100 + i * sizeof(float), rhs[i]);
  }

  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(completion.sequence_id == 7);
  assert(completion.pc == 0x100);
  assert(completion.instruction == xopennpux::EncodeTmma(12, 10, 11));
  assert(completion.csr_epoch == accepted_epoch);
  assert(completion.mac_operations == 12);
  assert(ReadFloat(memory, 0x200) == 58.0f);
  assert(ReadFloat(memory, 0x204) == 64.0f);
  assert(ReadFloat(memory, 0x208) == 139.0f);
  assert(ReadFloat(memory, 0x20c) == 154.0f);
  assert(completion.destination_address == kMemoryBase + 0x200);
  assert(completion.destination_bytes == 4 * sizeof(float));
  assert(completion.destination_checksum ==
         Fnv1a(memory, 0x200, completion.destination_bytes));
  assert(completion.destination_words[0] == UINT32_C(0x42680000));
  assert(completion.destination_words[1] == UINT32_C(0x42800000));
  assert(completion.destination_words[2] == UINT32_C(0x430b0000));
  assert(completion.destination_words[3] == UINT32_C(0x431a0000));
}

void TestRejectAndBackpressure() {
  Gem5TmmaCoprocessor coprocessor;
  assert(coprocessor.Submit(Packet(0)) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);
  ConfigureFp32(&coprocessor, 1, 1, 1);

  Gem5TmmaDispatchPacket invalid = Packet(0);
  invalid.instruction ^= 1u << 25;
  assert(coprocessor.Submit(invalid) ==
         Gem5TmmaSubmitResult::kIllegalInstruction);

  assert(coprocessor.WriteCsr(xopennpux::kCsrMmaShape,
                              xopennpux::EncodeMmaShape(1, 1, 1) |
                                  0x80000000u));
  assert(coprocessor.Submit(Packet(0)) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);
  ConfigureFp32(&coprocessor, 1, 1, 1);
  for (size_t i = 0; i < Gem5TmmaCoprocessor::kQueueCapacity; ++i) {
    assert(coprocessor.Submit(Packet(static_cast<uint32_t>(i))) ==
           Gem5TmmaSubmitResult::kAccepted);
  }
  assert(!coprocessor.ready());
  assert(coprocessor.Submit(Packet(99)) ==
         Gem5TmmaSubmitResult::kBackpressure);
}

void TestPacketCsrSnapshotAndFence() {
  Gem5TmmaCoprocessor coprocessor;
  Gem5TmmaDispatchPacket packet = Packet(42);
  packet.mma_shape = xopennpux::EncodeMmaShape(1, 1, 1);
  packet.mma_data_type = xopennpux::EncodeMmaDataTypes(
      xopennpux::DataType::kFp32, xopennpux::DataType::kFp32,
      xopennpux::DataType::kFp32);
  packet.csr_epoch = 9;
  assert(coprocessor.Submit(packet) == Gem5TmmaSubmitResult::kAccepted);

  Gem5TmmaDispatchPacket fence;
  fence.instruction = xopennpux::EncodeTfence();
  assert(coprocessor.Classify(fence) ==
         Gem5TmmaSubmitResult::kBackpressure);

  std::vector<uint8_t> memory(4096, 0);
  WriteFloat(&memory, 0, 2.0f);
  WriteFloat(&memory, 0x100, 3.0f);
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.csr_epoch == 9);
  assert(ReadFloat(memory, 0x200) == 6.0f);
  assert(coprocessor.Submit(fence) == Gem5TmmaSubmitResult::kAccepted);
}

void TestInvalidTypeAndAddressFault() {
  Gem5TmmaCoprocessor coprocessor;
  assert(coprocessor.WriteCsr(xopennpux::kCsrMmaShape,
                              xopennpux::EncodeMmaShape(1, 1, 1)));
  assert(coprocessor.WriteCsr(
      xopennpux::kCsrMmaDataType,
      xopennpux::EncodeMmaDataTypes(xopennpux::DataType::kInt8,
                                    xopennpux::DataType::kInt8,
                                    xopennpux::DataType::kInt8)));
  assert(coprocessor.Submit(Packet(1)) ==
         Gem5TmmaSubmitResult::kInvalidCsrState);

  ConfigureFp32(&coprocessor, 1, 1, 1);
  std::vector<uint8_t> memory(4096, 0);
  Gem5TmmaCompletion completion;
  Gem5TmmaDispatchPacket bad_address = Packet(2);
  bad_address.rd_value = kMemoryBase - 4;
  assert(coprocessor.Submit(bad_address) == Gem5TmmaSubmitResult::kAccepted);
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kAddress);
  assert(completion.faulting_address == kMemoryBase - 4);
}

void TestStridedTransposedAccumulatingTmma() {
  Gem5TmmaCoprocessor coprocessor;
  ConfigureFp32(&coprocessor, 2, 2, 2);
  assert(coprocessor.WriteCsr(xopennpux::kCsrMmaLhsStride, 12));
  assert(coprocessor.WriteCsr(xopennpux::kCsrMmaRhsStride, 12));
  assert(coprocessor.WriteCsr(xopennpux::kCsrMmaDstStride, 12));
  assert(coprocessor.WriteCsr(xopennpux::kCsrMmaFlags,
                              xopennpux::kMmaFlagTransposeRhs));
  assert(coprocessor.Submit(Packet(50)) ==
         Gem5TmmaSubmitResult::kAccepted);

  std::vector<uint8_t> memory(4096, 0);
  const float lhs[] = {1, 2, 0, 3, 4};
  const float rhs[] = {5, 6, 0, 7, 8};
  for (size_t index = 0; index < 5; ++index) {
    WriteFloat(&memory, index * 4, lhs[index]);
    WriteFloat(&memory, 0x100 + index * 4, rhs[index]);
  }
  Gem5TmmaCompletion completion;
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(ReadFloat(memory, 0x200) == 17.0f);
  assert(ReadFloat(memory, 0x204) == 23.0f);
  assert(ReadFloat(memory, 0x20c) == 39.0f);
  assert(ReadFloat(memory, 0x210) == 53.0f);

  ConfigureFp32(&coprocessor, 2, 2, 1);
  assert(coprocessor.WriteCsr(xopennpux::kCsrMmaFlags,
                              xopennpux::kMmaFlagTransposeRhs |
                                  xopennpux::kMmaFlagAccumulate));
  Gem5TmmaDispatchPacket accumulate = Packet(51);
  accumulate.rs1_value += 8;
  accumulate.rs2_value += 8;
  WriteFloat(&memory, 0x008, 2.0f);
  WriteFloat(&memory, 0x014, 3.0f);
  WriteFloat(&memory, 0x108, 4.0f);
  WriteFloat(&memory, 0x114, 5.0f);
  assert(coprocessor.Submit(accumulate) ==
         Gem5TmmaSubmitResult::kAccepted);
  assert(coprocessor.ExecuteNext(&memory, kMemoryBase, &completion));
  assert(completion.error == Gem5TmmaExecutionError::kNone);
  assert(ReadFloat(memory, 0x200) == 25.0f);
  assert(ReadFloat(memory, 0x204) == 33.0f);
  assert(ReadFloat(memory, 0x20c) == 51.0f);
  assert(ReadFloat(memory, 0x210) == 68.0f);
}

}  // namespace

int main() {
  TestEncodingAndSecondLevelDecode();
  TestInt4DequantStridedTailTile();
  TestInt4DequantGlobalGroupBase();
  TestInt4DequantRejectsInvalidStride();
  TestInt4DequantThenFp32Matmul();
  TestFp32TensorAdd();
  TestFp32TensorMul();
  TestFp32Sigmoid();
  TestFp32RowScale();
  TestFp32Dma();
  TestStatefulCausalDepthwiseConv();
  TestCausalGqaAttention();
  TestGatedAttention();
  TestGatedRecurrentUpdate();
  TestFp32GroupedConv2d();
  TestFp32RmsNorm();
  TestRmsNormFlags();
  TestRmsNormRejectsInvalidEpsilon();
  TestFp32Silu();
  TestFp32Softmax();
  TestFp32Gather();
  TestFp32RopeHalfSplit();
  TestFp32RopeAdjacent();
  TestRemainingOperatorValidation();
  TestGatherIndexFault();
  TestFp32TopK();
  TestFp32TopKSplitOutput();
  TestFp32MatmulAndSnapshot();
  TestRejectAndBackpressure();
  TestPacketCsrSnapshotAndFence();
  TestInvalidTypeAndAddressFault();
  TestStridedTransposedAccumulatingTmma();
  return 0;
}
