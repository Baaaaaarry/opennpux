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
  const uint32_t gather = xopennpux::EncodeTgather(12, 10, 11);
  assert(xopennpux::IsTgather(gather));
  assert(xopennpux::DecodeOperation(gather) ==
         xopennpux::Operation::kTgather);
  const uint32_t topk = xopennpux::EncodeTtopk(12, 10);
  assert(xopennpux::IsTtopk(topk));
  assert(xopennpux::DecodeOperation(topk) ==
         xopennpux::Operation::kTtopk);
  assert(!xopennpux::IsTtopk(topk | (1u << 20)));
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

}  // namespace

int main() {
  TestEncodingAndSecondLevelDecode();
  TestFp32TensorAdd();
  TestFp32TensorMul();
  TestFp32RmsNorm();
  TestRmsNormRejectsInvalidEpsilon();
  TestFp32Silu();
  TestFp32Softmax();
  TestFp32Gather();
  TestFp32RopeHalfSplit();
  TestFp32RopeAdjacent();
  TestRemainingOperatorValidation();
  TestGatherIndexFault();
  TestFp32TopK();
  TestFp32MatmulAndSnapshot();
  TestRejectAndBackpressure();
  TestPacketCsrSnapshotAndFence();
  TestInvalidTypeAndAddressFault();
  return 0;
}
