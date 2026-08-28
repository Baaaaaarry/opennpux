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
}

void TestFp32TensorAdd() {
  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  ConfigureFp32(&coprocessor, 2, 2, 2);
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
  TestFp32MatmulAndSnapshot();
  TestRejectAndBackpressure();
  TestPacketCsrSnapshotAndFence();
  TestInvalidTypeAndAddressFault();
  return 0;
}
