#include "hw_sim/gem5_bridge/gem5_npu_engine_adapter.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

int main() {
  Gem5NpuFunctionalEngineAdapter adapter;
  adapter.Reset();
  assert(adapter.CanAccept(Gem5NpuEngine::kTensor));

  Gem5TmmaDispatchPacket packet = {};
  packet.instruction = xopennpux::EncodeTadd(10, 11, 12);
  packet.sequence_id = 7;
  packet.csr_epoch = 1;
  packet.rs1_value = UINT32_C(0x20000000);
  packet.rs2_value = UINT32_C(0x20000010);
  packet.rd_value = UINT32_C(0x20000020);
  packet.tensor_shape = xopennpux::EncodeTensorShape(1, 4);
  packet.tensor_features = 4;
  packet.tensor_data_type = xopennpux::EncodeMmaDataTypes(
      xopennpux::DataType::kFp32, xopennpux::DataType::kFp32,
      xopennpux::DataType::kFp32);

  std::vector<uint8_t> memory(48);
  const float lhs[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float rhs[4] = {4.0f, 3.0f, 2.0f, 1.0f};
  std::memcpy(memory.data(), lhs, sizeof(lhs));
  std::memcpy(memory.data() + 16, rhs, sizeof(rhs));
  assert(adapter.Submit(Gem5NpuEngine::kVector, packet) ==
         Gem5TmmaSubmitResult::kAccepted);
  assert(adapter.pending_count() == 1);

  Gem5NpuEngineCompletion completion = {};
  assert(adapter.Poll(&memory, UINT32_C(0x20000000), &completion));
  assert(completion.engine == Gem5NpuEngine::kVector);
  assert(completion.execution.sequence_id == 7);
  assert(completion.execution.error == Gem5TmmaExecutionError::kNone);
  const float* output = reinterpret_cast<const float*>(memory.data() + 32);
  for (size_t index = 0; index < 4; ++index) assert(output[index] == 5.0f);
  assert(adapter.pending_count() == 0);

  adapter.Reset();
  Gem5TmmaDispatchPacket slow = packet;
  slow.sequence_id = 8;
  slow.rd_value = UINT32_C(0x20000020);
  Gem5TmmaDispatchPacket fast = packet;
  fast.sequence_id = 9;
  fast.rd_value = UINT32_C(0x20000020);
  assert(adapter.Submit(Gem5NpuEngine::kTensor, slow) ==
         Gem5TmmaSubmitResult::kAccepted);
  assert(adapter.Submit(Gem5NpuEngine::kTdma, fast) ==
         Gem5TmmaSubmitResult::kAccepted);
  assert(adapter.Poll(&memory, UINT32_C(0x20000000), &completion));
  assert(completion.engine == Gem5NpuEngine::kTdma);
  assert(completion.execution.sequence_id == 9);
  assert(adapter.Poll(&memory, UINT32_C(0x20000000), &completion));
  assert(completion.engine == Gem5NpuEngine::kTensor);
  assert(completion.execution.sequence_id == 8);
  return 0;
}
