#include "hw_sim/gem5_bridge/gem5_npu_engine_adapter.h"

void Gem5NpuFunctionalEngineAdapter::Reset() {
  coprocessor_.Reset();
  engines_.fill(Gem5NpuEngine::kControl);
  head_ = 0;
  pending_count_ = 0;
}

bool Gem5NpuFunctionalEngineAdapter::CanAccept(Gem5NpuEngine engine) const {
  return static_cast<size_t>(engine) < Gem5NpuTaskScheduler::kEngineCount &&
         pending_count_ < engines_.size() && coprocessor_.ready();
}

Gem5TmmaSubmitResult Gem5NpuFunctionalEngineAdapter::Submit(
    Gem5NpuEngine engine, const Gem5TmmaDispatchPacket& packet) {
  if (!CanAccept(engine)) return Gem5TmmaSubmitResult::kBackpressure;
  const Gem5TmmaSubmitResult result = coprocessor_.Submit(packet);
  if (result != Gem5TmmaSubmitResult::kAccepted) return result;
  engines_[(head_ + pending_count_) % engines_.size()] = engine;
  ++pending_count_;
  return result;
}

bool Gem5NpuFunctionalEngineAdapter::Poll(
    std::vector<uint8_t>* memory, uint32_t memory_base,
    Gem5NpuEngineCompletion* completion) {
  if (completion == nullptr || pending_count_ == 0) return false;
  completion->engine = engines_[head_];
  if (!coprocessor_.ExecuteNext(memory, memory_base,
                                &completion->execution)) {
    return false;
  }
  engines_[head_] = Gem5NpuEngine::kControl;
  head_ = (head_ + 1) % engines_.size();
  --pending_count_;
  return true;
}
