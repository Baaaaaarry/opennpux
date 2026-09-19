#include "hw_sim/gem5_bridge/gem5_npu_engine_adapter.h"

#include <algorithm>

uint64_t Gem5NpuFunctionalEngineAdapter::EngineLatency(
    Gem5NpuEngine engine) {
  switch (engine) {
    case Gem5NpuEngine::kTdma:
      return 2;
    case Gem5NpuEngine::kVector:
      return 3;
    case Gem5NpuEngine::kSfu:
      return 5;
    case Gem5NpuEngine::kTensor:
      return 8;
    case Gem5NpuEngine::kControl:
      return 1;
  }
  return 1;
}

void Gem5NpuFunctionalEngineAdapter::Reset() {
  pending_.fill({});
  current_cycle_ = 0;
  pending_count_ = 0;
}

bool Gem5NpuFunctionalEngineAdapter::CanAccept(Gem5NpuEngine engine) const {
  return static_cast<size_t>(engine) < Gem5NpuTaskScheduler::kEngineCount &&
         pending_count_ < pending_.size();
}

Gem5TmmaSubmitResult Gem5NpuFunctionalEngineAdapter::Submit(
    Gem5NpuEngine engine, const Gem5TmmaDispatchPacket& packet) {
  if (!CanAccept(engine)) return Gem5TmmaSubmitResult::kBackpressure;
  auto entry = std::find_if(pending_.begin(), pending_.end(),
                            [](const Pending& item) { return !item.valid; });
  if (entry == pending_.end()) return Gem5TmmaSubmitResult::kBackpressure;
  entry->engine = engine;
  entry->packet = packet;
  entry->ready_cycle = current_cycle_ + EngineLatency(engine);
  entry->valid = true;
  ++pending_count_;
  return Gem5TmmaSubmitResult::kAccepted;
}

bool Gem5NpuFunctionalEngineAdapter::Poll(
    std::vector<uint8_t>* memory, uint32_t memory_base,
    Gem5NpuEngineCompletion* completion) {
  if (completion == nullptr || pending_count_ == 0) return false;
  auto entry = std::min_element(
      pending_.begin(), pending_.end(), [](const Pending& lhs,
                                          const Pending& rhs) {
        if (!lhs.valid) return false;
        if (!rhs.valid) return true;
        return lhs.ready_cycle < rhs.ready_cycle;
      });
  if (entry == pending_.end() || !entry->valid) return false;

  Gem5XOpenNpuFunctionalCoprocessor coprocessor;
  coprocessor.Reset();
  if (coprocessor.Submit(entry->packet) != Gem5TmmaSubmitResult::kAccepted ||
      !coprocessor.ExecuteNext(memory, memory_base,
                              &completion->execution)) {
    return false;
  }
  current_cycle_ = std::max(current_cycle_, entry->ready_cycle);
  completion->engine = entry->engine;
  *entry = {};
  --pending_count_;
  return true;
}
