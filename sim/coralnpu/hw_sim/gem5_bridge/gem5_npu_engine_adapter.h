#ifndef HW_SIM_GEM5_BRIDGE_GEM5_NPU_ENGINE_ADAPTER_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_NPU_ENGINE_ADAPTER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hw_sim/gem5_bridge/gem5_npu_control_plane.h"
#include "hw_sim/gem5_bridge/gem5_tmma_coprocessor.h"

struct Gem5NpuEngineCompletion {
  Gem5NpuEngine engine = Gem5NpuEngine::kControl;
  Gem5TmmaCompletion execution{};
};

// Stable boundary between the control-plane scheduler and functional or RTL
// execution engines. Implementations may complete requests out of order.
class Gem5NpuEngineAdapter {
 public:
  virtual ~Gem5NpuEngineAdapter() = default;
  virtual void Reset() = 0;
  virtual bool CanAccept(Gem5NpuEngine engine) const = 0;
  virtual Gem5TmmaSubmitResult Submit(
      Gem5NpuEngine engine, const Gem5TmmaDispatchPacket& packet) = 0;
  virtual bool Poll(std::vector<uint8_t>* memory, uint32_t memory_base,
                    Gem5NpuEngineCompletion* completion) = 0;
  virtual size_t pending_count() const = 0;
};

class Gem5NpuFunctionalEngineAdapter final : public Gem5NpuEngineAdapter {
 public:
  void Reset() override;
  bool CanAccept(Gem5NpuEngine engine) const override;
  Gem5TmmaSubmitResult Submit(
      Gem5NpuEngine engine, const Gem5TmmaDispatchPacket& packet) override;
  bool Poll(std::vector<uint8_t>* memory, uint32_t memory_base,
            Gem5NpuEngineCompletion* completion) override;
 size_t pending_count() const override { return pending_count_; }

 private:
  struct Pending {
    Gem5NpuEngine engine = Gem5NpuEngine::kControl;
    Gem5TmmaDispatchPacket packet{};
    uint64_t ready_cycle = 0;
    bool valid = false;
  };

  static uint64_t EngineLatency(Gem5NpuEngine engine);

  std::array<Pending, Gem5XOpenNpuFunctionalCoprocessor::kQueueCapacity>
      pending_{};
  uint64_t current_cycle_ = 0;
  size_t pending_count_ = 0;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_NPU_ENGINE_ADAPTER_H_
