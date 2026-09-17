#ifndef HW_SIM_GEM5_BRIDGE_GEM5_NPU_CONTROL_PLANE_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_NPU_CONTROL_PLANE_H_

#include <array>
#include <cstddef>
#include <cstdint>

enum class Gem5NpuEngine : uint8_t {
  kTdma = 0,
  kTensor = 1,
  kVector = 2,
  kSfu = 3,
  kControl = 4,
};

enum class Gem5NpuCompletionStatus : uint8_t {
  kSuccess = 0,
  kExecutionError = 1,
  kMemoryFault = 2,
  kInvalidCommand = 3,
};

struct Gem5NpuTask {
  uint32_t sequence = 0;
  uint32_t submission_tag = 0;
  uint32_t command_id = 0;
  uint32_t ordering_epoch = 0;
  uint64_t dependency_mask = 0;
  Gem5NpuEngine engine = Gem5NpuEngine::kControl;
};

struct Gem5NpuCompletion {
  uint32_t sequence = 0;
  uint32_t submission_tag = 0;
  uint32_t command_id = 0;
  uint32_t ordering_epoch = 0;
  Gem5NpuEngine engine = Gem5NpuEngine::kControl;
  Gem5NpuCompletionStatus status = Gem5NpuCompletionStatus::kSuccess;
  uint32_t fault_address = 0;
  uint64_t operations = 0;
  uint64_t cycles = 0;
};

// Tracks retirement, not execution. A dependency becomes visible only after
// its completion record is committed in program order.
class Gem5NpuDependencyScoreboard {
 public:
  static constexpr size_t kWindowSize = 64;

  void Reset();
  bool Ready(uint64_t dependency_mask) const;
  bool Complete(uint32_t command_id);
  bool IsComplete(uint32_t command_id) const;
  uint64_t completed_mask() const { return completed_mask_; }

 private:
  uint64_t completed_mask_ = 0;
};

class Gem5NpuCompletionQueue {
 public:
  static constexpr size_t kCapacity = 16;

  void Reset();
  bool Push(const Gem5NpuCompletion& completion);
  bool Peek(Gem5NpuCompletion* completion) const;
  bool Pop(Gem5NpuCompletion* completion);
  bool Remove(uint32_t sequence, Gem5NpuCompletion* completion);
  bool empty() const { return size_ == 0; }
  bool full() const { return size_ == kCapacity; }
  size_t size() const { return size_; }

 private:
  struct Entry {
    Gem5NpuCompletion completion{};
    bool valid = false;
  };
  std::array<Entry, kCapacity> entries_{};
  size_t size_ = 0;
};

class Gem5NpuTaskScheduler {
 public:
  static constexpr size_t kCapacity = 64;
  static constexpr size_t kEngineCount = 5;

  void Reset();
  bool Submit(const Gem5NpuTask& task);
  bool Issue(Gem5NpuTask* task);
  bool Finish(const Gem5NpuCompletion& completion);
  bool Retire(Gem5NpuCompletion* completion);

  void SetEngineCredits(Gem5NpuEngine engine, uint8_t credits);
  size_t pending_count() const { return pending_count_; }
  size_t inflight_count() const { return inflight_count_; }
  size_t completion_count() const { return completions_.size(); }
  const Gem5NpuDependencyScoreboard& scoreboard() const { return scoreboard_; }

 private:
  enum class State : uint8_t { kEmpty, kPending, kIssued, kFinished };
  struct Entry {
    Gem5NpuTask task{};
    State state = State::kEmpty;
  };

  static size_t EngineIndex(Gem5NpuEngine engine);
  bool OlderEpochPending(const Gem5NpuTask& task) const;

  std::array<Entry, kCapacity> entries_{};
  std::array<uint8_t, kEngineCount> engine_credits_{};
  std::array<uint8_t, kEngineCount> engine_inflight_{};
  Gem5NpuDependencyScoreboard scoreboard_;
  Gem5NpuCompletionQueue completions_;
  size_t pending_count_ = 0;
  size_t inflight_count_ = 0;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_NPU_CONTROL_PLANE_H_
