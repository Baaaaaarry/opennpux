#include "hw_sim/gem5_bridge/gem5_npu_control_plane.h"

#include <algorithm>

void Gem5NpuDependencyScoreboard::Reset() { completed_mask_ = 0; }

bool Gem5NpuDependencyScoreboard::Ready(uint64_t dependency_mask) const {
  return (dependency_mask & ~completed_mask_) == 0;
}

bool Gem5NpuDependencyScoreboard::Complete(uint32_t command_id) {
  if (command_id >= kWindowSize) return false;
  completed_mask_ |= UINT64_C(1) << command_id;
  return true;
}

bool Gem5NpuDependencyScoreboard::IsComplete(uint32_t command_id) const {
  return command_id < kWindowSize &&
         (completed_mask_ & (UINT64_C(1) << command_id)) != 0;
}

void Gem5NpuCompletionQueue::Reset() {
  entries_.fill({});
  size_ = 0;
}

bool Gem5NpuCompletionQueue::Push(const Gem5NpuCompletion& completion) {
  if (full()) return false;
  auto entry = std::find_if(entries_.begin(), entries_.end(),
                            [](const Entry& item) { return !item.valid; });
  if (entry == entries_.end()) return false;
  entry->completion = completion;
  entry->valid = true;
  ++size_;
  return true;
}

bool Gem5NpuCompletionQueue::Peek(Gem5NpuCompletion* completion) const {
  if (completion == nullptr || empty()) return false;
  const auto entry = std::find_if(entries_.begin(), entries_.end(),
                                  [](const Entry& item) {
                                    return item.valid;
                                  });
  if (entry == entries_.end()) return false;
  *completion = entry->completion;
  return true;
}

bool Gem5NpuCompletionQueue::Pop(Gem5NpuCompletion* completion) {
  if (completion == nullptr || empty()) return false;
  auto entry = std::find_if(entries_.begin(), entries_.end(),
                            [](const Entry& item) { return item.valid; });
  if (entry == entries_.end()) return false;
  *completion = entry->completion;
  *entry = {};
  --size_;
  return true;
}

bool Gem5NpuCompletionQueue::Remove(uint32_t sequence,
                                    Gem5NpuCompletion* completion) {
  if (completion == nullptr) return false;
  auto entry = std::find_if(entries_.begin(), entries_.end(),
                            [sequence](const Entry& item) {
                              return item.valid &&
                                     item.completion.sequence == sequence;
                            });
  if (entry == entries_.end()) return false;
  *completion = entry->completion;
  *entry = {};
  --size_;
  return true;
}

size_t Gem5NpuTaskScheduler::EngineIndex(Gem5NpuEngine engine) {
  return static_cast<size_t>(engine);
}

void Gem5NpuTaskScheduler::Reset() {
  entries_.fill({});
  engine_credits_.fill(1);
  engine_inflight_.fill(0);
  scoreboard_.Reset();
  completions_.Reset();
  pending_count_ = 0;
  inflight_count_ = 0;
  stats_ = {};
}

void Gem5NpuTaskScheduler::SetEngineCredits(Gem5NpuEngine engine,
                                            uint8_t credits) {
  engine_credits_[EngineIndex(engine)] = credits;
}

bool Gem5NpuTaskScheduler::Submit(const Gem5NpuTask& task) {
  if (task.command_id >= Gem5NpuDependencyScoreboard::kWindowSize ||
      EngineIndex(task.engine) >= kEngineCount || pending_count_ == kCapacity) {
    return false;
  }
  auto entry = std::find_if(entries_.begin(), entries_.end(),
                            [](const Entry& item) {
                              return item.state == State::kEmpty;
                            });
  if (entry == entries_.end()) return false;
  entry->task = task;
  entry->state = State::kPending;
  ++pending_count_;
  ++stats_.tasks_submitted;
  return true;
}

bool Gem5NpuTaskScheduler::OlderEpochPending(const Gem5NpuTask& task) const {
  for (const Entry& entry : entries_) {
    if (entry.state != State::kEmpty &&
        entry.task.ordering_epoch < task.ordering_epoch) {
      return true;
    }
  }
  return false;
}

bool Gem5NpuTaskScheduler::Issue(Gem5NpuTask* task) {
  if (task == nullptr) return false;
  if (completions_.full()) {
    ++stats_.completion_backpressure_stalls;
    return false;
  }
  bool dependency_blocked = false;
  bool epoch_blocked = false;
  bool credit_blocked = false;
  for (Entry& entry : entries_) {
    const size_t engine = EngineIndex(entry.task.engine);
    if (entry.state != State::kPending) {
      continue;
    }
    if (!scoreboard_.Ready(entry.task.dependency_mask)) {
      dependency_blocked = true;
      continue;
    }
    if (OlderEpochPending(entry.task)) {
      epoch_blocked = true;
      continue;
    }
    if (engine_inflight_[engine] >= engine_credits_[engine]) {
      credit_blocked = true;
      continue;
    }
    entry.state = State::kIssued;
    ++engine_inflight_[engine];
    ++inflight_count_;
    --pending_count_;
    ++stats_.tasks_issued;
    stats_.max_inflight = std::max(stats_.max_inflight, inflight_count_);
    *task = entry.task;
    return true;
  }
  stats_.dependency_stalls += dependency_blocked ? 1 : 0;
  stats_.epoch_stalls += epoch_blocked ? 1 : 0;
  stats_.engine_credit_stalls += credit_blocked ? 1 : 0;
  return false;
}

bool Gem5NpuTaskScheduler::Finish(const Gem5NpuCompletion& completion) {
  for (Entry& entry : entries_) {
    if (entry.state != State::kIssued ||
        entry.task.sequence != completion.sequence) {
      continue;
    }
    if (!completions_.Push(completion)) return false;
    entry.state = State::kFinished;
    const size_t engine = EngineIndex(entry.task.engine);
    --engine_inflight_[engine];
    --inflight_count_;
    stats_.max_completion_queue =
        std::max(stats_.max_completion_queue, completions_.size());
    return true;
  }
  return false;
}

bool Gem5NpuTaskScheduler::Retire(Gem5NpuCompletion* completion) {
  const Entry* oldest = nullptr;
  for (const Entry& entry : entries_) {
    if (entry.state == State::kEmpty ||
        (oldest != nullptr &&
         (entry.task.ordering_epoch > oldest->task.ordering_epoch ||
          (entry.task.ordering_epoch == oldest->task.ordering_epoch &&
           entry.task.sequence >= oldest->task.sequence)))) {
      continue;
    }
    oldest = &entry;
  }
  if (oldest == nullptr || oldest->state != State::kFinished) return false;
  Gem5NpuCompletion front = {};
  if (!completions_.Remove(oldest->task.sequence, &front)) return false;
  for (Entry& entry : entries_) {
    if (entry.state == State::kFinished &&
        entry.task.sequence == front.sequence) {
      if (front.status == Gem5NpuCompletionStatus::kSuccess) {
        scoreboard_.Complete(entry.task.command_id);
      }
      entry = {};
      ++stats_.tasks_retired;
      if (completion != nullptr) *completion = front;
      return true;
    }
  }
  return false;
}
