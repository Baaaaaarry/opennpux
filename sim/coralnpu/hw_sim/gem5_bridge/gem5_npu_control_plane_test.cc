#include "hw_sim/gem5_bridge/gem5_npu_control_plane.h"

#include <cassert>

namespace {

Gem5NpuTask Task(uint32_t sequence, uint32_t command_id,
                 uint64_t dependencies, Gem5NpuEngine engine,
                 uint32_t epoch = 0) {
  Gem5NpuTask task = {};
  task.sequence = sequence;
  task.command_id = command_id;
  task.dependency_mask = dependencies;
  task.engine = engine;
  task.ordering_epoch = epoch;
  return task;
}

Gem5NpuCompletion Completion(const Gem5NpuTask& task) {
  Gem5NpuCompletion completion = {};
  completion.sequence = task.sequence;
  completion.command_id = task.command_id;
  completion.engine = task.engine;
  completion.ordering_epoch = task.ordering_epoch;
  return completion;
}

void TestDependencyAndRetirement() {
  Gem5NpuTaskScheduler scheduler;
  scheduler.Reset();
  assert(scheduler.Submit(Task(10, 0, 0, Gem5NpuEngine::kTdma)));
  assert(scheduler.Submit(
      Task(11, 1, UINT64_C(1), Gem5NpuEngine::kTensor)));

  Gem5NpuTask issued = {};
  assert(scheduler.Issue(&issued) && issued.command_id == 0);
  assert(!scheduler.Issue(&issued));
  assert(scheduler.Finish(Completion(issued)));
  assert(!scheduler.scoreboard().IsComplete(0));

  Gem5NpuCompletion retired = {};
  assert(scheduler.Retire(&retired));
  assert(scheduler.scoreboard().IsComplete(0));
  assert(scheduler.Issue(&issued) && issued.command_id == 1);
}

void TestEngineCreditsAndEpochFence() {
  Gem5NpuTaskScheduler scheduler;
  scheduler.Reset();
  scheduler.SetEngineCredits(Gem5NpuEngine::kTensor, 2);
  assert(scheduler.Submit(Task(1, 0, 0, Gem5NpuEngine::kTensor, 0)));
  assert(scheduler.Submit(Task(2, 1, 0, Gem5NpuEngine::kTensor, 0)));
  assert(scheduler.Submit(Task(3, 2, 0, Gem5NpuEngine::kVector, 1)));

  Gem5NpuTask first = {};
  Gem5NpuTask second = {};
  assert(scheduler.Issue(&first));
  assert(scheduler.Issue(&second));
  assert(scheduler.inflight_count() == 2);
  Gem5NpuTask blocked = {};
  assert(!scheduler.Issue(&blocked));

  assert(scheduler.Finish(Completion(first)));
  assert(scheduler.Retire(nullptr));
  assert(scheduler.Finish(Completion(second)));
  assert(scheduler.Retire(nullptr));
  assert(scheduler.Issue(&blocked) && blocked.ordering_epoch == 1);
}

void TestCompletionErrorsDoNotSatisfyDependencies() {
  Gem5NpuTaskScheduler scheduler;
  scheduler.Reset();
  const Gem5NpuTask producer = Task(1, 0, 0, Gem5NpuEngine::kTdma);
  assert(scheduler.Submit(producer));
  Gem5NpuTask issued = {};
  assert(scheduler.Issue(&issued));
  Gem5NpuCompletion failed = Completion(issued);
  failed.status = Gem5NpuCompletionStatus::kMemoryFault;
  failed.fault_address = 0x20001000;
  assert(scheduler.Finish(failed));
  Gem5NpuCompletion retired = {};
  assert(scheduler.Retire(&retired));
  assert(retired.status == Gem5NpuCompletionStatus::kMemoryFault);
  assert(!scheduler.scoreboard().IsComplete(0));
}

void TestOutOfOrderCompletionRetiresInOrder() {
  Gem5NpuTaskScheduler scheduler;
  scheduler.Reset();
  scheduler.SetEngineCredits(Gem5NpuEngine::kTensor, 2);
  const Gem5NpuTask first = Task(10, 0, 0, Gem5NpuEngine::kTensor);
  const Gem5NpuTask second = Task(11, 1, 0, Gem5NpuEngine::kTensor);
  assert(scheduler.Submit(first));
  assert(scheduler.Submit(second));
  Gem5NpuTask issued = {};
  assert(scheduler.Issue(&issued));
  assert(scheduler.Issue(&issued));
  assert(scheduler.Finish(Completion(second)));
  assert(!scheduler.Retire(nullptr));
  assert(scheduler.Finish(Completion(first)));
  Gem5NpuCompletion retired = {};
  assert(scheduler.Retire(&retired) && retired.sequence == first.sequence);
  assert(scheduler.Retire(&retired) && retired.sequence == second.sequence);
}

}  // namespace

int main() {
  TestDependencyAndRetirement();
  TestEngineCreditsAndEpochFence();
  TestCompletionErrorsDoNotSatisfyDependencies();
  TestOutOfOrderCompletionRetiresInOrder();
  return 0;
}
