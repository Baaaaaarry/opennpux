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
  assert(scheduler.stats().epoch_stalls == 1);

  assert(scheduler.Finish(Completion(first)));
  assert(scheduler.Retire(nullptr));
  assert(scheduler.Finish(Completion(second)));
  assert(scheduler.Retire(nullptr));
  assert(scheduler.Issue(&blocked) && blocked.ordering_epoch == 1);
}

void TestEngineCreditBackpressure() {
  Gem5NpuTaskScheduler scheduler;
  scheduler.Reset();
  assert(scheduler.Submit(Task(30, 0, 0, Gem5NpuEngine::kTensor)));
  assert(scheduler.Submit(Task(31, 1, 0, Gem5NpuEngine::kTensor)));
  Gem5NpuTask issued = {};
  assert(scheduler.Issue(&issued));
  assert(!scheduler.Issue(&issued));
  assert(scheduler.stats().engine_credit_stalls == 1);
}

void TestCompletionQueueBackpressure() {
  Gem5NpuTaskScheduler scheduler;
  scheduler.Reset();
  scheduler.SetEngineCredits(Gem5NpuEngine::kVector, 32);
  for (uint32_t index = 0; index < 17; ++index) {
    assert(scheduler.Submit(
        Task(100 + index, index, 0, Gem5NpuEngine::kVector)));
  }
  for (uint32_t index = 0; index < 16; ++index) {
    Gem5NpuTask issued = {};
    assert(scheduler.Issue(&issued));
    assert(scheduler.Finish(Completion(issued)));
  }
  assert(scheduler.completion_count() == Gem5NpuCompletionQueue::kCapacity);
  Gem5NpuTask blocked = {};
  assert(!scheduler.Issue(&blocked));
  assert(scheduler.stats().completion_backpressure_stalls == 1);

  Gem5NpuCompletion retired = {};
  assert(scheduler.Retire(&retired));
  assert(scheduler.Issue(&blocked));
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

void TestIndependentEnginesIssueConcurrently() {
  Gem5NpuTaskScheduler scheduler;
  scheduler.Reset();
  const Gem5NpuTask tdma = Task(20, 0, 0, Gem5NpuEngine::kTdma);
  const Gem5NpuTask tensor = Task(21, 1, 0, Gem5NpuEngine::kTensor);
  const Gem5NpuTask dependent =
      Task(22, 2, UINT64_C(1) << 1, Gem5NpuEngine::kVector);
  assert(scheduler.Submit(tdma));
  assert(scheduler.Submit(tensor));
  assert(scheduler.Submit(dependent));

  Gem5NpuTask first = {};
  Gem5NpuTask second = {};
  assert(scheduler.Issue(&first));
  assert(scheduler.Issue(&second));
  assert(first.command_id == 0 && second.command_id == 1);
  assert(scheduler.inflight_count() == 2);
  Gem5NpuTask blocked = {};
  assert(!scheduler.Issue(&blocked));

  assert(scheduler.Finish(Completion(second)));
  assert(!scheduler.Retire(nullptr));
  assert(scheduler.Finish(Completion(first)));
  assert(scheduler.Retire(nullptr));
  assert(scheduler.Retire(nullptr));
  assert(scheduler.Issue(&blocked) && blocked.command_id == 2);
  assert(scheduler.stats().tasks_submitted == 3);
  assert(scheduler.stats().tasks_issued == 3);
  assert(scheduler.stats().tasks_retired == 2);
  assert(scheduler.stats().dependency_stalls == 1);
  assert(scheduler.stats().max_inflight == 2);
  assert(scheduler.stats().max_completion_queue == 2);
}

}  // namespace

int main() {
  TestDependencyAndRetirement();
  TestEngineCreditsAndEpochFence();
  TestEngineCreditBackpressure();
  TestCompletionQueueBackpressure();
  TestCompletionErrorsDoNotSatisfyDependencies();
  TestOutOfOrderCompletionRetiresInOrder();
  TestIndependentEnginesIssueConcurrently();
  return 0;
}
