#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HOST_FUNCTIONAL_GRAPH_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HOST_FUNCTIONAL_GRAPH_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "opennpux/npu_functional_materializer.h"
#include "hw_sim/gem5_bridge/gem5_generic_command_dispatch.h"
#include "hw_sim/gem5_bridge/gem5_host_tensor_arena.h"

class Gem5HostWeightProvider;
struct Gem5HostWeightBinding;

enum class Gem5HostFunctionalExecutionPath {
  kGenericRequest,
  kHostFusedRoutedExpert,
};

// Synchronously observes the exact address-based request presented to the
// numerical backend. The observer is non-owning and must not retain pointers
// to requests or regions after Observe() returns.
class Gem5HostFunctionalRequestObserver {
 public:
  virtual ~Gem5HostFunctionalRequestObserver() = default;
  virtual void Observe(
      Gem5HostFunctionalExecutionPath path,
      const opennpux_npu_functional_request& request,
      const Gem5FunctionalMemoryRegion* regions,
      size_t region_count) = 0;
};

struct Gem5HostFunctionalGraphStats {
  uint32_t completed_commands;
  uint64_t operations;
  uint64_t modeled_cycles;
  uint64_t bytes_read;
  uint64_t bytes_written;
};

void ApplyGem5HostBfloat16OutputBoundaries(
    const opennpux_npu_functional_request& request,
    const std::vector<Gem5FunctionalMemoryRegion>& regions);

// Joins one instantiated invocation with its SSA tensor plan. Weight operands
// remain caller supplied because they may come from a page cache or a direct
// host mapping without changing graph scheduling.
class Gem5HostFunctionalGraph {
 public:
  bool LoadTensorPlan(const std::string& tensor_plan_path);
  bool Configure(const void* submission, size_t submission_size,
                 uint32_t submission_base,
                 uint32_t arena_base = UINT32_C(0x30000000));
  bool ConfigureRuntime(
      const void* submission, size_t submission_size, uint32_t submission_base,
      const opennpux_npu_tensor_plan_runtime& runtime,
      uint32_t arena_base = UINT32_C(0x30000000));
  bool ConfigureRuntimePreservingPersistent(
      const void* submission, size_t submission_size, uint32_t submission_base,
      const opennpux_npu_tensor_plan_runtime& runtime,
      uint32_t arena_base = UINT32_C(0x30000000));
  bool Materialize(
      uint32_t command_index,
      const opennpux_npu_functional_operand* extra_operands,
      uint32_t extra_operand_count,
      opennpux_npu_functional_request* request) const;
  bool Execute(
      opennpux_npu_functional_request* request,
      const Gem5FunctionalMemoryRegion* extra_regions = nullptr,
      size_t extra_region_count = 0);
  bool ExecuteGptqProjection(uint32_t command_index,
                             Gem5HostWeightProvider* weights,
                             uint32_t role_id, uint64_t expert_id,
                             uint32_t slot_id);
  bool ExecuteGptqQkv(uint32_t command_index,
                      Gem5HostWeightProvider* weights);
  bool ExecuteGptqRouter(uint32_t command_index,
                         Gem5HostWeightProvider* weights);
  bool ExecuteRoutedExpert(uint32_t command_index,
                           Gem5HostWeightProvider* weights);
  bool ExecuteGptqExpert(uint32_t command_index,
                         Gem5HostWeightProvider* weights,
                         const Gem5HostWeightBinding& binding);
  bool ExecuteCommand(uint32_t command_index,
                      Gem5HostWeightProvider* weights);
  bool ComputeLinearAttentionGateProjection(
      uint32_t command_index, Gem5HostWeightProvider* weights,
      std::vector<float>* output) const;
  bool ExecuteProgram(Gem5HostWeightProvider* weights,
                      uint32_t* failed_command = nullptr);
  bool SetInputTokenIds(const uint32_t* token_ids, size_t token_count);
  bool ReadNextToken(uint32_t* token_id) const;
  void ResetInvocation();
  void SetRequestObserver(Gem5HostFunctionalRequestObserver* observer) {
    observer_ = observer;
  }

  uint32_t command_count() const {
    return configured_ ? program_.header->command_count : 0;
  }
  const opennpux_npu_command* command(uint32_t index) const;
  Gem5HostTensorArena& arena() { return arena_; }
  const Gem5HostTensorArena& arena() const { return arena_; }
  const Gem5HostFunctionalGraphStats& stats() const { return stats_; }

 private:
  bool ExecuteFloatWeight(uint32_t command_index,
                          Gem5HostWeightProvider* weights,
                          const Gem5HostWeightBinding& binding);
  bool ExecuteLinearAttentionProjection(
      uint32_t command_index, Gem5HostWeightProvider* weights,
      const std::vector<Gem5HostWeightBinding>& bindings);
  bool ExecuteLinearAttentionGateNorm(
      uint32_t command_index, Gem5HostWeightProvider* weights,
      const std::vector<Gem5HostWeightBinding>& bindings);
  bool ExecuteLinearAttentionRecurrent(
      uint32_t command_index, Gem5HostWeightProvider* weights,
      const std::vector<Gem5HostWeightBinding>& bindings);
  bool ExecuteFloatSharedExpert(
      uint32_t command_index, Gem5HostWeightProvider* weights,
      const std::vector<Gem5HostWeightBinding>& bindings);
  bool ExecuteFloatQkv(
      uint32_t command_index, Gem5HostWeightProvider* weights,
      const std::vector<Gem5HostWeightBinding>& bindings);
  bool ExecutePositioned(uint32_t command_index);
  bool ConfigureRuntimeInternal(
      const void* submission, size_t submission_size, uint32_t submission_base,
      const opennpux_npu_tensor_plan_runtime& runtime, uint32_t arena_base,
      bool preserve_persistent);
  std::vector<uint8_t> submission_;
  uint32_t submission_base_ = 0;
  Gem5HostTensorArena arena_;
  opennpux_npu_functional_program program_ = {};
  Gem5HostFunctionalGraphStats stats_ = {};
  Gem5HostFunctionalRequestObserver* observer_ = nullptr;
  bool configured_ = false;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HOST_FUNCTIONAL_GRAPH_H_
