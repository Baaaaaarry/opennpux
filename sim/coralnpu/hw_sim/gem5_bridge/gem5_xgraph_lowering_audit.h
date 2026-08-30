#ifndef HW_SIM_GEM5_BRIDGE_GEM5_XGRAPH_LOWERING_AUDIT_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_XGRAPH_LOWERING_AUDIT_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"

struct Gem5XGraphLoweringAuditFailure {
  uint32_t opcode = 0;
  int error_code = 0;
  uint32_t count = 0;
  uint32_t first_command = UINT32_MAX;
};

struct Gem5XGraphLoweringAuditStats {
  uint32_t observed_requests = 0;
  uint32_t lowerable_requests = 0;
  uint32_t host_fused_requests = 0;
  uint64_t emitted_commands = 0;
  std::array<Gem5XGraphLoweringAuditFailure, 32> failures = {};
  uint32_t failure_count = 0;
};

// Audits the exact materialized requests used by Host numerical execution
// against the generic XGraph lowering contract. It never changes execution.
class Gem5XGraphLoweringAudit final
    : public Gem5HostFunctionalRequestObserver {
 public:
  void Observe(Gem5HostFunctionalExecutionPath path,
               const opennpux_npu_functional_request& request,
               const Gem5FunctionalMemoryRegion* regions,
               size_t region_count) override;

  const Gem5XGraphLoweringAuditStats& stats() const { return stats_; }
  void Print(FILE* stream) const;

 private:
  void RecordFailure(uint32_t opcode, int error_code, uint32_t command_id);

  Gem5XGraphLoweringAuditStats stats_ = {};
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_XGRAPH_LOWERING_AUDIT_H_
