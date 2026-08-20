#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"
#include "hw_sim/gem5_bridge/gem5_generic_gptq_executor.h"
#include "opennpux/npu_weight_ranges.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

int main(int argc, char** argv) {
  assert(argc == 3);
  Gem5HostWeightProvider provider;
  assert(provider.Load(argv[1], argv[2], 4096));

  uint32_t q_command = UINT32_MAX;
  uint32_t embedding_command = UINT32_MAX;
  opennpux_npu_weight_ranges ranges = {};
  assert(opennpux_npu_weight_ranges_load(argv[2], &ranges) == 0);
  for (uint32_t index = 0; index < ranges.header->range_count; ++index) {
    if (ranges.records[index].role_id ==
        OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ) {
      q_command = ranges.records[index].command_id;
    }
    if (ranges.records[index].role_id ==
            OPENNPUX_NPU_WEIGHT_ROLE_EMBEDDING &&
        ranges.records[index].component_id ==
            OPENNPUX_NPU_WEIGHT_COMPONENT_WEIGHT) {
      embedding_command = ranges.records[index].command_id;
    }
  }
  assert(q_command != UINT32_MAX);
  Gem5GenericGptqWeights projection = {};
  assert(provider.LoadProjection(
      q_command, OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ,
      OPENNPUX_NPU_WEIGHT_EXPERT_NONE, OPENNPUX_NPU_WEIGHT_SLOT_Q_PROJ,
      &projection));
  assert(projection.qweight.data != nullptr && projection.qweight.size == 288);
  assert(projection.scales.data != nullptr && projection.scales.size == 48);
  assert(embedding_command != UINT32_MAX);
  std::vector<float> embedding;
  assert(provider.LoadFloatWeight(
      embedding_command, OPENNPUX_NPU_WEIGHT_ROLE_EMBEDDING,
      OPENNPUX_NPU_WEIGHT_EXPERT_NONE, OPENNPUX_NPU_WEIGHT_SLOT_DEFAULT,
      &embedding));
  assert(embedding.size() == 8);
  assert(embedding[0] == 0.0f && embedding[7] == 7.0f);
  for (float value : embedding) {
    assert(std::isfinite(value));
  }
  opennpux_npu_weight_ranges_unload(&ranges);
  provider.Reset();
  assert(!provider.loaded());
  return 0;
}
