#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HOST_WEIGHT_PROVIDER_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HOST_WEIGHT_PROVIDER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct Gem5GenericGptqWeights;

// Owns direct host mappings of semantic GPTQ components. A private
// implementation prevents the runtime's full submission ABI from leaking
// into bridge headers that use Coral's reduced standalone ABI copy.
class Gem5HostWeightProvider {
 public:
  Gem5HostWeightProvider();
  ~Gem5HostWeightProvider();
  Gem5HostWeightProvider(const Gem5HostWeightProvider&) = delete;
  Gem5HostWeightProvider& operator=(const Gem5HostWeightProvider&) = delete;

  bool Load(const std::string& manifest_path,
            const std::string& weight_ranges_path,
            size_t max_component_size = SIZE_MAX);
  void Reset();

  bool LoadProjection(uint32_t command_id, uint32_t role_id,
                      uint64_t expert_id, uint32_t slot_id,
                      Gem5GenericGptqWeights* weights);
  bool LoadFloatWeight(uint32_t command_id, uint32_t role_id,
                       uint64_t expert_id, uint32_t slot_id,
                       std::vector<float>* weights);
  bool ConfigureRoutedExpert(uint32_t command_id);
  static bool ProvideRoutedExpert(void* opaque, uint64_t expert_id,
                                  Gem5GenericGptqWeights* gate,
                                  Gem5GenericGptqWeights* up,
                                  Gem5GenericGptqWeights* down);
  bool loaded() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HOST_WEIGHT_PROVIDER_H_
