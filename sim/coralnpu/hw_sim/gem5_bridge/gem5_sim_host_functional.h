#ifndef HW_SIM_GEM5_BRIDGE_GEM5_SIM_HOST_FUNCTIONAL_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_SIM_HOST_FUNCTIONAL_H_

#include <cstdint>
#include <memory>
#include <vector>

// Executes an instantiated generic NPU graph with host C++ kernels. The PIMPL
// keeps the full OpenNPUX runtime ABI out of the standalone Coral wrapper.
class Gem5SimHostFunctional {
 public:
  Gem5SimHostFunctional();
  ~Gem5SimHostFunctional();
  Gem5SimHostFunctional(const Gem5SimHostFunctional&) = delete;
  Gem5SimHostFunctional& operator=(const Gem5SimHostFunctional&) = delete;

  void Reset();
  int Service(std::vector<uint8_t>* extmem);
  bool enabled() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_SIM_HOST_FUNCTIONAL_H_
