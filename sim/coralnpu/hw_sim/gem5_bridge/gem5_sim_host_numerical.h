#ifndef HW_SIM_GEM5_BRIDGE_GEM5_SIM_HOST_NUMERICAL_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_SIM_HOST_NUMERICAL_H_

#include <cstdint>
#include <vector>

#include "hw_sim/gem5_bridge/npu_numerical_result.h"

class Gem5SimHostNumerical {
 public:
  Gem5SimHostNumerical();
  void Reset();
  int Publish(std::vector<uint8_t>* extmem);
  bool enabled() const { return configured_; }

 private:
  bool configured_ = false;
  bool valid_ = false;
  bool published_ = false;
  opennpux_npu_numerical_result result_ = {};
};

#endif
