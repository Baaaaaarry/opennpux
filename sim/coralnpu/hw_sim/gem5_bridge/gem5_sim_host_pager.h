#ifndef HW_SIM_GEM5_BRIDGE_GEM5_SIM_HOST_PAGER_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_SIM_HOST_PAGER_H_

#include <cstdio>
#include <cstdint>
#include <vector>

#include "hw_sim/gem5_bridge/npu_page_bundle.h"

class Gem5SimHostPager {
 public:
  Gem5SimHostPager();
  ~Gem5SimHostPager();

  Gem5SimHostPager(const Gem5SimHostPager&) = delete;
  Gem5SimHostPager& operator=(const Gem5SimHostPager&) = delete;

  void Reset();
  int Service(std::vector<uint8_t>* extmem);
  bool enabled() const { return configured_; }
  uint64_t serviced() const { return serviced_; }

 private:
  bool RangeValid(uint64_t address, uint64_t size,
                  const std::vector<uint8_t>& extmem) const;
  int Fail(const char* reason);

  FILE* bundle_ = nullptr;
  bool configured_ = false;
  opennpux_npu_page_bundle_header header_ = {};
  uint64_t serviced_ = 0;
  uint64_t consumed_ = 0;
  uint64_t transferred_ = 0;
  bool failed_ = false;
};

#endif
