#ifndef HW_SIM_GEM5_BRIDGE_GEM5_CUSTOM_MAC_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_CUSTOM_MAC_H_

#include <cstdint>

#include "VOpenNpuXCustomMac.h"
#include "verilated.h"

class Gem5CustomMac {
 public:
  static constexpr uint32_t kBase = 0x30000000;
  static constexpr uint32_t kSize = 0x100;

  explicit Gem5CustomMac(VerilatedContext* context)
      : context_(context), model_(context, "custom_mac"), active_cycles_(0) {
    model_.clk_i = 0;
    model_.rst_ni = 1;
    model_.write_en_i = 0;
    model_.addr_i = 0;
    model_.write_data_i = 0;
    Reset();
  }

  void Reset() {
    model_.rst_ni = 0;
    Step();
    model_.rst_ni = 1;
    Step();
    active_cycles_ = 0;
  }

  void Step() {
    context_->timeInc(1);
    model_.clk_i = 1;
    model_.eval();
    context_->timeInc(1);
    model_.clk_i = 0;
    model_.eval();
  }

  void StepIfActive() {
    if (active_cycles_ == 0) {
      return;
    }
    Step();
    --active_cycles_;
  }

  bool Contains(uint32_t addr, uint32_t size) const {
    return addr >= kBase && size != 0 && addr - kBase < kSize &&
           size <= kSize - (addr - kBase);
  }

  void Write32(uint32_t addr, uint32_t value) {
    model_.addr_i = addr - kBase;
    model_.write_data_i = value;
    model_.write_en_i = 1;
    Step();
    model_.write_en_i = 0;
    model_.eval();
    if (addr == kBase + 0x0c && (value & 1) != 0) {
      active_cycles_ = 3;
    }
  }

  uint32_t Read32(uint32_t addr) {
    model_.addr_i = addr - kBase;
    model_.eval();
    return model_.read_data_o;
  }

 private:
  VerilatedContext* const context_;
  VOpenNpuXCustomMac model_;
  uint32_t active_cycles_;
};

#endif
