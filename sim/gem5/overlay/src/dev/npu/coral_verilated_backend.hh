/*
 * Placeholder backend for future official Coral RTL integration.
 */

#ifndef __DEV_NPU_CORAL_VERILATED_BACKEND_HH__
#define __DEV_NPU_CORAL_VERILATED_BACKEND_HH__

#include <string>

#include "dev/npu/coral_backend.hh"

namespace gem5
{

class CoralVerilatedBackend : public CoralBackend
{
  private:
    std::string coralRepo;
    std::string wrapperPath;
    Tick rtlTickPeriod;

  public:
    CoralVerilatedBackend(const std::string &coral_repo,
                          const std::string &wrapper_path,
                          Tick rtl_tick_period);

    const char *name() const override { return "verilated-coral"; }
    bool read(PacketPtr pkt, Addr pio_addr) override;
    bool write(PacketPtr pkt, Addr pio_addr) override;
    TranslationGenPtr translate(Addr vaddr, Addr size) override;

    bool hasPendingEvent() const override { return false; }
    Tick nextEventTick() const override { return 0; }
    void processEvent() override;
};

} // namespace gem5

#endif // __DEV_NPU_CORAL_VERILATED_BACKEND_HH__
