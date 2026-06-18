/*
 * Placeholder backend for future official Coral RTL integration.
 */

#include "dev/npu/coral_verilated_backend.hh"

#include "base/logging.hh"

namespace gem5
{

CoralVerilatedBackend::CoralVerilatedBackend(const std::string &coral_repo,
                                             const std::string &wrapper_path,
                                             Tick rtl_tick_period)
  : coralRepo(coral_repo),
    wrapperPath(wrapper_path),
    rtlTickPeriod(rtl_tick_period)
{
    fatal(
        "NPU backend 'verilated-coral' is scaffolding only. "
        "Implement the official Coral CoreMiniAxi wrapper bridge first.\n"
        "coral_repo='%s' wrapper_path='%s' rtl_tick_period=%llu",
        coralRepo, wrapperPath, rtlTickPeriod);
}

bool
CoralVerilatedBackend::read(PacketPtr pkt, Addr pio_addr)
{
    panic("CoralVerilatedBackend::read called before bridge implementation");
}

bool
CoralVerilatedBackend::write(PacketPtr pkt, Addr pio_addr)
{
    panic("CoralVerilatedBackend::write called before bridge implementation");
}

TranslationGenPtr
CoralVerilatedBackend::translate(Addr vaddr, Addr size)
{
    panic(
        "CoralVerilatedBackend::translate called before bridge implementation");
}

void
CoralVerilatedBackend::processEvent()
{
    panic(
        "CoralVerilatedBackend::processEvent called before bridge implementation");
}

} // namespace gem5
