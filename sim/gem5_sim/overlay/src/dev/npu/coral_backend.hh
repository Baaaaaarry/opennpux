/*
 * Backend interface for Coral-compatible NPU models in gem5.
 *
 * The NPUDevice SimObject owns the SoC-facing MMIO aperture and DMA port.
 * Concrete backends implement the execution model behind that aperture, from
 * the current stage-A transaction-level model to a future Verilated RTL
 * bridge based on the official google-coral/coralnpu framework.
 */

#ifndef __DEV_NPU_CORAL_BACKEND_HH__
#define __DEV_NPU_CORAL_BACKEND_HH__

#include <string>

#include "base/types.hh"
#include "mem/packet.hh"
#include "mem/translation_gen.hh"

namespace gem5
{

class CoralBackend
{
  public:
    virtual ~CoralBackend() = default;

    virtual const char *name() const = 0;
    virtual bool read(PacketPtr pkt, Addr pioAddr) = 0;
    virtual bool write(PacketPtr pkt, Addr pioAddr) = 0;
    virtual TranslationGenPtr translate(Addr vaddr, Addr size) = 0;

    virtual bool hasPendingEvent() const = 0;
    virtual Tick nextEventTick() const = 0;
    virtual void processEvent() = 0;
};

} // namespace gem5

#endif // __DEV_NPU_CORAL_BACKEND_HH__
