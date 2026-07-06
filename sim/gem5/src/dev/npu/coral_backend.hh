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

#include <array>
#include <string>

#include "base/types.hh"
#include "dev/npu/coralnpu_gem5_abi.h"
#include "mem/packet.hh"
#include "mem/translation_gen.hh"

namespace gem5
{

enum class CoralDmaType
{
    Read,
    Write,
};

struct CoralDmaRequest
{
    CoralDmaType type;
    Addr addr;
    uint32_t size;
    std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES> data;
};

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

    virtual bool hasDmaRequest() const { return false; }
    virtual const CoralDmaRequest &dmaRequest() const;
    virtual void completeDma(const uint8_t *data, size_t size, bool error);
    virtual bool hasLocalExtmem() const { return false; }
    virtual void readLocalExtmem(Addr addr, void *data, size_t size);
    virtual void writeLocalExtmem(Addr addr, const void *data, size_t size);
};

} // namespace gem5

#endif // __DEV_NPU_CORAL_BACKEND_HH__
