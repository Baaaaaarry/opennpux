/*
 * Coral-compatible NPU device shell for gem5.
 *
 * This SimObject owns the SoC-facing MMIO aperture, DMA port, and board-level
 * integration. Execution behavior is delegated to a pluggable backend:
 * - stage-a: current transaction-level run-to-halt stub
 * - verilated-coral: future bridge to the official Coral RTL framework
 */

#ifndef __DEV_NPU_NPU_DEVICE_HH__
#define __DEV_NPU_NPU_DEVICE_HH__

#include <cstdint>
#include <array>
#include <memory>
#include <vector>

#include "dev/dma_virt_device.hh"
#include "dev/npu/coral_backend.hh"
#include "params/NPUDevice.hh"
#include "sim/eventq.hh"

namespace gem5
{

class NPUDevice : public DmaVirtDevice
{
  private:
    Addr pioAddr;
    Addr pioSize;
    Tick pioDelay;
    Addr dmaExtmemBase;
    Addr dmaSharedBase;
    Addr dmaSharedSize;
    bool fastDma;
    uint32_t backendId;
    uint32_t firmwareEntry;
    bool dmaActive;
    uint32_t dmaRequests;
    uint32_t dmaCompletions;
    uint32_t dmaErrors;
    std::vector<uint8_t> fastDmaCache;
    std::vector<bool> fastDmaPageValid;
    std::vector<bool> fastDmaPageDirty;

    bool dmaQuiesced() const;
    void checkDrainDone();
    void processBackendEvent();
    void syncBackendEvent();
    void startBackendDma();
    void completeBackendDma(
        const std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES> &data);
    void completeBackendDmaError();
    void fastDmaAccess(const CoralDmaRequest &request, Addr hostAddr,
                       std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES> &data);
    void flushFastDmaCache();
    void invalidateFastDmaCache();
    void functionalMemoryAccess(MemCmd command, Addr addr, size_t size,
                                uint8_t *data);

    std::unique_ptr<CoralBackend> backend;
    MemberEventWrapper<&NPUDevice::processBackendEvent> backendEvent;

  public:
    using Params = NPUDeviceParams;

    NPUDevice(const Params &p);

    DrainState drain() override;
    void serialize(CheckpointOut &cp) const override;
    void unserialize(CheckpointIn &cp) override;
    Tick read(PacketPtr pkt) override;
    Tick write(PacketPtr pkt) override;
    AddrRangeList getAddrRanges() const override;
    TranslationGenPtr translate(Addr vaddr, Addr size) override;
};

} // namespace gem5

#endif // __DEV_NPU_NPU_DEVICE_HH__
