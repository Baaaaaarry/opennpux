/*
 * Runtime-loaded bridge to an official Coral AXI RTL model.
 */

#ifndef __DEV_NPU_CORAL_VERILATED_BACKEND_HH__
#define __DEV_NPU_CORAL_VERILATED_BACKEND_HH__

#include <cstdint>
#include <string>

#include "dev/npu/coral_backend.hh"
#include "dev/npu/coralnpu_gem5_abi.h"

namespace gem5
{

class CoralVerilatedBackend : public CoralBackend
{
  private:
    using AbiVersionFn = uint32_t (*)();
    using CreateFn = coral_gem5_handle *(*)();
    using DestroyFn = void (*)(coral_gem5_handle *);
    using ResetFn = int (*)(coral_gem5_handle *);
    using MmioReadFn =
        int (*)(coral_gem5_handle *, uint32_t, void *, size_t);
    using MmioWriteFn =
        int (*)(coral_gem5_handle *, uint32_t, const void *, size_t);
    using StepFn = int (*)(coral_gem5_handle *, uint32_t);
    using DmaRequestGetFn =
        int (*)(coral_gem5_handle *, coral_gem5_dma_request *);
    using DmaCompleteFn =
        int (*)(coral_gem5_handle *, const void *, size_t, int);

    std::string coralRepo;
    std::string wrapperPath;
    std::string firmwarePath;
    Tick rtlTickPeriod;
    uint32_t rtlCyclesPerEvent;

    void *libraryHandle;
    coral_gem5_handle *modelHandle;
    DestroyFn destroyModel;
    MmioReadFn mmioRead;
    MmioWriteFn mmioWrite;
    StepFn stepModel;
    DmaRequestGetFn dmaRequestGet;
    DmaCompleteFn dmaComplete;

    bool running;
    Tick pendingEventTick;
    uint32_t resetControl;
    uint32_t firmwareEntry;
    bool dmaRequestPending;
    bool wfiObserved;
    uint64_t rtlEventCount;
    CoralDmaRequest pendingDmaRequest;

    template <class Function>
    Function loadSymbol(const char *name);
    void loadFirmware();

  public:
    CoralVerilatedBackend(const std::string &coral_repo,
                          const std::string &wrapper_path,
                          const std::string &firmware_path,
                          Tick rtl_tick_period,
                          uint32_t rtl_cycles_per_event);
    ~CoralVerilatedBackend() override;

    const char *name() const override { return "verilated-coral"; }
    bool read(PacketPtr pkt, Addr pio_addr) override;
    bool write(PacketPtr pkt, Addr pio_addr) override;
    TranslationGenPtr translate(Addr vaddr, Addr size) override;

    bool hasPendingEvent() const override;
    Tick nextEventTick() const override;
    void processEvent() override;
    uint32_t entryPoint() const { return firmwareEntry; }
    bool hasDmaRequest() const override { return dmaRequestPending; }
    const CoralDmaRequest &dmaRequest() const override;
    void completeDma(const uint8_t *data, size_t size, bool error) override;
};

} // namespace gem5

#endif // __DEV_NPU_CORAL_VERILATED_BACKEND_HH__
