/*
 * Stage-A Coral-compatible transaction-level backend.
 */

#ifndef __DEV_NPU_CORAL_STAGEA_BACKEND_HH__
#define __DEV_NPU_CORAL_STAGEA_BACKEND_HH__

#include <vector>

#include "dev/npu/coral_backend.hh"

namespace gem5
{

class CoralStageABackend : public CoralBackend
{
  private:
    static constexpr Addr kDtcmOffset = 0x10000;
    static constexpr Addr kCsrOffset = 0x30000;
    static constexpr Addr kCsrSize = 0x1000;

    Addr pioAddr;
    Addr pioSize;
    Addr itcmSize;
    Addr dtcmSize;
    Tick executionLatency;
    bool autoHalt;

    uint32_t resetControl;
    uint32_t pcStart;
    bool halted;
    bool fault;

    Tick pendingEventTick;

    std::vector<uint8_t> itcm;
    std::vector<uint8_t> dtcm;

    bool inItcm(Addr offset, Addr size) const;
    bool inDtcm(Addr offset, Addr size) const;
    bool inCsr(Addr offset, Addr size) const;
    bool readMemoryWindow(PacketPtr pkt, std::vector<uint8_t> &window,
                          Addr windowBase);
    bool writeMemoryWindow(PacketPtr pkt, std::vector<uint8_t> &window,
                           Addr windowBase);
    bool readCsr(PacketPtr pkt, Addr offset);
    bool writeCsr(PacketPtr pkt, Addr offset);
    void startExecution();
    void stopExecution(bool setHalted);
    void completeExecution();

  public:
    CoralStageABackend(Addr pio_addr, Addr pio_size, Addr itcm_size,
                       Addr dtcm_size, Tick execution_latency,
                       bool auto_halt);

    const char *name() const override { return "stage-a"; }
    bool read(PacketPtr pkt, Addr pio_addr) override;
    bool write(PacketPtr pkt, Addr pio_addr) override;
    TranslationGenPtr translate(Addr vaddr, Addr size) override;

    bool hasPendingEvent() const override;
    Tick nextEventTick() const override;
    void processEvent() override;
};

} // namespace gem5

#endif // __DEV_NPU_CORAL_STAGEA_BACKEND_HH__
