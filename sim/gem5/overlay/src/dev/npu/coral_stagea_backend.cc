/*
 * Stage-A Coral-compatible transaction-level backend.
 */

#include "dev/npu/coral_stagea_backend.hh"

#include <vector>

#include "base/logging.hh"
#include "debug/NPUDevice.hh"
#include "mem/packet_access.hh"

namespace gem5
{

namespace
{

constexpr uint32_t kResetBit = 1u << 0;
constexpr uint32_t kClockGateBit = 1u << 1;

class IdentityTranslationGen : public TranslationGen
{
  public:
    IdentityTranslationGen(Addr vaddr, Addr size)
      : TranslationGen(vaddr, size)
    {}

  private:
    void translate(Range &range) const override { range.paddr = range.vaddr; }
};

} // namespace

CoralStageABackend::CoralStageABackend(Addr pio_addr, Addr pio_size,
                                       Addr itcm_size, Addr dtcm_size,
                                       Tick execution_latency,
                                       bool auto_halt)
  : pioAddr(pio_addr),
    pioSize(pio_size),
    itcmSize(itcm_size),
    dtcmSize(dtcm_size),
    executionLatency(execution_latency),
    autoHalt(auto_halt),
    resetControl(kResetBit | kClockGateBit),
    pcStart(0),
    halted(false),
    fault(false),
    pendingEventTick(0),
    itcm(itcmSize, 0),
    dtcm(dtcmSize, 0)
{
    const Addr minPioSize = kCsrOffset + kCsrSize;
    fatal_if(pioSize < minPioSize,
        "NPUDevice aperture %#x is too small; need at least %#x bytes",
        pioSize, minPioSize);
}

bool
CoralStageABackend::inItcm(Addr offset, Addr size) const
{
    return offset < itcmSize && size <= itcmSize - offset;
}

bool
CoralStageABackend::inDtcm(Addr offset, Addr size) const
{
    return offset >= kDtcmOffset && offset - kDtcmOffset < dtcmSize &&
           size <= dtcmSize - (offset - kDtcmOffset);
}

bool
CoralStageABackend::inCsr(Addr offset, Addr size) const
{
    return offset >= kCsrOffset && offset - kCsrOffset < kCsrSize &&
           size <= kCsrSize - (offset - kCsrOffset);
}

bool
CoralStageABackend::readMemoryWindow(PacketPtr pkt, std::vector<uint8_t> &window,
                                     Addr windowBase)
{
    const Addr offset = pkt->getAddr() - pioAddr - windowBase;
    if (offset + pkt->getSize() > window.size()) {
        return false;
    }

    pkt->setData(&window[offset]);
    pkt->makeAtomicResponse();
    return true;
}

bool
CoralStageABackend::writeMemoryWindow(PacketPtr pkt,
                                      std::vector<uint8_t> &window,
                                      Addr windowBase)
{
    const Addr offset = pkt->getAddr() - pioAddr - windowBase;
    if (offset + pkt->getSize() > window.size()) {
        return false;
    }

    pkt->writeData(&window[offset]);
    pkt->makeAtomicResponse();
    return true;
}

bool
CoralStageABackend::readCsr(PacketPtr pkt, Addr offset)
{
    if (pkt->getSize() != sizeof(uint32_t) || (offset & 0x3) != 0) {
        return false;
    }

    uint32_t value = 0;
    switch (offset - kCsrOffset) {
      case 0x0:
        value = resetControl;
        break;
      case 0x4:
        value = pcStart;
        break;
      case 0x8:
        value = (halted ? 0x1 : 0x0) | (fault ? 0x2 : 0x0);
        break;
      default:
        return false;
    }

    pkt->setLE<uint32_t>(value);
    pkt->makeAtomicResponse();
    return true;
}

bool
CoralStageABackend::writeCsr(PacketPtr pkt, Addr offset)
{
    if (pkt->getSize() != sizeof(uint32_t) || (offset & 0x3) != 0) {
        return false;
    }

    switch (offset - kCsrOffset) {
      case 0x0: {
        const uint32_t oldControl = resetControl;
        resetControl = pkt->getLE<uint32_t>() & (kResetBit | kClockGateBit);

        if (resetControl & kResetBit) {
            stopExecution(false);
        } else if ((oldControl & kResetBit) &&
                   !(resetControl & kClockGateBit)) {
            startExecution();
        }
        break;
      }
      case 0x4:
        pcStart = pkt->getLE<uint32_t>();
        break;
      default:
        return false;
    }

    pkt->makeAtomicResponse();
    return true;
}

void
CoralStageABackend::startExecution()
{
    halted = false;
    fault = false;

    if (autoHalt) {
        pendingEventTick = curTick() + executionLatency;
    }
}

void
CoralStageABackend::stopExecution(bool setHalted)
{
    pendingEventTick = 0;
    halted = setHalted;
}

void
CoralStageABackend::completeExecution()
{
    halted = true;
    pendingEventTick = 0;
}

bool
CoralStageABackend::read(PacketPtr pkt, Addr pio_addr)
{
    const Addr offset = pkt->getAddr() - pio_addr;
    bool handled = false;

    if (inItcm(offset, pkt->getSize())) {
        handled = readMemoryWindow(pkt, itcm, 0);
    } else if (inDtcm(offset, pkt->getSize())) {
        handled = readMemoryWindow(pkt, dtcm, kDtcmOffset);
    } else if (inCsr(offset, pkt->getSize())) {
        handled = readCsr(pkt, offset);
    }

    return handled;
}

bool
CoralStageABackend::write(PacketPtr pkt, Addr pio_addr)
{
    const Addr offset = pkt->getAddr() - pio_addr;
    bool handled = false;

    if (inItcm(offset, pkt->getSize())) {
        handled = writeMemoryWindow(pkt, itcm, 0);
    } else if (inDtcm(offset, pkt->getSize())) {
        handled = writeMemoryWindow(pkt, dtcm, kDtcmOffset);
    } else if (inCsr(offset, pkt->getSize())) {
        handled = writeCsr(pkt, offset);
    }

    return handled;
}

TranslationGenPtr
CoralStageABackend::translate(Addr vaddr, Addr size)
{
    return TranslationGenPtr(new IdentityTranslationGen(vaddr, size));
}

bool
CoralStageABackend::hasPendingEvent() const
{
    return pendingEventTick != 0;
}

Tick
CoralStageABackend::nextEventTick() const
{
    return pendingEventTick;
}

void
CoralStageABackend::processEvent()
{
    completeExecution();
}

} // namespace gem5
