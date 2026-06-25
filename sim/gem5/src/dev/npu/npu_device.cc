/*
 * Coral-compatible NPU device shell for gem5.
 */

#include "dev/npu/npu_device.hh"

#include "base/addr_range.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/NPUDevice.hh"
#include "dev/npu/coral_stagea_backend.hh"
#include "dev/npu/coral_verilated_backend.hh"
#include "mem/packet_access.hh"

namespace gem5
{

namespace
{

constexpr Addr kBackendIdOffset = 0x30ffc;
constexpr Addr kFirmwareEntryOffset = 0x30ff8;
constexpr uint32_t kStageABackendId = 0x4e505501;
constexpr uint32_t kVerilatedCoralBackendId = 0x4e505502;

} // namespace

NPUDevice::NPUDevice(const Params &p)
  : DmaVirtDevice(p),
    pioAddr(p.pioAddr),
    pioSize(p.pioSize),
    pioDelay(p.pioDelay),
    backendId(0),
    firmwareEntry(0),
    dmaActive(false),
    backendEvent(*this)
{
    if (p.backendType == "stage-a") {
        backendId = kStageABackendId;
        backend = std::make_unique<CoralStageABackend>(
            pioAddr, pioSize, p.itcmSize, p.dtcmSize, p.executionLatency,
            p.autoHalt);
    } else if (p.backendType == "verilated-coral") {
        backendId = kVerilatedCoralBackendId;
        auto rtlBackend = std::make_unique<CoralVerilatedBackend>(
            p.coralRepo, p.verilatedWrapper, p.rtlFirmware, p.rtlTickPeriod,
            p.rtlCyclesPerEvent);
        firmwareEntry = rtlBackend->entryPoint();
        backend = std::move(rtlBackend);
    } else {
        fatal("Unknown NPU backend type '%s'", p.backendType);
    }

    DPRINTFR(NPUDevice, "Created NPUDevice backend=%s pio=%#x size=%#x\n",
             backend->name(), pioAddr, pioSize);
    syncBackendEvent();
}

void
NPUDevice::processBackendEvent()
{
    backend->processEvent();
    startBackendDma();
    syncBackendEvent();
}

void
NPUDevice::startBackendDma()
{
    if (dmaActive || !backend->hasDmaRequest()) {
        return;
    }

    const CoralDmaRequest &request = backend->dmaRequest();
    auto *callback =
        new DmaVirtCallback<std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES>>(
            [this](const auto &data) { completeBackendDma(data); },
            request.data);
    dmaActive = true;

    if (request.type == CoralDmaType::Read) {
        dmaReadVirt(request.addr, request.size, callback,
                    callback->dmaBuffer.data());
    } else {
        dmaWriteVirt(request.addr, request.size, callback,
                     callback->dmaBuffer.data());
    }
}

void
NPUDevice::completeBackendDma(
    const std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES> &data)
{
    fatal_if(!dmaActive, "Coral NPU DMA completion without active request");
    const CoralDmaRequest &request = backend->dmaRequest();
    backend->completeDma(data.data(), request.size, false);
    dmaActive = false;
    syncBackendEvent();
}

void
NPUDevice::syncBackendEvent()
{
    if (backend->hasPendingEvent()) {
        const Tick when = backend->nextEventTick();
        if (!backendEvent.scheduled() || backendEvent.when() != when) {
            if (backendEvent.scheduled()) {
                deschedule(backendEvent);
            }
            schedule(backendEvent, when);
        }
    } else if (backendEvent.scheduled()) {
        deschedule(backendEvent);
    }
}

Tick
NPUDevice::read(PacketPtr pkt)
{
    DPRINTFR(NPUDevice, "backend=%s read addr=%#x size=%u\n",
             backend->name(), pkt->getAddr(), pkt->getSize());

    const Addr offset = pkt->getAddr() - pioAddr;
    if (offset == kFirmwareEntryOffset &&
        pkt->getSize() == sizeof(firmwareEntry)) {
        pkt->setLE<uint32_t>(firmwareEntry);
        pkt->makeAtomicResponse();
    } else if (offset == kBackendIdOffset &&
        pkt->getSize() == sizeof(backendId)) {
        pkt->setLE<uint32_t>(backendId);
        pkt->makeAtomicResponse();
    } else if (!backend->read(pkt, pioAddr)) {
        DPRINTFR(NPUDevice, "backend=%s bad read addr=%#x size=%u\n",
                 backend->name(), pkt->getAddr(), pkt->getSize());
        pkt->makeAtomicResponse();
        pkt->setBadAddress();
    }

    syncBackendEvent();
    return pioDelay;
}

Tick
NPUDevice::write(PacketPtr pkt)
{
    DPRINTFR(NPUDevice, "backend=%s write addr=%#x size=%u\n",
             backend->name(), pkt->getAddr(), pkt->getSize());

    if (!backend->write(pkt, pioAddr)) {
        DPRINTFR(NPUDevice, "backend=%s bad write addr=%#x size=%u\n",
                 backend->name(), pkt->getAddr(), pkt->getSize());
        pkt->makeAtomicResponse();
        pkt->setBadAddress();
    }

    syncBackendEvent();
    return pioDelay;
}

AddrRangeList
NPUDevice::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(RangeSize(pioAddr, pioSize));
    return ranges;
}

TranslationGenPtr
NPUDevice::translate(Addr vaddr, Addr size)
{
    return backend->translate(vaddr, size);
}

} // namespace gem5
