/*
 * Coral-compatible NPU device shell for gem5.
 */

#include "dev/npu/npu_device.hh"

#include <limits>

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
constexpr Addr kSharedSizeOffset = 0x30ff4;
constexpr Addr kSharedBaseOffset = 0x30ff0;
constexpr Addr kDmaStateOffset = 0x30fec;
constexpr Addr kDmaCompletionsOffset = 0x30fe8;
constexpr Addr kDmaRequestsOffset = 0x30fe4;
constexpr uint32_t kStageABackendId = 0x4e505501;
constexpr uint32_t kVerilatedCoralBackendId = 0x4e505502;

} // namespace

NPUDevice::NPUDevice(const Params &p)
  : DmaVirtDevice(p),
    pioAddr(p.pioAddr),
    pioSize(p.pioSize),
    pioDelay(p.pioDelay),
    dmaExtmemBase(p.dmaExtmemBase),
    dmaSharedBase(p.dmaSharedBase),
    dmaSharedSize(p.dmaSharedSize),
    backendId(0),
    firmwareEntry(0),
    dmaActive(false),
    dmaRequests(0),
    dmaCompletions(0),
    backendEvent(*this)
{
    fatal_if(dmaSharedSize == 0, "Coral NPU DMA shared size is zero");
    fatal_if(dmaSharedBase > std::numeric_limits<uint32_t>::max(),
             "Coral NPU DMA shared base must fit the 32-bit shell CSR");
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
    fatal_if(request.addr < dmaExtmemBase ||
                 request.addr - dmaExtmemBase >= dmaSharedSize ||
                 request.size >
                     dmaSharedSize - (request.addr - dmaExtmemBase),
             "Coral DMA address %#x size=%u is outside EXTMEM window "
             "[%#x:%#x]",
             request.addr, request.size, dmaExtmemBase,
             dmaExtmemBase + dmaSharedSize - 1);
    const Addr hostAddr =
        dmaSharedBase + (request.addr - dmaExtmemBase);
    auto *callback =
        new DmaVirtCallback<std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES>>(
            [this](const auto &data) { completeBackendDma(data); },
            request.data);
    dmaActive = true;
    ++dmaRequests;
    DPRINTFR(NPUDevice,
             "Coral DMA start count=%u type=%s coral=%#x host=%#x size=%u\n",
             dmaRequests,
             request.type == CoralDmaType::Read ? "read" : "write",
             request.addr, hostAddr, request.size);

    if (request.type == CoralDmaType::Read) {
        dmaReadVirt(hostAddr, request.size, callback,
                    callback->dmaBuffer.data());
    } else {
        dmaWriteVirt(hostAddr, request.size, callback,
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
    ++dmaCompletions;
    DPRINTFR(NPUDevice, "Coral DMA complete count=%u\n", dmaCompletions);
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
    if (offset == kDmaRequestsOffset &&
        pkt->getSize() == sizeof(uint32_t)) {
        pkt->setLE<uint32_t>(dmaRequests);
        pkt->makeAtomicResponse();
    } else if (offset == kDmaCompletionsOffset &&
        pkt->getSize() == sizeof(uint32_t)) {
        pkt->setLE<uint32_t>(dmaCompletions);
        pkt->makeAtomicResponse();
    } else if (offset == kDmaStateOffset &&
        pkt->getSize() == sizeof(uint32_t)) {
        const uint32_t state =
            (backend->hasDmaRequest() ? 0x1 : 0x0) |
            (dmaActive ? 0x2 : 0x0);
        pkt->setLE<uint32_t>(state);
        pkt->makeAtomicResponse();
    } else if (offset == kSharedBaseOffset &&
        pkt->getSize() == sizeof(uint32_t)) {
        pkt->setLE<uint32_t>(dmaSharedBase);
        pkt->makeAtomicResponse();
    } else if (offset == kSharedSizeOffset &&
        pkt->getSize() == sizeof(uint32_t)) {
        pkt->setLE<uint32_t>(dmaSharedSize);
        pkt->makeAtomicResponse();
    } else if (offset == kFirmwareEntryOffset &&
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
