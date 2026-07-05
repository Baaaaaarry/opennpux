/*
 * Coral-compatible NPU device shell for gem5.
 */

#include "dev/npu/npu_device.hh"

#include <algorithm>
#include <limits>

#include "base/addr_range.hh"
#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/NPUDevice.hh"
#include "dev/npu/coral_stagea_backend.hh"
#include "dev/npu/coral_verilated_backend.hh"
#include "mem/packet_access.hh"
#include "sim/serialize.hh"

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
constexpr Addr kDmaErrorsOffset = 0x30fe0;
constexpr Addr kResetControlOffset = 0x30000;
constexpr uint32_t kStageABackendId = 0x4e505501;
constexpr uint32_t kVerilatedCoralBackendId = 0x4e505502;
constexpr Addr kFastDmaPageSize = 4096;

} // namespace

NPUDevice::NPUDevice(const Params &p)
  : DmaVirtDevice(p),
    pioAddr(p.pioAddr),
    pioSize(p.pioSize),
    pioDelay(p.pioDelay),
    dmaExtmemBase(p.dmaExtmemBase),
    dmaSharedBase(p.dmaSharedBase),
    dmaSharedSize(p.dmaSharedSize),
    fastDma(p.fastDma),
    backendId(0),
    firmwareEntry(0),
    dmaActive(false),
    dmaRequests(0),
    dmaCompletions(0),
    dmaErrors(0),
    fastDmaCache(fastDma ? dmaSharedSize : 0, 0),
    fastDmaPageValid(fastDma ?
        (dmaSharedSize + kFastDmaPageSize - 1) / kFastDmaPageSize : 0,
        false),
    fastDmaPageDirty(fastDmaPageValid.size(), false),
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

bool
NPUDevice::dmaQuiesced() const
{
    return !dmaActive && !backend->hasDmaRequest() &&
           !backend->hasPendingEvent();
}

void
NPUDevice::checkDrainDone()
{
    if (drainState() == DrainState::Draining && dmaQuiesced()) {
        DPRINTFR(NPUDevice, "Coral NPU drain completed\n");
        signalDrainDone();
    }
}

DrainState
NPUDevice::drain()
{
    if (dmaQuiesced()) {
        return DrainState::Drained;
    }

    DPRINTFR(NPUDevice,
             "Coral NPU draining active=%u backend_dma=%u pending_event=%u\n",
             dmaActive, backend->hasDmaRequest(),
             backend->hasPendingEvent());
    return DrainState::Draining;
}

void
NPUDevice::serialize(CheckpointOut &cp) const
{
    fatal_if(!dmaQuiesced(),
             "Cannot serialize Coral NPU while RTL/DMA is in flight");
    DmaVirtDevice::serialize(cp);
    SERIALIZE_SCALAR(dmaRequests);
    SERIALIZE_SCALAR(dmaCompletions);
    SERIALIZE_SCALAR(dmaErrors);
}

void
NPUDevice::unserialize(CheckpointIn &cp)
{
    DmaVirtDevice::unserialize(cp);
    UNSERIALIZE_OPT_SCALAR(dmaRequests);
    UNSERIALIZE_OPT_SCALAR(dmaCompletions);
    UNSERIALIZE_OPT_SCALAR(dmaErrors);
}

void
NPUDevice::processBackendEvent()
{
    // Functional inference does not assign timing meaning to each DMA or RTL
    // batch. Keep crossing the Verilator/gem5 boundary locally until the
    // batch budget is exhausted or execution stops. Timing mode processes one
    // batch and preserves the original event/DMA scheduling behavior.
    const unsigned batchBudget = fastDma ? 1024 : 1;
    for (unsigned batch = 0; batch < batchBudget; ++batch) {
        backend->processEvent();
        startBackendDma();
        if (!fastDma || dmaActive || !backend->hasPendingEvent()) {
            break;
        }
        if (backendEvent.scheduled()) {
            deschedule(backendEvent);
        }
    }
    if (fastDma && !backend->hasPendingEvent() &&
        !backend->hasDmaRequest()) {
        flushFastDmaCache();
    }
    syncBackendEvent();
    checkDrainDone();
}

void
NPUDevice::startBackendDma()
{
    if (dmaActive || !backend->hasDmaRequest()) {
        return;
    }

    const CoralDmaRequest &request = backend->dmaRequest();
    if (request.addr < dmaExtmemBase ||
        request.addr - dmaExtmemBase >= dmaSharedSize ||
        request.size > dmaSharedSize - (request.addr - dmaExtmemBase)) {
        DPRINTFR(NPUDevice,
                 "Coral DMA rejected type=%s coral=%#x size=%u outside "
                 "EXTMEM window [%#x:%#x]\n",
                 request.type == CoralDmaType::Read ? "read" : "write",
                 request.addr, request.size, dmaExtmemBase,
                 dmaExtmemBase + dmaSharedSize - 1);
        completeBackendDmaError();
        return;
    }
    const Addr hostAddr =
        dmaSharedBase + (request.addr - dmaExtmemBase);
    auto *callback =
        new DmaVirtCallback<std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES>>(
            [this](const auto &data) { completeBackendDma(data); },
            request.data);
    dmaActive = true;
    ++dmaRequests;
    if (!fastDma || dmaRequests <= 10 || dmaRequests % 1000 == 0) {
        DPRINTFR(NPUDevice,
                 "Coral DMA start count=%u type=%s coral=%#x host=%#x "
                 "size=%u mode=%s\n",
                 dmaRequests,
                 request.type == CoralDmaType::Read ? "read" : "write",
                 request.addr, hostAddr, request.size,
                 fastDma ? "functional" : "timing");
    }

    if (fastDma) {
        std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES> data = request.data;
        fastDmaAccess(request, hostAddr, data);
        completeBackendDma(data);
        return;
    }

    if (request.type == CoralDmaType::Read) {
        dmaReadVirt(hostAddr, request.size, callback,
                    callback->dmaBuffer.data());
    } else {
        dmaWriteVirt(hostAddr, request.size, callback,
                     callback->dmaBuffer.data());
    }
}

void
NPUDevice::functionalMemoryAccess(MemCmd command, Addr addr, size_t size,
                                  uint8_t *data)
{
    RequestPtr request = std::make_shared<Request>(
        addr, size, 0, dmaPort.requestorId);
    request->taskId(context_switch_task_id::DMA);
    Packet packet(request, command);
    packet.dataStatic(data);
    dmaPort.sendFunctional(&packet);
    fatal_if(packet.isError(),
             "Coral fast DMA functional access failed at %#x size=%llu",
             addr, static_cast<unsigned long long>(size));
}

void
NPUDevice::functionalMemoryRange(MemCmd command, Addr addr, size_t size,
                                 uint8_t *data)
{
    const size_t blockSize = cacheBlockSize();
    fatal_if(blockSize == 0, "Coral fast DMA cache block size is zero");
    size_t completed = 0;
    while (completed < size) {
        const Addr current = addr + completed;
        const size_t boundary = blockSize - current % blockSize;
        const size_t chunk = std::min(boundary, size - completed);
        functionalMemoryAccess(command, current, chunk, data + completed);
        completed += chunk;
    }
}

void
NPUDevice::fastDmaAccess(
    const CoralDmaRequest &request, Addr hostAddr,
    std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES> &data)
{
    const Addr offset = hostAddr - dmaSharedBase;
    const size_t page = offset / kFastDmaPageSize;
    const size_t inPage = offset % kFastDmaPageSize;
    fatal_if(page >= fastDmaPageValid.size() ||
                 request.size > kFastDmaPageSize - inPage,
             "Coral fast DMA request crosses cache page addr=%#x size=%u",
             request.addr, request.size);

    const size_t pageOffset = page * kFastDmaPageSize;
    const size_t pageSize = std::min<size_t>(
        kFastDmaPageSize, fastDmaCache.size() - pageOffset);
    if (!fastDmaPageValid[page]) {
        functionalMemoryRange(MemCmd::ReadReq,
                              dmaSharedBase + pageOffset, pageSize,
                              fastDmaCache.data() + pageOffset);
        fastDmaPageValid[page] = true;
    }

    auto *cacheData = fastDmaCache.data() + offset;
    if (request.type == CoralDmaType::Read) {
        std::copy_n(cacheData, request.size, data.data());
    } else {
        std::copy_n(data.data(), request.size, cacheData);
        fastDmaPageDirty[page] = true;
    }
}

void
NPUDevice::flushFastDmaCache()
{
    for (size_t page = 0; page < fastDmaPageDirty.size(); ++page) {
        if (!fastDmaPageDirty[page]) {
            continue;
        }
        const size_t offset = page * kFastDmaPageSize;
        const size_t size = std::min<size_t>(
            kFastDmaPageSize, fastDmaCache.size() - offset);
        functionalMemoryRange(MemCmd::WriteReq, dmaSharedBase + offset,
                              size, fastDmaCache.data() + offset);
        fastDmaPageDirty[page] = false;
    }
}

void
NPUDevice::invalidateFastDmaCache()
{
    std::fill(fastDmaPageValid.begin(), fastDmaPageValid.end(), false);
    std::fill(fastDmaPageDirty.begin(), fastDmaPageDirty.end(), false);
}

void
NPUDevice::completeBackendDma(
    const std::array<uint8_t, CORAL_GEM5_DMA_DATA_BYTES> &data)
{
    fatal_if(!dmaActive, "Coral NPU DMA completion without active request");
    const CoralDmaRequest &request = backend->dmaRequest();
    dmaActive = false;
    backend->completeDma(data.data(), request.size, false);
    ++dmaCompletions;
    if (!fastDma || dmaCompletions <= 10 || dmaCompletions % 1000 == 0) {
        DPRINTFR(NPUDevice, "Coral DMA complete count=%u mode=%s\n",
                 dmaCompletions, fastDma ? "functional" : "timing");
    }
    fatal_if(backend->hasDmaRequest(),
             "Coral backend retained DMA request after completion");
    syncBackendEvent();
    checkDrainDone();
}

void
NPUDevice::completeBackendDmaError()
{
    fatal_if(dmaActive, "Coral NPU DMA error while DMA is active");
    fatal_if(!backend->hasDmaRequest(),
             "Coral NPU DMA error without pending backend request");
    backend->completeDma(nullptr, 0, true);
    ++dmaErrors;
    DPRINTFR(NPUDevice, "Coral DMA error count=%u\n", dmaErrors);
    fatal_if(backend->hasDmaRequest(),
             "Coral backend retained DMA request after error completion");
    syncBackendEvent();
    checkDrainDone();
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
    } else if (offset == kDmaErrorsOffset &&
        pkt->getSize() == sizeof(uint32_t)) {
        pkt->setLE<uint32_t>(dmaErrors);
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

    const Addr offset = pkt->getAddr() - pioAddr;
    if (fastDma && offset == kResetControlOffset &&
        pkt->getSize() == sizeof(uint32_t) &&
        (pkt->getLE<uint32_t>() & 0x1) != 0) {
        invalidateFastDmaCache();
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
