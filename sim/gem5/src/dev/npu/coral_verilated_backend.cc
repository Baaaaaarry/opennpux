/*
 * Runtime-loaded bridge to the official Coral CoreMiniAxi RTL model.
 */

#include "dev/npu/coral_verilated_backend.hh"

#include <dlfcn.h>
#include <elf.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>

#include "base/logging.hh"
#include "base/trace.hh"
#include "debug/NPUDevice.hh"
#include "mem/packet_access.hh"

namespace gem5
{

namespace
{

constexpr Addr kResetControlOffset = 0x30000;
constexpr Addr kMepcOffset = 0x30104;
constexpr Addr kMtvalOffset = 0x30108;
constexpr Addr kMcauseOffset = 0x3010c;
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

template <class Function>
Function
CoralVerilatedBackend::loadSymbol(const char *name)
{
    dlerror();
    void *symbol = dlsym(libraryHandle, name);
    const char *error = dlerror();
    fatal_if(error != nullptr || symbol == nullptr,
             "Unable to resolve Coral RTL symbol '%s' from '%s': %s",
             name, wrapperPath, error ? error : "symbol not found");
    return reinterpret_cast<Function>(symbol);
}

CoralVerilatedBackend::CoralVerilatedBackend(const std::string &coral_repo,
                                             const std::string &wrapper_path,
                                             const std::string &firmware_path,
                                             Tick rtl_tick_period,
                                             uint32_t rtl_cycles_per_event,
                                             bool enable_local_extmem)
  : coralRepo(coral_repo),
    wrapperPath(wrapper_path),
    firmwarePath(firmware_path),
    rtlTickPeriod(rtl_tick_period),
    rtlCyclesPerEvent(rtl_cycles_per_event),
    libraryHandle(nullptr),
    modelHandle(nullptr),
    destroyModel(nullptr),
    mmioRead(nullptr),
    mmioWrite(nullptr),
    stepModel(nullptr),
    dmaRequestGet(nullptr),
    dmaComplete(nullptr),
    extmemRead(nullptr),
    extmemWrite(nullptr),
    localExtmemEnabled(enable_local_extmem),
    running(false),
    pendingEventTick(0),
    resetControl(kResetBit | kClockGateBit),
    firmwareEntry(0),
    dmaRequestPending(false),
    wfiObserved(false),
    rtlEventCount(0),
    rtlDmaRequestCount(0),
    pendingDmaRequest()
{
    fatal_if(wrapperPath.empty(),
             "NPU backend 'verilated-coral' requires "
             "--npu-verilated-wrapper=<libcoralnpu_gem5_bridge.so>");
    fatal_if(rtlTickPeriod == 0, "Coral RTL tick period must be non-zero");
    fatal_if(rtlCyclesPerEvent == 0,
             "Coral RTL cycles per event must be non-zero");

    libraryHandle = dlopen(wrapperPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    fatal_if(libraryHandle == nullptr,
             "Unable to load Coral RTL bridge '%s': %s",
             wrapperPath, dlerror());

    const auto abiVersion = loadSymbol<AbiVersionFn>(
        "coral_gem5_abi_version");
    const auto createModel = loadSymbol<CreateFn>("coral_gem5_create");
    destroyModel = loadSymbol<DestroyFn>("coral_gem5_destroy");
    const auto resetModel = loadSymbol<ResetFn>("coral_gem5_reset");
    mmioRead = loadSymbol<MmioReadFn>("coral_gem5_mmio_read");
    mmioWrite = loadSymbol<MmioWriteFn>("coral_gem5_mmio_write");
    stepModel = loadSymbol<StepFn>("coral_gem5_step");
    dmaRequestGet = loadSymbol<DmaRequestGetFn>(
        "coral_gem5_dma_request_get");
    dmaComplete = loadSymbol<DmaCompleteFn>("coral_gem5_dma_complete");
    const auto extmemEnable = loadSymbol<ExtmemEnableFn>(
        "coral_gem5_extmem_enable");
    extmemRead = loadSymbol<ExtmemReadFn>("coral_gem5_extmem_read");
    extmemWrite = loadSymbol<ExtmemWriteFn>("coral_gem5_extmem_write");

    fatal_if(abiVersion() != CORAL_GEM5_ABI_VERSION,
             "Coral RTL bridge ABI mismatch: gem5=%u library=%u",
             CORAL_GEM5_ABI_VERSION, abiVersion());

    modelHandle = createModel();
    fatal_if(modelHandle == nullptr,
             "Coral RTL bridge failed to create AXI model");
    fatal_if(resetModel(modelHandle) != 0,
             "Coral RTL bridge failed to reset AXI model");
    fatal_if(extmemEnable(modelHandle, localExtmemEnabled ? 1 : 0) != 0,
             "Coral RTL bridge failed to configure local EXTMEM");
    loadFirmware();

    DPRINTFR(NPUDevice,
             "Loaded Coral RTL bridge '%s' firmware='%s' entry=%#x "
             "repo='%s' tick=%llu cycles=%u\n",
             wrapperPath, firmwarePath, firmwareEntry, coralRepo,
             rtlTickPeriod, rtlCyclesPerEvent);
}

void
CoralVerilatedBackend::loadFirmware()
{
    fatal_if(firmwarePath.empty(),
             "NPU backend 'verilated-coral' requires --npu-rtl-firmware");

    std::ifstream stream(firmwarePath, std::ios::binary | std::ios::ate);
    fatal_if(!stream, "Unable to open Coral RTL firmware '%s'", firmwarePath);
    const std::streamsize fileSize = stream.tellg();
    fatal_if(fileSize < static_cast<std::streamsize>(sizeof(Elf32_Ehdr)),
             "Coral RTL firmware '%s' is not a valid ELF file", firmwarePath);

    stream.seekg(0);
    std::vector<uint8_t> image(fileSize);
    fatal_if(!stream.read(reinterpret_cast<char *>(image.data()), fileSize),
             "Unable to read Coral RTL firmware '%s'", firmwarePath);

    const auto *header =
        reinterpret_cast<const Elf32_Ehdr *>(image.data());
    fatal_if(std::memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 ||
             header->e_ident[EI_CLASS] != ELFCLASS32 ||
             header->e_ident[EI_DATA] != ELFDATA2LSB,
             "Coral RTL firmware '%s' must be a 32-bit little-endian ELF",
             firmwarePath);
    fatal_if(header->e_phentsize != sizeof(Elf32_Phdr),
             "Coral RTL firmware '%s' has unsupported program headers",
             firmwarePath);

    const uint64_t tableEnd = static_cast<uint64_t>(header->e_phoff) +
        static_cast<uint64_t>(header->e_phnum) * sizeof(Elf32_Phdr);
    fatal_if(tableEnd > image.size(),
             "Coral RTL firmware '%s' has truncated program headers",
             firmwarePath);

    for (uint16_t i = 0; i < header->e_phnum; ++i) {
        const auto *segment = reinterpret_cast<const Elf32_Phdr *>(
            image.data() + header->e_phoff + i * sizeof(Elf32_Phdr));
        if (segment->p_type != PT_LOAD || segment->p_filesz == 0) {
            continue;
        }
        fatal_if(static_cast<uint64_t>(segment->p_offset) +
                     segment->p_filesz > image.size(),
                 "Coral RTL firmware '%s' has a truncated load segment",
                 firmwarePath);
        fatal_if(mmioWrite(modelHandle, segment->p_paddr,
                           image.data() + segment->p_offset,
                           segment->p_filesz) != 0,
                 "Unable to load Coral RTL firmware segment at %#x",
                 segment->p_paddr);

        // TCM is directly reachable through the model's AXI slave port.
        // EXTMEM/DDR use the deferred AXI master path and cannot be read back
        // synchronously while constructing the backend.
        if (segment->p_paddr >= 0x20000000) {
            DPRINTFR(NPUDevice,
                     "Loaded external Coral firmware LOAD segment addr=%#x "
                     "filesz=%#x memsz=%#x flags=%#x\n",
                     segment->p_paddr, segment->p_filesz, segment->p_memsz,
                     segment->p_flags);
            continue;
        }

        std::vector<uint8_t> loaded(segment->p_filesz);
        fatal_if(mmioRead(modelHandle, segment->p_paddr, loaded.data(),
                          loaded.size()) != 0,
                 "Unable to read back Coral RTL firmware segment at %#x",
                 segment->p_paddr);
        const auto *expected = image.data() + segment->p_offset;
        const auto mismatch = std::mismatch(
            loaded.begin(), loaded.end(), expected);
        if (mismatch.first != loaded.end()) {
            const auto offset = std::distance(loaded.begin(), mismatch.first);
            fatal("Coral RTL firmware verification failed at %#x: "
                  "expected=%#x actual=%#x",
                  segment->p_paddr + offset,
                  static_cast<unsigned int>(expected[offset]),
                  static_cast<unsigned int>(loaded[offset]));
        }
        DPRINTFR(NPUDevice,
                 "Verified Coral firmware LOAD segment addr=%#x "
                 "filesz=%#x memsz=%#x flags=%#x\n",
                 segment->p_paddr, segment->p_filesz, segment->p_memsz,
                 segment->p_flags);
    }

    firmwareEntry = header->e_entry;
}

CoralVerilatedBackend::~CoralVerilatedBackend()
{
    if (modelHandle != nullptr && destroyModel != nullptr) {
        destroyModel(modelHandle);
    }
    if (libraryHandle != nullptr) {
        dlclose(libraryHandle);
    }
}

bool
CoralVerilatedBackend::read(PacketPtr pkt, Addr pio_addr)
{
    const Addr offset = pkt->getAddr() - pio_addr;
    if (offset > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    std::vector<uint8_t> data(pkt->getSize());
    if (mmioRead(modelHandle, static_cast<uint32_t>(offset),
                 data.data(), data.size()) != 0) {
        return false;
    }

    pkt->setData(data.data());
    pkt->makeAtomicResponse();
    return true;
}

bool
CoralVerilatedBackend::write(PacketPtr pkt, Addr pio_addr)
{
    const Addr offset = pkt->getAddr() - pio_addr;
    if (offset > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    std::vector<uint8_t> data(pkt->getSize());
    pkt->writeData(data.data());
    if (mmioWrite(modelHandle, static_cast<uint32_t>(offset),
                  data.data(), data.size()) != 0) {
        return false;
    }

    if (offset == kResetControlOffset &&
        pkt->getSize() == sizeof(uint32_t)) {
        const bool wasRunning = running;
        resetControl = pkt->getLE<uint32_t>() &
            (kResetBit | kClockGateBit);
        running = !(resetControl & (kResetBit | kClockGateBit));
        if (running && !wasRunning) {
            rtlEventCount = 0;
            rtlDmaRequestCount = 0;
        }
        pendingEventTick = running ? curTick() + rtlTickPeriod : 0;
    }

    pkt->makeAtomicResponse();
    return true;
}

TranslationGenPtr
CoralVerilatedBackend::translate(Addr vaddr, Addr size)
{
    return TranslationGenPtr(new IdentityTranslationGen(vaddr, size));
}

bool
CoralVerilatedBackend::hasPendingEvent() const
{
    return pendingEventTick != 0;
}

Tick
CoralVerilatedBackend::nextEventTick() const
{
    return pendingEventTick;
}

void
CoralVerilatedBackend::processEvent()
{
    if (!running) {
        pendingEventTick = 0;
        return;
    }

    ++rtlEventCount;
    const bool traceProgress = rtlEventCount <= 10 ||
        rtlEventCount == 100 || rtlEventCount % 1000 == 0;
    if (traceProgress) {
        DPRINTFR(NPUDevice,
                 "Coral RTL step batch start event=%llu cycles=%u\n",
                 static_cast<unsigned long long>(rtlEventCount),
                 rtlCyclesPerEvent);
    }
    const int result = stepModel(modelHandle, rtlCyclesPerEvent);
    fatal_if(result < 0, "Coral RTL bridge failed while stepping model");
    if (result == 4) {
        uint32_t mepc = 0;
        uint32_t mtval = 0;
        uint32_t mcause = 0;
        const bool faultCsrsRead =
            mmioRead(modelHandle, kMepcOffset, &mepc, sizeof(mepc)) == 0 &&
            mmioRead(modelHandle, kMtvalOffset, &mtval, sizeof(mtval)) == 0 &&
            mmioRead(modelHandle, kMcauseOffset, &mcause, sizeof(mcause)) == 0;
        fatal_if(!faultCsrsRead,
                 "Coral RTL raised a core fault and fault CSR readback failed");
        fatal("Coral RTL core fault: mepc=%#x mtval=%#x mcause=%#x",
              mepc, mtval, mcause);
    }
    if (traceProgress) {
        DPRINTFR(NPUDevice,
                 "Coral RTL heartbeat event=%llu requested_cycles=%llu "
                 "step_result=%d\n",
                 static_cast<unsigned long long>(rtlEventCount),
                 static_cast<unsigned long long>(
                     rtlEventCount * rtlCyclesPerEvent),
                 result);
    }

    if (result == 3) {
        if (!wfiObserved) {
            DPRINTFR(NPUDevice,
                     "Coral RTL entered WFI; keeping backend scheduled\n");
            wfiObserved = true;
        }
        pendingEventTick = curTick() + rtlTickPeriod;
    } else if (result == 2) {
        wfiObserved = false;
        coral_gem5_dma_request request = {};
        fatal_if(dmaRequestGet(modelHandle, &request) != 1,
                 "Coral RTL reported DMA wait without a request");
        fatal_if(request.size == 0 ||
                     request.size > pendingDmaRequest.data.size(),
                 "Unsupported Coral RTL DMA size %u", request.size);
        fatal_if(request.type != CORAL_GEM5_DMA_READ &&
                     request.type != CORAL_GEM5_DMA_WRITE,
                 "Unsupported Coral RTL DMA type %u", request.type);

        pendingDmaRequest.type = request.type == CORAL_GEM5_DMA_READ ?
            CoralDmaType::Read : CoralDmaType::Write;
        pendingDmaRequest.addr = request.addr;
        pendingDmaRequest.size = request.size;
        std::copy(std::begin(request.data), std::end(request.data),
                  pendingDmaRequest.data.begin());
        dmaRequestPending = true;
        pendingEventTick = 0;
        ++rtlDmaRequestCount;
        if (rtlDmaRequestCount <= 10 || rtlDmaRequestCount % 1000 == 0) {
            DPRINTFR(NPUDevice,
                     "Coral RTL waiting for DMA count=%llu type=%s "
                     "addr=%#x size=%u\n",
                     static_cast<unsigned long long>(rtlDmaRequestCount),
                     pendingDmaRequest.type == CoralDmaType::Read ?
                         "read" : "write",
                     pendingDmaRequest.addr, pendingDmaRequest.size);
        }
    } else if (result > 0) {
        wfiObserved = false;
        running = false;
        pendingEventTick = 0;
        DPRINTFR(NPUDevice, "Coral RTL reached halted/WFI state\n");
    } else {
        wfiObserved = false;
        pendingEventTick = curTick() + rtlTickPeriod;
    }
}

const CoralDmaRequest &
CoralVerilatedBackend::dmaRequest() const
{
    fatal_if(!dmaRequestPending, "Coral RTL has no pending DMA request");
    return pendingDmaRequest;
}

void
CoralVerilatedBackend::completeDma(
    const uint8_t *data, size_t size, bool error)
{
    fatal_if(!dmaRequestPending, "Coral RTL DMA completion without request");
    const void *responseData =
        pendingDmaRequest.type == CoralDmaType::Read ? data : nullptr;
    const size_t responseSize =
        pendingDmaRequest.type == CoralDmaType::Read ? size : 0;
    fatal_if(dmaComplete(modelHandle, responseData, responseSize,
                         error ? 1 : 0) != 0,
             "Coral RTL bridge rejected DMA completion");
    dmaRequestPending = false;
    pendingDmaRequest = {};
    pendingEventTick = running ? curTick() + rtlTickPeriod : 0;
}

void
CoralVerilatedBackend::readLocalExtmem(
    Addr addr, void *data, size_t size)
{
    fatal_if(!localExtmemEnabled || addr > std::numeric_limits<uint32_t>::max()
                 || extmemRead(modelHandle, addr, data, size) != 0,
             "Unable to read Coral bridge-local EXTMEM at %#x size=%llu",
             addr, static_cast<unsigned long long>(size));
}

void
CoralVerilatedBackend::writeLocalExtmem(
    Addr addr, const void *data, size_t size)
{
    fatal_if(!localExtmemEnabled || addr > std::numeric_limits<uint32_t>::max()
                 || extmemWrite(modelHandle, addr, data, size) != 0,
             "Unable to write Coral bridge-local EXTMEM at %#x size=%llu",
             addr, static_cast<unsigned long long>(size));
}

} // namespace gem5
