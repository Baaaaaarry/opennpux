/*
 * Runtime-loaded bridge to the official Coral CoreMiniAxi RTL model.
 */

#include "dev/npu/coral_verilated_backend.hh"

#include <dlfcn.h>

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
                                             Tick rtl_tick_period,
                                             uint32_t rtl_cycles_per_event)
  : coralRepo(coral_repo),
    wrapperPath(wrapper_path),
    rtlTickPeriod(rtl_tick_period),
    rtlCyclesPerEvent(rtl_cycles_per_event),
    libraryHandle(nullptr),
    modelHandle(nullptr),
    destroyModel(nullptr),
    mmioRead(nullptr),
    mmioWrite(nullptr),
    stepModel(nullptr),
    running(false),
    pendingEventTick(0),
    resetControl(kResetBit | kClockGateBit)
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

    fatal_if(abiVersion() != CORAL_GEM5_ABI_VERSION,
             "Coral RTL bridge ABI mismatch: gem5=%u library=%u",
             CORAL_GEM5_ABI_VERSION, abiVersion());

    modelHandle = createModel();
    fatal_if(modelHandle == nullptr,
             "Coral RTL bridge failed to create CoreMiniAxi model");
    fatal_if(resetModel(modelHandle) != 0,
             "Coral RTL bridge failed to reset CoreMiniAxi model");

    DPRINTFR(NPUDevice,
             "Loaded Coral RTL bridge '%s' repo='%s' tick=%llu cycles=%u\n",
             wrapperPath, coralRepo, rtlTickPeriod, rtlCyclesPerEvent);
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
        resetControl = pkt->getLE<uint32_t>() &
            (kResetBit | kClockGateBit);
        running = !(resetControl & (kResetBit | kClockGateBit));
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

    const int result = stepModel(modelHandle, rtlCyclesPerEvent);
    fatal_if(result < 0, "Coral RTL bridge failed while stepping model");

    if (result > 0) {
        running = false;
        pendingEventTick = 0;
        DPRINTFR(NPUDevice, "Coral RTL reached halted/WFI state\n");
    } else {
        pendingEventTick = curTick() + rtlTickPeriod;
    }
}

} // namespace gem5
