#include "hw_sim/gem5_bridge/coralnpu_gem5_abi.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

#include "hw_sim/gem5_bridge/gem5_core_mini_axi_wrapper.h"
#include "hw_sim/gem5_bridge/coral_mobilenet.h"
#include "hw_sim/gem5_bridge/gem5_custom_mac.h"
#include "hw_sim/gem5_bridge/gem5_dma_request_builder.h"

namespace {

constexpr uint8_t kAxiSlvErr = 2;
constexpr uint32_t kFirmwareProgressAddr =
    OPENNPUX_CORAL_MOBILENET_PROGRESS_ADDR;

bool IsCustomWordAccess(const AxiAddr& addr) {
  return addr.addr_bits_len == 0 && addr.addr_bits_size == 2 &&
         addr.addr_bits_burst == 1 && (addr.addr_bits_addr & 3) == 0;
}

}  // namespace

struct coral_gem5_handle {
  VerilatedContext context;
  Gem5CoreMiniAxiWrapper wrapper;
  Gem5CustomMac custom_mac;
  uint32_t firmware_progress;
  coral_gem5_dma_request pending_dma;
  bool dma_pending;
  uint32_t dma_beat_size;
  uint32_t dma_beat_count;

  coral_gem5_handle()
      : context(),
        wrapper(&context),
        custom_mac(&context),
        firmware_progress(0),
        pending_dma(),
        dma_pending(false),
        dma_beat_size(0),
        dma_beat_count(0) {
    wrapper.RegisterDeferredReadCallback([this](const AxiAddr& addr) {
      if (custom_mac.Contains(addr.addr_bits_addr, 4)) {
        AxiRData response = {};
        response.read_data_bits_id = addr.addr_bits_id;
        response.read_data_bits_last = 1;
        if (!IsCustomWordAccess(addr)) {
          response.read_data_bits_resp = kAxiSlvErr;
        } else {
          const uint32_t value =
              addr.addr_bits_addr == kFirmwareProgressAddr
                  ? firmware_progress
                  : custom_mac.Read32(addr.addr_bits_addr);
          auto* destination = reinterpret_cast<uint8_t*>(
              &response.read_data_bits_data[0]);
          std::memcpy(destination + (addr.addr_bits_addr & 0xf), &value,
                      sizeof(value));
        }
        wrapper.QueueReadResponse(response);
        return;
      }
      if (!BuildGem5DmaReadRequest(
              addr, &pending_dma, &dma_beat_size, &dma_beat_count)) {
        AxiRData response = {};
        response.read_data_bits_id = addr.addr_bits_id;
        response.read_data_bits_resp = kAxiSlvErr;
        response.read_data_bits_last = 1;
        wrapper.QueueReadResponse(response);
        return;
      }
      dma_pending = true;
    });
    wrapper.RegisterDeferredWriteCallback(
        [this](const AxiAddr& addr, const std::vector<AxiWData>& data) {
          if (custom_mac.Contains(addr.addr_bits_addr, 4)) {
            AxiWResp response = {};
            response.write_resp_bits_id = addr.addr_bits_id;
            const uint32_t lane = addr.addr_bits_addr & 0xf;
            if (!IsCustomWordAccess(addr) || data.size() != 1 ||
                data[0].write_data_bits_last == 0 ||
                ((data[0].write_data_bits_strb >> lane) & 0xf) != 0xf) {
              response.write_resp_bits_resp = kAxiSlvErr;
            } else {
              uint32_t value = 0;
              const auto* source = reinterpret_cast<const uint8_t*>(
                  &data[0].write_data_bits_data[0]);
              std::memcpy(&value, source + lane, sizeof(value));
              if (addr.addr_bits_addr == kFirmwareProgressAddr) {
                firmware_progress = value;
                std::fprintf(stderr, "Coral firmware progress=0x%08x\n",
                             firmware_progress);
                std::fflush(stderr);
              } else {
                custom_mac.Write32(addr.addr_bits_addr, value);
              }
            }
            wrapper.QueueWriteResponse(response);
            return;
          }
          if (!BuildGem5DmaWriteRequest(addr, data, &pending_dma)) {
            AxiWResp response = {};
            response.write_resp_bits_id = addr.addr_bits_id;
            response.write_resp_bits_resp = kAxiSlvErr;
            wrapper.QueueWriteResponse(response);
            return;
          }
          dma_pending = true;
        });
    wrapper.Reset();
  }
};

extern "C" uint32_t
coral_gem5_abi_version(void)
{
  return CORAL_GEM5_ABI_VERSION;
}

extern "C" coral_gem5_handle*
coral_gem5_create(void)
{
  return new (std::nothrow) coral_gem5_handle();
}

extern "C" void
coral_gem5_destroy(coral_gem5_handle* handle)
{
  delete handle;
}

extern "C" int
coral_gem5_reset(coral_gem5_handle* handle)
{
  if (handle == nullptr) {
    return -1;
  }
  handle->wrapper.Reset();
  handle->custom_mac.Reset();
  handle->firmware_progress = 0;
  return 0;
}

extern "C" int
coral_gem5_mmio_read(
    coral_gem5_handle* handle, uint32_t addr, void* data, size_t size)
{
  if (handle == nullptr || data == nullptr || size == 0) {
    return -1;
  }
  const std::vector<uint8_t> value = handle->wrapper.Read(addr, size);
  if (value.size() != size) {
    return -1;
  }
  std::memcpy(data, value.data(), size);
  return 0;
}

extern "C" int
coral_gem5_mmio_write(
    coral_gem5_handle* handle, uint32_t addr, const void* data, size_t size)
{
  if (handle == nullptr || data == nullptr || size == 0) {
    return -1;
  }
  handle->wrapper.Write(
      addr, size, reinterpret_cast<const char*>(data));
  return 0;
}

extern "C" int
coral_gem5_step(coral_gem5_handle* handle, uint32_t cycles)
{
  if (handle == nullptr || cycles == 0) {
    return -1;
  }
  if (handle->dma_pending) {
    return 2;
  }
  for (uint32_t i = 0; i < cycles; ++i) {
    if (handle->wrapper.HasFault()) {
      return 4;
    }
    if (handle->wrapper.IsHalted()) {
      return 1;
    }
    handle->wrapper.Step();
    handle->custom_mac.Step();
    if (handle->dma_pending) {
      return 2;
    }
    if (handle->wrapper.HasFault()) {
      return 4;
    }
    if (handle->wrapper.IsHalted()) {
      return 1;
    }
  }
  return handle->wrapper.IsWfi() ? 3 : 0;
}

extern "C" int
coral_gem5_dma_request_get(
    coral_gem5_handle* handle, coral_gem5_dma_request* request)
{
  if (handle == nullptr || request == nullptr) {
    return -1;
  }
  if (!handle->dma_pending) {
    return 0;
  }
  *request = handle->pending_dma;
  return 1;
}

extern "C" int
coral_gem5_dma_complete(
    coral_gem5_handle* handle, const void* data, size_t size, int error)
{
  if (handle == nullptr || !handle->dma_pending) {
    return -1;
  }

  const uint8_t response = error ? 2 : 0;
  if (handle->pending_dma.type == CORAL_GEM5_DMA_READ) {
    if (!error &&
        (data == nullptr || size != handle->pending_dma.size)) {
      return -1;
    }
    if (!error) {
      const auto* source = reinterpret_cast<const uint8_t*>(data);
      for (uint32_t beat = 0; beat < handle->dma_beat_count; ++beat) {
        AxiRData axi_response = {};
        const uint32_t beat_addr =
            handle->pending_dma.addr + beat * handle->dma_beat_size;
        const uint32_t lane =
            beat_addr & (CORAL_GEM5_AXI_DATA_BYTES - 1);
        auto* destination = reinterpret_cast<uint8_t*>(
            &axi_response.read_data_bits_data[0]);
        std::memcpy(destination + lane,
                    source + beat * handle->dma_beat_size,
                    handle->dma_beat_size);
        axi_response.read_data_bits_id = handle->pending_dma.id;
        axi_response.read_data_bits_resp = 0;
        axi_response.read_data_bits_last =
            beat + 1 == handle->dma_beat_count;
        handle->wrapper.QueueReadResponse(axi_response);
      }
    } else {
      AxiRData axi_response = {};
      axi_response.read_data_bits_id = handle->pending_dma.id;
      axi_response.read_data_bits_resp = response;
      axi_response.read_data_bits_last = 1;
      handle->wrapper.QueueReadResponse(axi_response);
    }
  } else if (handle->pending_dma.type == CORAL_GEM5_DMA_WRITE) {
    AxiWResp axi_response = {};
    axi_response.write_resp_bits_id = handle->pending_dma.id;
    axi_response.write_resp_bits_resp = response;
    handle->wrapper.QueueWriteResponse(axi_response);
  } else {
    return -1;
  }

  handle->dma_pending = false;
  std::memset(&handle->pending_dma, 0, sizeof(handle->pending_dma));
  return 0;
}
