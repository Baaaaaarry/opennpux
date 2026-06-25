#include "hw_sim/gem5_bridge/coralnpu_gem5_abi.h"

#include <cstring>
#include <new>
#include <vector>

#include "hw_sim/core_mini_axi_wrapper.h"

struct coral_gem5_handle {
  VerilatedContext context;
  CoreMiniAxiWrapper wrapper;
  coral_gem5_dma_request pending_dma;
  bool dma_pending;
  uint32_t dma_lane;

  coral_gem5_handle()
      : context(), wrapper(&context), pending_dma(), dma_pending(false),
        dma_lane(0) {
    wrapper.RegisterDeferredReadCallback([this](const AxiAddr& addr) {
      std::memset(&pending_dma, 0, sizeof(pending_dma));
      pending_dma.type = CORAL_GEM5_DMA_READ;
      pending_dma.addr = addr.addr_bits_addr;
      pending_dma.size = addr.addr_bits_len == 0 &&
                                 addr.addr_bits_size <= 4 ?
          (1u << addr.addr_bits_size) : 0;
      pending_dma.id = addr.addr_bits_id;
      dma_lane = addr.addr_bits_addr & (CORAL_GEM5_DMA_DATA_BYTES - 1);
      if (pending_dma.size > CORAL_GEM5_DMA_DATA_BYTES ||
          dma_lane + pending_dma.size > CORAL_GEM5_DMA_DATA_BYTES) {
        pending_dma.size = 0;
      }
      dma_pending = true;
    });
    wrapper.RegisterDeferredWriteCallback(
        [this](const AxiAddr& addr, const AxiWData& data) {
          std::memset(&pending_dma, 0, sizeof(pending_dma));
          pending_dma.type = CORAL_GEM5_DMA_WRITE;
          pending_dma.addr = addr.addr_bits_addr;
          pending_dma.size = addr.addr_bits_len == 0 &&
                                     addr.addr_bits_size <= 4 ?
              (1u << addr.addr_bits_size) : 0;
          pending_dma.id = addr.addr_bits_id;
          dma_lane = addr.addr_bits_addr &
              (CORAL_GEM5_DMA_DATA_BYTES - 1);
          const auto* src =
              reinterpret_cast<const uint8_t*>(&data.write_data_bits_data[0]);
          if (pending_dma.size > CORAL_GEM5_DMA_DATA_BYTES ||
              dma_lane + pending_dma.size > CORAL_GEM5_DMA_DATA_BYTES) {
            pending_dma.size = 0;
          } else {
            for (uint32_t i = 0; i < pending_dma.size; ++i) {
              const uint32_t lane = dma_lane + i;
              if ((data.write_data_bits_strb & (1u << lane)) == 0) {
                pending_dma.size = 0;
                break;
              }
              pending_dma.data[i] = src[lane];
            }
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
  const bool stopped = handle->wrapper.WaitForTermination(cycles);
  if (handle->dma_pending) {
    return 2;
  }
  return stopped ? 1 : 0;
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
    AxiRData axi_response = {};
    if (!error) {
      auto* dst =
          reinterpret_cast<uint8_t*>(&axi_response.read_data_bits_data[0]);
      std::memcpy(dst + handle->dma_lane, data, size);
    }
    axi_response.read_data_bits_id = handle->pending_dma.id;
    axi_response.read_data_bits_resp = response;
    axi_response.read_data_bits_last = 1;
    handle->wrapper.QueueReadResponse(axi_response);
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
