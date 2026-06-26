#ifndef HW_SIM_GEM5_BRIDGE_GEM5_DMA_REQUEST_BUILDER_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_DMA_REQUEST_BUILDER_H_

#include <cstring>
#include <vector>

#include "hw_sim/gem5_bridge/coralnpu_gem5_abi.h"
#include "hw_sim/hw_primitives.h"

inline bool
BuildGem5DmaReadRequest(const AxiAddr& addr,
                        coral_gem5_dma_request* request,
                        uint32_t* beat_size, uint32_t* beat_count)
{
  if (request == nullptr || beat_size == nullptr || beat_count == nullptr ||
      addr.addr_bits_size > 4 || addr.addr_bits_burst != 1) {
    return false;
  }

  const uint32_t bytes_per_beat = 1u << addr.addr_bits_size;
  const uint32_t beats = static_cast<uint32_t>(addr.addr_bits_len) + 1;
  const uint32_t size = bytes_per_beat * beats;
  const uint32_t page_offset = addr.addr_bits_addr & 0xfff;
  if (size > CORAL_GEM5_DMA_DATA_BYTES || page_offset + size > 4096) {
    return false;
  }
  for (uint32_t beat = 0; beat < beats; ++beat) {
    const uint32_t beat_addr =
        addr.addr_bits_addr + beat * bytes_per_beat;
    const uint32_t lane =
        beat_addr & (CORAL_GEM5_AXI_DATA_BYTES - 1);
    if (lane + bytes_per_beat > CORAL_GEM5_AXI_DATA_BYTES) {
      return false;
    }
  }

  std::memset(request, 0, sizeof(*request));
  request->type = CORAL_GEM5_DMA_READ;
  request->addr = addr.addr_bits_addr;
  request->size = size;
  request->id = addr.addr_bits_id;
  *beat_size = bytes_per_beat;
  *beat_count = beats;
  return true;
}

inline bool
BuildGem5DmaWriteRequest(const AxiAddr& addr,
                         const std::vector<AxiWData>& data,
                         coral_gem5_dma_request* request)
{
  if (request == nullptr || addr.addr_bits_size > 4 ||
      addr.addr_bits_burst != 1) {
    return false;
  }

  const uint32_t bytes_per_beat = 1u << addr.addr_bits_size;
  const uint32_t beats = static_cast<uint32_t>(addr.addr_bits_len) + 1;
  const uint32_t size = bytes_per_beat * beats;
  const uint32_t page_offset = addr.addr_bits_addr & 0xfff;
  if (data.size() != beats || size > CORAL_GEM5_DMA_DATA_BYTES ||
      page_offset + size > 4096) {
    return false;
  }

  std::memset(request, 0, sizeof(*request));
  request->type = CORAL_GEM5_DMA_WRITE;
  request->addr = addr.addr_bits_addr;
  request->size = size;
  request->id = addr.addr_bits_id;

  for (uint32_t beat = 0; beat < beats; ++beat) {
    const bool last = beat + 1 == beats;
    if (data[beat].write_data_bits_last != last) {
      return false;
    }
    const uint32_t beat_addr =
        addr.addr_bits_addr + beat * bytes_per_beat;
    const uint32_t lane =
        beat_addr & (CORAL_GEM5_AXI_DATA_BYTES - 1);
    if (lane + bytes_per_beat > CORAL_GEM5_AXI_DATA_BYTES) {
      return false;
    }

    const auto* source = reinterpret_cast<const uint8_t*>(
        &data[beat].write_data_bits_data[0]);
    for (uint32_t i = 0; i < bytes_per_beat; ++i) {
      const uint32_t byte_lane = lane + i;
      if ((data[beat].write_data_bits_strb & (1u << byte_lane)) == 0) {
        return false;
      }
      request->data[beat * bytes_per_beat + i] = source[byte_lane];
    }
  }
  return true;
}

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_DMA_REQUEST_BUILDER_H_
