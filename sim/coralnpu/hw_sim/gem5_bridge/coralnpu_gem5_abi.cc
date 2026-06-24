#include "hw_sim/gem5_bridge/coralnpu_gem5_abi.h"

#include <cstring>
#include <new>
#include <vector>

#include "hw_sim/core_mini_axi_wrapper.h"

struct coral_gem5_handle {
  VerilatedContext context;
  CoreMiniAxiWrapper wrapper;

  coral_gem5_handle() : context(), wrapper(&context) {
    wrapper.RegisterReadCallback([this](const AxiAddr& addr) {
      AxiRData response = {};
      const auto& mailbox = wrapper.mailbox();
      std::memcpy(&response.read_data_bits_data[0], mailbox.message,
                  sizeof(mailbox.message));
      response.read_data_bits_id = addr.addr_bits_id;
      response.read_data_bits_resp = 0;
      response.read_data_bits_last = 1;
      return response;
    });
    wrapper.RegisterWriteCallback(
        [this](const AxiAddr& addr, const AxiWData& data) {
          auto& mailbox = wrapper.mailbox();
          auto* dst = reinterpret_cast<uint8_t*>(mailbox.message);
          const auto* src =
              reinterpret_cast<const uint8_t*>(&data.write_data_bits_data[0]);
          for (size_t i = 0; i < sizeof(mailbox.message); ++i) {
            if (data.write_data_bits_strb & (1u << i)) {
              dst[i] = src[i];
            }
          }
          AxiWResp response = {};
          response.write_resp_bits_id = addr.addr_bits_id;
          response.write_resp_bits_resp = 0;
          return response;
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
  return handle->wrapper.WaitForTermination(cycles) ? 1 : 0;
}
