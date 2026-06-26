#ifndef HW_SIM_GEM5_BRIDGE_GEM5_CORE_MINI_AXI_WRAPPER_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_CORE_MINI_AXI_WRAPPER_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "VCoreMiniAxi.h"
#include "hw_sim/gem5_bridge/gem5_axi_master_drivers.h"
#include "hw_sim/hw_primitives.h"

class Gem5CoreMiniAxiWrapper {
 public:
  explicit Gem5CoreMiniAxiWrapper(VerilatedContext* context)
      : context_(context),
        core_(context, "core"),
        clock_(context, &core_.io_aclk, &core_),
        slave_write_(&clock_, &core_.io_axi_slave_write_addr_valid,
                     &core_.io_axi_slave_write_addr_bits_addr,
                     &core_.io_axi_slave_write_addr_bits_prot,
                     &core_.io_axi_slave_write_addr_bits_id,
                     &core_.io_axi_slave_write_addr_bits_len,
                     &core_.io_axi_slave_write_addr_bits_size,
                     &core_.io_axi_slave_write_addr_bits_burst,
                     &core_.io_axi_slave_write_addr_bits_lock,
                     &core_.io_axi_slave_write_addr_bits_cache,
                     &core_.io_axi_slave_write_addr_bits_qos,
                     &core_.io_axi_slave_write_addr_bits_region,
                     &core_.io_axi_slave_write_addr_ready,
                     &core_.io_axi_slave_write_data_valid,
                     &core_.io_axi_slave_write_data_bits_data,
                     &core_.io_axi_slave_write_data_bits_strb,
                     &core_.io_axi_slave_write_data_bits_last,
                     &core_.io_axi_slave_write_data_ready,
                     &core_.io_axi_slave_write_resp_valid,
                     &core_.io_axi_slave_write_resp_bits_id,
                     &core_.io_axi_slave_write_resp_bits_resp,
                     &core_.io_axi_slave_write_resp_ready),
        slave_read_(&clock_, &core_.io_axi_slave_read_addr_valid,
                    &core_.io_axi_slave_read_addr_bits_addr,
                    &core_.io_axi_slave_read_addr_bits_prot,
                    &core_.io_axi_slave_read_addr_bits_id,
                    &core_.io_axi_slave_read_addr_bits_len,
                    &core_.io_axi_slave_read_addr_bits_size,
                    &core_.io_axi_slave_read_addr_bits_burst,
                    &core_.io_axi_slave_read_addr_bits_lock,
                    &core_.io_axi_slave_read_addr_bits_cache,
                    &core_.io_axi_slave_read_addr_bits_qos,
                    &core_.io_axi_slave_read_addr_bits_region,
                    &core_.io_axi_slave_read_addr_ready,
                    &core_.io_axi_slave_read_data_valid,
                    &core_.io_axi_slave_read_data_bits_data,
                    &core_.io_axi_slave_read_data_bits_id,
                    &core_.io_axi_slave_read_data_bits_resp,
                    &core_.io_axi_slave_read_data_bits_last,
                    &core_.io_axi_slave_read_data_ready),
        master_read_(&clock_, &core_.io_axi_master_read_addr_valid,
                     &core_.io_axi_master_read_addr_bits_addr,
                     &core_.io_axi_master_read_addr_bits_prot,
                     &core_.io_axi_master_read_addr_bits_id,
                     &core_.io_axi_master_read_addr_bits_len,
                     &core_.io_axi_master_read_addr_bits_size,
                     &core_.io_axi_master_read_addr_bits_burst,
                     &core_.io_axi_master_read_addr_bits_lock,
                     &core_.io_axi_master_read_addr_bits_cache,
                     &core_.io_axi_master_read_addr_bits_qos,
                     &core_.io_axi_master_read_addr_bits_region,
                     &core_.io_axi_master_read_addr_ready,
                     &core_.io_axi_master_read_data_valid,
                     &core_.io_axi_master_read_data_bits_data,
                     &core_.io_axi_master_read_data_bits_id,
                     &core_.io_axi_master_read_data_bits_resp,
                     &core_.io_axi_master_read_data_bits_last,
                     &core_.io_axi_master_read_data_ready),
        master_write_(&clock_, &core_.io_axi_master_write_addr_valid,
                      &core_.io_axi_master_write_addr_bits_addr,
                      &core_.io_axi_master_write_addr_bits_prot,
                      &core_.io_axi_master_write_addr_bits_id,
                      &core_.io_axi_master_write_addr_bits_len,
                      &core_.io_axi_master_write_addr_bits_size,
                      &core_.io_axi_master_write_addr_bits_burst,
                      &core_.io_axi_master_write_addr_bits_lock,
                      &core_.io_axi_master_write_addr_bits_cache,
                      &core_.io_axi_master_write_addr_bits_qos,
                      &core_.io_axi_master_write_addr_bits_region,
                      &core_.io_axi_master_write_addr_ready,
                      &core_.io_axi_master_write_data_valid,
                      &core_.io_axi_master_write_data_bits_data,
                      &core_.io_axi_master_write_data_bits_strb,
                      &core_.io_axi_master_write_data_bits_last,
                      &core_.io_axi_master_write_data_ready,
                      &core_.io_axi_master_write_resp_valid,
                      &core_.io_axi_master_write_resp_bits_id,
                      &core_.io_axi_master_write_resp_bits_resp,
                      &core_.io_axi_master_write_resp_ready) {}

  void Reset() {
    core_.io_aresetn = 1;
    context_->timeInc(1);
    core_.io_aresetn = 0;
    context_->timeInc(1);
    core_.io_aresetn = 1;
    context_->timeInc(1);
  }

  void Step() { clock_.Step(); }

  void Write(uint32_t addr, uint32_t len, const char* data) {
    auto remaining =
        absl::Span<const uint8_t>(reinterpret_cast<const uint8_t*>(data), len);
    while (!remaining.empty()) {
      const uint32_t boundary = 4096 - addr % 4096;
      const uint32_t bytes =
          std::min(static_cast<uint32_t>(remaining.size()), boundary);
      auto transaction =
          slave_write_.WriteTransaction(0, addr, remaining.subspan(0, bytes));
      while (!*transaction) {
        Step();
      }
      remaining.remove_prefix(bytes);
      addr += bytes;
    }
  }

  std::vector<uint8_t> Read(uint32_t addr, uint32_t len) {
    std::vector<uint8_t> result;
    result.reserve(len);
    while (result.size() < len) {
      const uint32_t boundary = 4096 - addr % 4096;
      const uint32_t bytes = std::min(
          len - static_cast<uint32_t>(result.size()), boundary);
      auto transaction = slave_read_.ReadTransaction(0, addr, bytes);
      while (!transaction->finished) {
        Step();
      }
      result.insert(result.end(), transaction->data.begin(),
                    transaction->data.end());
      addr += bytes;
    }
    return result;
  }

  void RegisterDeferredReadCallback(
      std::function<void(const AxiAddr&)> callback) {
    master_read_.RegisterDeferredCallback(callback);
  }

  void RegisterDeferredWriteCallback(
      std::function<void(const AxiAddr&, const std::vector<AxiWData>&)>
          callback) {
    master_write_.RegisterDeferredCallback(callback);
  }

  void QueueReadResponse(const AxiRData& response) {
    master_read_.QueueResponse(response);
  }

  void QueueWriteResponse(const AxiWResp& response) {
    master_write_.QueueResponse(response);
  }

  bool HasDeferredRequest() const {
    return master_read_.HasDeferredRequest() ||
           master_write_.HasDeferredRequest();
  }

  bool IsHalted() const { return core_.io_halted; }
  bool IsWfi() const { return core_.io_wfi; }

 private:
  VerilatedContext* const context_;
  VCoreMiniAxi core_;
  Clock clock_;
  AxiSlaveWriteDriver slave_write_;
  AxiSlaveReadDriver slave_read_;
  Gem5AxiMasterReadDriver master_read_;
  Gem5AxiMasterWriteDriver master_write_;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_CORE_MINI_AXI_WRAPPER_H_
