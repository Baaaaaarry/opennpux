#ifndef HW_SIM_GEM5_BRIDGE_GEM5_AXI_MASTER_DRIVERS_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_AXI_MASTER_DRIVERS_H_

#include <functional>
#include <queue>
#include <vector>

#include "hw_sim/hw_primitives.h"

class Gem5AxiMasterReadDriver : Clock::Observer {
 public:
  Gem5AxiMasterReadDriver(
      Clock* clock, const uint8_t* addr_valid, const uint32_t* addr,
      const uint8_t* prot, const uint8_t* id, const uint8_t* len,
      const uint8_t* size, const uint8_t* burst, const uint8_t* lock,
      const uint8_t* cache, const uint8_t* qos, const uint8_t* region,
      uint8_t* addr_ready, uint8_t* data_valid, VlWide<4>* data,
      uint8_t* data_id, uint8_t* data_resp, uint8_t* data_last,
      const uint8_t* data_ready)
      : Clock::Observer(clock),
        addr_valid_(addr_valid),
        addr_(addr),
        prot_(prot),
        id_(id),
        len_(len),
        size_(size),
        burst_(burst),
        lock_(lock),
        cache_(cache),
        qos_(qos),
        region_(region),
        addr_ready_(addr_ready),
        data_valid_(data_valid),
        data_(data),
        data_id_(data_id),
        data_resp_(data_resp),
        data_last_(data_last),
        data_ready_(data_ready) {
    *addr_ready_ = !request_pending_;
  }

  void RegisterDeferredCallback(
      std::function<void(const AxiAddr&)> callback) {
    callback_ = callback;
  }

  void QueueResponse(const AxiRData& response) {
    responses_.push(response);
  }

  bool HasDeferredRequest() const { return request_pending_; }

 private:
  void OnFallingEdge() final {
    const bool addr_handshake = *addr_valid_ && *addr_ready_;
    const bool response_handshake = response_handshake_pending_;
    response_handshake_pending_ = false;

    if (response_handshake) {
      const bool last = responses_.front().read_data_bits_last;
      responses_.pop();
      if (last) {
        request_pending_ = false;
      }
    }

    if (!*addr_valid_) {
      request_seen_ = false;
    }
    *addr_ready_ = !request_pending_;

    *data_valid_ = !responses_.empty();
    if (!responses_.empty()) {
      *data_ = responses_.front().read_data_bits_data;
      *data_id_ = responses_.front().read_data_bits_id;
      *data_resp_ = responses_.front().read_data_bits_resp;
      *data_last_ = responses_.front().read_data_bits_last;
    }
    clock().Eval();
    response_handshake_pending_ = *data_valid_ && *data_ready_;

    if (addr_handshake && !request_seen_ && !request_pending_) {
      request_seen_ = true;
      request_.addr_bits_addr = *addr_;
      request_.addr_bits_prot = *prot_;
      request_.addr_bits_id = *id_;
      request_.addr_bits_len = *len_;
      request_.addr_bits_size = *size_;
      request_.addr_bits_burst = *burst_;
      request_.addr_bits_lock = *lock_;
      request_.addr_bits_cache = *cache_;
      request_.addr_bits_qos = *qos_;
      request_.addr_bits_region = *region_;
      request_pending_ = true;
      callback_(request_);
      *addr_ready_ = 0;
      clock().Eval();
    }
  }

  const uint8_t* const addr_valid_;
  const uint32_t* const addr_;
  const uint8_t* const prot_;
  const uint8_t* const id_;
  const uint8_t* const len_;
  const uint8_t* const size_;
  const uint8_t* const burst_;
  const uint8_t* const lock_;
  const uint8_t* const cache_;
  const uint8_t* const qos_;
  const uint8_t* const region_;
  uint8_t* const addr_ready_;
  uint8_t* const data_valid_;
  VlWide<4>* const data_;
  uint8_t* const data_id_;
  uint8_t* const data_resp_;
  uint8_t* const data_last_;
  const uint8_t* const data_ready_;

  std::queue<AxiRData> responses_;
  AxiAddr request_;
  std::function<void(const AxiAddr&)> callback_;
  bool request_pending_ = false;
  bool request_seen_ = false;
  bool response_handshake_pending_ = false;
};

class Gem5AxiMasterWriteDriver : Clock::Observer {
 public:
  Gem5AxiMasterWriteDriver(
      Clock* clock, const uint8_t* addr_valid, const uint32_t* addr,
      const uint8_t* prot, const uint8_t* id, const uint8_t* len,
      const uint8_t* size, const uint8_t* burst, const uint8_t* lock,
      const uint8_t* cache, const uint8_t* qos, const uint8_t* region,
      uint8_t* addr_ready, const uint8_t* data_valid,
      const VlWide<4>* data, const uint16_t* strb, const uint8_t* last,
      uint8_t* data_ready, uint8_t* resp_valid, uint8_t* resp_id,
      uint8_t* resp, const uint8_t* resp_ready)
      : Clock::Observer(clock),
        addr_valid_(addr_valid),
        addr_(addr),
        prot_(prot),
        id_(id),
        len_(len),
        size_(size),
        burst_(burst),
        lock_(lock),
        cache_(cache),
        qos_(qos),
        region_(region),
        addr_ready_(addr_ready),
        data_valid_(data_valid),
        data_(data),
        strb_(strb),
        last_(last),
        data_ready_(data_ready),
        resp_valid_(resp_valid),
        resp_id_(resp_id),
        resp_(resp),
        resp_ready_(resp_ready) {
    *addr_ready_ = 0;
    *data_ready_ = 0;
  }

  void RegisterDeferredCallback(
      std::function<void(const AxiAddr&, const std::vector<AxiWData>&)>
          callback) {
    callback_ = callback;
  }

  void QueueResponse(const AxiWResp& response) {
    responses_.push(response);
  }

  bool HasDeferredRequest() const { return request_pending_; }

 private:
  void OnFallingEdge() final {
    const bool addr_handshake = *addr_valid_ && *addr_ready_;
    const bool data_handshake = *data_valid_ && *data_ready_;
    const bool response_handshake = response_handshake_pending_;
    response_handshake_pending_ = false;

    if (response_handshake) {
      responses_.pop();
      request_pending_ = false;
      addr_captured_ = false;
      data_complete_ = false;
      request_submitted_ = false;
      request_data_.clear();
    }

    *resp_valid_ = !responses_.empty();
    if (!responses_.empty()) {
      *resp_id_ = responses_.front().write_resp_bits_id;
      *resp_ = responses_.front().write_resp_bits_resp;
    }
    clock().Eval();
    response_handshake_pending_ = *resp_valid_ && *resp_ready_;

    if (addr_handshake && !addr_captured_) {
      addr_captured_ = true;
      request_addr_.addr_bits_addr = *addr_;
      request_addr_.addr_bits_prot = *prot_;
      request_addr_.addr_bits_id = *id_;
      request_addr_.addr_bits_len = *len_;
      request_addr_.addr_bits_size = *size_;
      request_addr_.addr_bits_burst = *burst_;
      request_addr_.addr_bits_lock = *lock_;
      request_addr_.addr_bits_cache = *cache_;
      request_addr_.addr_bits_qos = *qos_;
      request_addr_.addr_bits_region = *region_;
    }

    if (data_handshake && !data_complete_) {
      AxiWData beat = {};
      beat.write_data_bits_data = *data_;
      beat.write_data_bits_strb = *strb_;
      beat.write_data_bits_last = *last_;
      request_data_.push_back(beat);
      data_complete_ = *last_;
    }

    if (addr_captured_ && data_complete_ && !request_submitted_) {
      request_submitted_ = true;
      request_pending_ = true;
      callback_(request_addr_, request_data_);
    }

    *addr_ready_ = !request_pending_ && !addr_captured_;
    *data_ready_ = !request_pending_ && !data_complete_ &&
                   request_data_.size() < 256;
    clock().Eval();
  }

  const uint8_t* const addr_valid_;
  const uint32_t* const addr_;
  const uint8_t* const prot_;
  const uint8_t* const id_;
  const uint8_t* const len_;
  const uint8_t* const size_;
  const uint8_t* const burst_;
  const uint8_t* const lock_;
  const uint8_t* const cache_;
  const uint8_t* const qos_;
  const uint8_t* const region_;
  uint8_t* const addr_ready_;
  const uint8_t* const data_valid_;
  const VlWide<4>* const data_;
  const uint16_t* const strb_;
  const uint8_t* const last_;
  uint8_t* const data_ready_;
  uint8_t* const resp_valid_;
  uint8_t* const resp_id_;
  uint8_t* const resp_;
  const uint8_t* const resp_ready_;

  std::queue<AxiWResp> responses_;
  AxiAddr request_addr_;
  std::vector<AxiWData> request_data_;
  std::function<void(const AxiAddr&, const std::vector<AxiWData>&)> callback_;
  bool request_pending_ = false;
  bool addr_captured_ = false;
  bool data_complete_ = false;
  bool request_submitted_ = false;
  bool response_handshake_pending_ = false;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_AXI_MASTER_DRIVERS_H_
