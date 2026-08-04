#ifndef HW_SIM_GEM5_BRIDGE_GEM5_AXI_MASTER_DRIVERS_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_AXI_MASTER_DRIVERS_H_

#include <algorithm>
#include <array>
#include <cstdio>
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

  // Dumps the driver-internal flags. Called from the bridge watchdog dump to
  // show whether a stall is a missing response or a lost request capture.
  void DumpState() const {
    std::fprintf(stderr,
                 "Coral AXI mrd driver request_pending=%u request_seen=%u "
                 "responses=%zu last_handshake_addr=0x%08x "
                 "handshakes=%llu\n",
                 request_pending_ ? 1u : 0u, request_seen_ ? 1u : 0u,
                 responses_.size(), last_handshake_addr_,
                 static_cast<unsigned long long>(handshake_count_));
    std::fprintf(stderr, "Coral AXI mrd driver recent_handshakes=");
    const size_t count =
        std::min<size_t>(handshake_history_cursor_, handshake_history_.size());
    for (size_t i = 0; i < count; ++i) {
      const size_t index = (handshake_history_cursor_ - count + i) %
                           handshake_history_.size();
      std::fprintf(stderr, "%s(addr=0x%08x,size=%u,len=%u,id=%u)",
                   i == 0 ? "" : ",",
                   handshake_history_[index].addr,
                   handshake_history_[index].size,
                   handshake_history_[index].len,
                   handshake_history_[index].id);
    }
    std::fprintf(stderr, "\n");
  }

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
      last_handshake_addr_ = *addr_;
      ++handshake_count_;
      HandshakeRecord& record =
          handshake_history_[handshake_history_cursor_ %
                             handshake_history_.size()];
      record.addr = *addr_;
      record.size = *size_;
      record.len = *len_;
      record.id = *id_;
      ++handshake_history_cursor_;
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
      // Do NOT drop addr_ready_ here. The capture above samples the handshake
      // half a cycle before the DUT evaluates it (the DUT samples ready at
      // the next rising edge). Dropping ready now would let this driver
      // consume an address the DUT never transferred, deadlocking the bus:
      // the driver would wait for read completion while the DUT keeps
      // addr_valid high waiting for addr_ready. addr_ready_ is recomputed as
      // !request_pending_ at the top of the next OnFallingEdge, which gives
      // the DUT one full rising edge with valid && ready to complete the
      // handshake.
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
  uint32_t last_handshake_addr_ = 0;
  uint64_t handshake_count_ = 0;
  struct HandshakeRecord {
    uint32_t addr;
    uint8_t size;
    uint8_t len;
    uint8_t id;
  };
  std::array<HandshakeRecord, 8> handshake_history_{};
  size_t handshake_history_cursor_ = 0;
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

  // Dumps the driver-internal flags. Called from the bridge watchdog dump to
  // show where in the capture/submit/response pipeline a write is stuck.
  void DumpState() const {
    std::fprintf(stderr,
                 "Coral AXI mwr driver request_pending=%u addr_captured=%u "
                 "data_complete=%u request_submitted=%u request_data=%zu "
                 "responses=%zu hold_addr_ready=%u hold_data_ready=%u\n",
                 request_pending_ ? 1u : 0u, addr_captured_ ? 1u : 0u,
                 data_complete_ ? 1u : 0u, request_submitted_ ? 1u : 0u,
                 request_data_.size(), responses_.size(),
                 hold_addr_ready_ ? 1u : 0u, hold_data_ready_ ? 1u : 0u);
  }

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
      // Keep addr_ready_ high for one more falling edge: this capture samples
      // the handshake half a cycle before the DUT evaluates it, so the DUT
      // must still see ready && valid at the next rising edge. Dropping
      // addr_ready_ now would consume an address the DUT never transferred
      // and deadlock the bus (driver waits for W data, DUT waits for
      // addr_ready).
      hold_addr_ready_ = true;
    }

    if (data_handshake && !data_complete_) {
      AxiWData beat = {};
      beat.write_data_bits_data = *data_;
      beat.write_data_bits_strb = *strb_;
      beat.write_data_bits_last = *last_;
      request_data_.push_back(beat);
      data_complete_ = *last_;
      if (*last_) {
        // Same deferred-ready discipline for the final W beat: keep
        // data_ready_ high until the DUT had its rising-edge handshake.
        hold_data_ready_ = true;
      }
    }

    if (addr_captured_ && data_complete_ && !request_submitted_) {
      request_submitted_ = true;
      request_pending_ = true;
      callback_(request_addr_, request_data_);
    }

    bool next_addr_ready = !request_pending_ && !addr_captured_;
    if (hold_addr_ready_) {
      next_addr_ready = true;
      hold_addr_ready_ = false;
    }
    *addr_ready_ = next_addr_ready;
    bool next_data_ready = !request_pending_ && !data_complete_ &&
                           request_data_.size() < 256;
    if (hold_data_ready_) {
      next_data_ready = true;
      hold_data_ready_ = false;
    }
    *data_ready_ = next_data_ready;
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
  // One-shot flags that delay the ready drop by one falling edge after a
  // capture, so the DUT always sees a full rising edge with valid && ready
  // before the driver consumes the transfer (see OnFallingEdge).
  bool hold_addr_ready_ = false;
  bool hold_data_ready_ = false;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_AXI_MASTER_DRIVERS_H_
