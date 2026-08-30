#ifndef HW_SIM_GEM5_BRIDGE_GEM5_CORE_MINI_AXI_WRAPPER_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_CORE_MINI_AXI_WRAPPER_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#ifdef CORAL_GEM5_RVV_HIGHMEM
#include "VRvvCoreMiniHighmemAxi.h"
using Gem5CoralAxiModel = VRvvCoreMiniHighmemAxi;
#else
#include "VCoreMiniAxi.h"
using Gem5CoralAxiModel = VCoreMiniAxi;
#endif
#include "hw_sim/gem5_bridge/gem5_axi_master_drivers.h"
#include "hw_sim/gem5_bridge/gem5_tmma_coprocessor.h"
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
                      &core_.io_axi_master_write_resp_ready),
        fault_seen_(false) {
    core_.io_xnpu_request_ready = 0;
    core_.io_xnpu_reject = 0;
    debug_last_pcs_.fill(0);
  }

  void Reset() {
    core_.io_xnpu_request_ready = 0;
    core_.io_xnpu_reject = 0;
    core_.io_aresetn = 1;
    context_->timeInc(1);
    core_.io_aresetn = 0;
    context_->timeInc(1);
    core_.io_aresetn = 1;
    context_->timeInc(1);
    fault_seen_ = false;
    debug_dispatch_count_ = 0;
    debug_last_dump_count_ = 0;
    debug_pc_cursor_ = 0;
    debug_last_pcs_.fill(0);
    debug_fetch_count_ = 0;
    debug_last_fetch_pc_ = 0;
    debug_retire_count_ = 0;
    debug_last_retire_dump_ = 0;
    debug_last_retire_pc_ = 0;
    debug_last_retire_inst_ = 0;
    debug_last_retire_trap_ = 0;
    debug_retire_cursor_ = 0;
    debug_last_retire_ring_dump_ = 0;
    debug_retire_ring_.fill(DebugRetireEntry{0, 0, 0});
  }

  bool PrepareXOpenNpu(Gem5XOpenNpuFunctionalCoprocessor* coprocessor,
                       uint32_t sequence_id,
                       Gem5TmmaSubmitResult* result) {
    core_.io_xnpu_request_ready = 0;
    core_.io_xnpu_reject = 0;
    core_.eval();
    if (!core_.io_xnpu_request_valid) {
      return false;
    }

    Gem5TmmaDispatchPacket packet;
    packet.instruction = core_.io_xnpu_request_bits_instruction;
    packet.pc = core_.io_xnpu_request_bits_pc;
    packet.rs1_value = core_.io_xnpu_request_bits_rs1Value;
    packet.rs2_value = core_.io_xnpu_request_bits_rs2Value;
    packet.rd_value = core_.io_xnpu_request_bits_rdValue;
    packet.mma_shape = core_.io_xnpu_request_bits_csr_mmaShape;
    packet.mma_data_type = core_.io_xnpu_request_bits_csr_mmaDataType;
    packet.tensor_shape = core_.io_xnpu_request_bits_csr_tensorShape;
    packet.tensor_data_type = core_.io_xnpu_request_bits_csr_tensorDataType;
    packet.scalar_param0 = core_.io_xnpu_request_bits_csr_scalarParam0;
    packet.quant_qzeros_address =
        core_.io_xnpu_request_bits_csr_quantQzerosAddress;
    packet.quant_scales_address =
        core_.io_xnpu_request_bits_csr_quantScalesAddress;
    packet.quant_g_idx_address =
        core_.io_xnpu_request_bits_csr_quantGIdxAddress;
    packet.quant_config = core_.io_xnpu_request_bits_csr_quantConfig;
    packet.quant_qweight_stride =
        core_.io_xnpu_request_bits_csr_quantQweightStride;
    packet.quant_qzeros_stride =
        core_.io_xnpu_request_bits_csr_quantQzerosStride;
    packet.quant_scales_stride =
        core_.io_xnpu_request_bits_csr_quantScalesStride;
    packet.quant_group_range =
        core_.io_xnpu_request_bits_csr_quantGroupRange;
    packet.tensor_aux_source_address =
        core_.io_xnpu_request_bits_csr_tensorAuxSourceAddress;
    packet.tensor_aux_destination_address =
        core_.io_xnpu_request_bits_csr_tensorAuxDestinationAddress;
    packet.attention_heads = core_.io_xnpu_request_bits_csr_attentionHeads;
    packet.attention_head_dim_flags =
        core_.io_xnpu_request_bits_csr_attentionHeadDimFlags;
    packet.attention_kv_length =
        core_.io_xnpu_request_bits_csr_attentionKvLength;
    packet.recurrent_heads = core_.io_xnpu_request_bits_csr_recurrentHeads;
    packet.recurrent_dims = core_.io_xnpu_request_bits_csr_recurrentDims;
    packet.recurrent_beta_address =
        core_.io_xnpu_request_bits_csr_recurrentBetaAddress;
    packet.recurrent_a_log_address =
        core_.io_xnpu_request_bits_csr_recurrentALogAddress;
    packet.recurrent_dt_bias_address =
        core_.io_xnpu_request_bits_csr_recurrentDtBiasAddress;
    packet.conv_input_hw = core_.io_xnpu_request_bits_csr_convInputHw;
    packet.conv_output_hw = core_.io_xnpu_request_bits_csr_convOutputHw;
    packet.conv_channels_groups =
        core_.io_xnpu_request_bits_csr_convChannelsGroups;
    packet.conv_kernel_hw = core_.io_xnpu_request_bits_csr_convKernelHw;
    packet.conv_stride_hw = core_.io_xnpu_request_bits_csr_convStrideHw;
    packet.conv_padding_tl = core_.io_xnpu_request_bits_csr_convPaddingTl;
    packet.conv_padding_br = core_.io_xnpu_request_bits_csr_convPaddingBr;
    packet.conv_dilation_hw = core_.io_xnpu_request_bits_csr_convDilationHw;
    packet.conv_bias_address = core_.io_xnpu_request_bits_csr_convBiasAddress;
    packet.csr_epoch = core_.io_xnpu_request_bits_csr_epoch;
    packet.sequence_id = sequence_id;

    *result = coprocessor->Classify(packet);
    core_.io_xnpu_request_ready =
        *result == Gem5TmmaSubmitResult::kAccepted;
    core_.io_xnpu_reject =
        *result == Gem5TmmaSubmitResult::kIllegalInstruction ||
        *result == Gem5TmmaSubmitResult::kInvalidCsrState;
    if (core_.io_xnpu_reject) {
      std::fprintf(stderr,
                   "Coral XOpenNPU reject pc=%#x inst=%#x epoch=%u "
                   "reason=%u\n",
                   packet.pc, packet.instruction, packet.csr_epoch,
                   static_cast<unsigned>(*result));
    }
    core_.eval();
    if (core_.io_xnpu_request_valid && core_.io_xnpu_request_ready) {
      *result = coprocessor->Submit(packet);
      if (*result == Gem5TmmaSubmitResult::kAccepted) {
        std::fprintf(
            stderr,
            "Coral XOpenNPU dispatch sequence=%u operation=%s pc=%#x "
            "inst=%#x epoch=%u rs1=%#x rs2=%#x rd=%#x\n",
            sequence_id,
            xopennpux::OperationName(
                xopennpux::DecodeOperation(packet.instruction)),
            packet.pc, packet.instruction, packet.csr_epoch,
            packet.rs1_value, packet.rs2_value, packet.rd_value);
      }
      return *result == Gem5TmmaSubmitResult::kAccepted;
    }
    return false;
  }

  void Step() {
    clock_.Step();
    fault_seen_ = fault_seen_ || core_.io_fault;
    SampleDebugDispatch();
  }

  // Records instruction-dispatch activity from the core debug ports so a
  // silent core (no AXI master traffic) can be classified as spinning
  // (dispatch PCs keep changing) vs. pipeline-stalled (no dispatch).
  void SampleDebugDispatch() {
#ifdef CORAL_GEM5_RVV_HIGHMEM
    const uint8_t fire[4] = {core_.io_debug_dispatch_0_instFire,
                             core_.io_debug_dispatch_1_instFire,
                             core_.io_debug_dispatch_2_instFire,
                             core_.io_debug_dispatch_3_instFire};
    const uint32_t pc[4] = {core_.io_debug_dispatch_0_instAddr,
                            core_.io_debug_dispatch_1_instAddr,
                            core_.io_debug_dispatch_2_instAddr,
                            core_.io_debug_dispatch_3_instAddr};
    for (int lane = 0; lane < 4; ++lane) {
      if (fire[lane]) {
        ++debug_dispatch_count_;
        debug_last_pcs_[debug_pc_cursor_ % debug_last_pcs_.size()] = pc[lane];
        ++debug_pc_cursor_;
      }
    }
    if (core_.io_debug_en & 1) {
      ++debug_fetch_count_;
      debug_last_fetch_pc_ = core_.io_debug_addr_0;
    }
    const uint8_t rb_valid[8] = {core_.io_debug_rb_inst_0_valid,
                                 core_.io_debug_rb_inst_1_valid,
                                 core_.io_debug_rb_inst_2_valid,
                                 core_.io_debug_rb_inst_3_valid,
                                 core_.io_debug_rb_inst_4_valid,
                                 core_.io_debug_rb_inst_5_valid,
                                 core_.io_debug_rb_inst_6_valid,
                                 core_.io_debug_rb_inst_7_valid};
    const uint32_t rb_pc[8] = {core_.io_debug_rb_inst_0_bits_pc,
                               core_.io_debug_rb_inst_1_bits_pc,
                               core_.io_debug_rb_inst_2_bits_pc,
                               core_.io_debug_rb_inst_3_bits_pc,
                               core_.io_debug_rb_inst_4_bits_pc,
                               core_.io_debug_rb_inst_5_bits_pc,
                               core_.io_debug_rb_inst_6_bits_pc,
                               core_.io_debug_rb_inst_7_bits_pc};
    const uint32_t rb_inst[8] = {core_.io_debug_rb_inst_0_bits_inst,
                                 core_.io_debug_rb_inst_1_bits_inst,
                                 core_.io_debug_rb_inst_2_bits_inst,
                                 core_.io_debug_rb_inst_3_bits_inst,
                                 core_.io_debug_rb_inst_4_bits_inst,
                                 core_.io_debug_rb_inst_5_bits_inst,
                                 core_.io_debug_rb_inst_6_bits_inst,
                                 core_.io_debug_rb_inst_7_bits_inst};
    const uint8_t rb_trap[8] = {core_.io_debug_rb_inst_0_bits_trap,
                                core_.io_debug_rb_inst_1_bits_trap,
                                core_.io_debug_rb_inst_2_bits_trap,
                                core_.io_debug_rb_inst_3_bits_trap,
                                core_.io_debug_rb_inst_4_bits_trap,
                                core_.io_debug_rb_inst_5_bits_trap,
                                core_.io_debug_rb_inst_6_bits_trap,
                                core_.io_debug_rb_inst_7_bits_trap};
    const uint32_t rb_data[8] = {core_.io_debug_rb_inst_0_bits_data[0],
                                 core_.io_debug_rb_inst_1_bits_data[0],
                                 core_.io_debug_rb_inst_2_bits_data[0],
                                 core_.io_debug_rb_inst_3_bits_data[0],
                                 core_.io_debug_rb_inst_4_bits_data[0],
                                 core_.io_debug_rb_inst_5_bits_data[0],
                                 core_.io_debug_rb_inst_6_bits_data[0],
                                 core_.io_debug_rb_inst_7_bits_data[0]};
    for (int slot = 0; slot < 8; ++slot) {
      if (rb_valid[slot]) {
        ++debug_retire_count_;
        debug_last_retire_pc_ = rb_pc[slot];
        debug_last_retire_inst_ = rb_inst[slot];
        debug_last_retire_trap_ = rb_trap[slot];
        DebugRetireEntry& entry =
            debug_retire_ring_[debug_retire_cursor_ %
                               debug_retire_ring_.size()];
        entry.pc = rb_pc[slot];
        entry.inst = rb_inst[slot];
        entry.data = rb_data[slot];
        ++debug_retire_cursor_;
      }
    }
#endif
  }

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

  // Dumps the live AXI master channel handshake levels. Used by the bridge
  // watchdog to diagnose a core stall: the stuck channel pair (valid/ready)
  // shows exactly where a transaction is not completing.
  void DumpMasterChannels(uint64_t rtl_cycles) const {
    std::fprintf(stderr,
                 "Coral AXI watchdog cycles=%llu "
                 "mrd[arvalid=%u arready=%u araddr=0x%08x "
                 "rvalid=%u rready=%u] "
                 "mwr[awvalid=%u awready=%u awaddr=0x%08x "
                 "wvalid=%u wready=%u bvalid=%u bready=%u] "
                 "deferred_request=%u halted=%u wfi=%u fault=%u\n",
                 static_cast<unsigned long long>(rtl_cycles),
                 core_.io_axi_master_read_addr_valid,
                 core_.io_axi_master_read_addr_ready,
                 core_.io_axi_master_read_addr_bits_addr,
                 core_.io_axi_master_read_data_valid,
                 core_.io_axi_master_read_data_ready,
                 core_.io_axi_master_write_addr_valid,
                 core_.io_axi_master_write_addr_ready,
                 core_.io_axi_master_write_addr_bits_addr,
                 core_.io_axi_master_write_data_valid,
                 core_.io_axi_master_write_data_ready,
                 core_.io_axi_master_write_resp_valid,
                 core_.io_axi_master_write_resp_ready,
                 HasDeferredRequest() ? 1u : 0u,
                 core_.io_halted ? 1u : 0u, core_.io_wfi ? 1u : 0u,
                 core_.io_fault ? 1u : 0u);
    std::fprintf(stderr,
                 "Coral core debug dispatch_count=%llu (delta=%llu) "
                 "recent_pcs=",
                 static_cast<unsigned long long>(debug_dispatch_count_),
                 static_cast<unsigned long long>(debug_dispatch_count_ -
                                                 debug_last_dump_count_));
    debug_last_dump_count_ = debug_dispatch_count_;
    const size_t count = std::min<size_t>(debug_pc_cursor_,
                                          debug_last_pcs_.size());
    for (size_t i = 0; i < count; ++i) {
      const size_t index =
          (debug_pc_cursor_ - count + i) % debug_last_pcs_.size();
      std::fprintf(stderr, "%s0x%08x", i == 0 ? "" : ",",
                   debug_last_pcs_[index]);
    }
    std::fprintf(stderr, "\n");
    std::fprintf(stderr,
                 "Coral core debug fetch_count=%llu last_fetch_pc=0x%08x "
                 "retire_count=%llu (delta=%llu) last_retire_pc=0x%08x "
                 "inst=0x%08x trap=%u\n",
                 static_cast<unsigned long long>(debug_fetch_count_),
                 debug_last_fetch_pc_,
                 static_cast<unsigned long long>(debug_retire_count_),
                 static_cast<unsigned long long>(debug_retire_count_ -
                                                 debug_last_retire_dump_),
                 debug_last_retire_pc_, debug_last_retire_inst_,
                 debug_last_retire_trap_);
    debug_last_retire_dump_ = debug_retire_count_;
    if (debug_retire_cursor_ != debug_last_retire_ring_dump_) {
      debug_last_retire_ring_dump_ = debug_retire_cursor_;
      const size_t count = std::min<size_t>(debug_retire_cursor_,
                                            debug_retire_ring_.size());
      std::fprintf(stderr, "Coral core debug retire_ring (oldest first):\n");
      for (size_t i = 0; i < count; ++i) {
        const size_t index =
            (debug_retire_cursor_ - count + i) % debug_retire_ring_.size();
        const DebugRetireEntry& entry = debug_retire_ring_[index];
        std::fprintf(stderr, "  pc=0x%08x inst=0x%08x data=0x%08x\n",
                     entry.pc, entry.inst, entry.data);
      }
    }
    master_read_.DumpState();
    master_write_.DumpState();
    std::fflush(stderr);
  }

  bool IsHalted() const { return core_.io_halted; }
  bool IsWfi() const { return core_.io_wfi; }
  bool HasFault() const { return fault_seen_; }

 private:
  VerilatedContext* const context_;
  Gem5CoralAxiModel core_;
  Clock clock_;
  AxiSlaveWriteDriver slave_write_;
  AxiSlaveReadDriver slave_read_;
  Gem5AxiMasterReadDriver master_read_;
  Gem5AxiMasterWriteDriver master_write_;
  bool fault_seen_;
  uint64_t debug_dispatch_count_ = 0;
  mutable uint64_t debug_last_dump_count_ = 0;
  uint64_t debug_pc_cursor_ = 0;
  std::array<uint32_t, 32> debug_last_pcs_;
  uint64_t debug_fetch_count_ = 0;
  uint32_t debug_last_fetch_pc_ = 0;
  uint64_t debug_retire_count_ = 0;
  mutable uint64_t debug_last_retire_dump_ = 0;
  uint32_t debug_last_retire_pc_ = 0;
  uint32_t debug_last_retire_inst_ = 0;
  uint8_t debug_last_retire_trap_ = 0;
  struct DebugRetireEntry {
    uint32_t pc;
    uint32_t inst;
    uint32_t data;
  };
  uint64_t debug_retire_cursor_ = 0;
  mutable uint64_t debug_last_retire_ring_dump_ = 0;
  std::array<DebugRetireEntry, 128> debug_retire_ring_;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_CORE_MINI_AXI_WRAPPER_H_
